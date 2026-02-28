"""
SNAP-C1 V5: Resonance Block (Component 2)
==========================================
Dual-path local + global attention blocks.

Path A: Sliding Window Attention — O(n × window) local context
Path B: Global Linear Attention  — O(n × d²) global pattern recognition
        ELU+1 feature map (no softmax, no complex numbers)
        Every position attends to every other position

Combined via gated fusion + SwiGLU FFN + RMSNorm.

NOTE: The original design used FFT spectral mixing. DirectML rejects
ComplexFloat tensors entirely ("Invalid or unsupported data type ComplexFloat").
Global Linear Attention provides the same architectural purpose (global mixing)
using only real-valued matmul ops.

DirectML safety:
  - F.scaled_dot_product_attention ✓
  - F.elu + matmul (linear attention) ✓
  - StableSigmoid (tanh-based) ✓
  - All matmul-based ✓
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_core.utils.dml_ops import RMSNorm, SwiGLU, stable_sigmoid


# ---------------------------------------------------------------------------
# Path A: Sliding Window Attention
# ---------------------------------------------------------------------------
class SlidingWindowAttention(nn.Module):
    """
    Causal attention restricted to a local window.
    Each token attends to at most `window_size` previous tokens.
    O(n × window) instead of O(n²).

    Code has strong local dependencies: if→else, variable decl→use.
    A 128-token window covers ~50 lines of code.
    """

    def __init__(self, d_model: int = 1024, n_heads: int = 8, window_size: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _build_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build causal sliding window mask. True = attended, False = masked.
        positions[i,j] = i - j  (how far back j is from i)
        Causal: i >= j (attend only to past/present)
        Window: i - j < window_size
        """
        rows = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
        cols = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        diff = rows - cols  # [T, T] — diff[i,j] = i - j
        mask = (diff >= 0) & (diff < self.window_size)
        return mask  # [T, T] bool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3*d_model]
        Q, K, V = qkv.chunk(3, dim=-1)  # each [B, T, d_model]

        # Reshape for multi-head: [B, n_heads, T, head_dim]
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Build causal sliding window mask (cached per sequence length)
        if not hasattr(self, '_cached_mask_T') or self._cached_mask_T != T:
            mask = self._build_window_mask(T, x.device)  # [T, T] bool
            attn_mask = torch.zeros(T, T, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(~mask, float('-inf'))
            self._cached_mask = attn_mask
            self._cached_mask_T = T
        attn_mask = self._cached_mask.to(device=x.device, dtype=x.dtype)

        # Scaled dot-product attention with mask
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale
        )  # [B, n_heads, T, head_dim]

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# Path B: Global Linear Attention (replaces FFT — DirectML rejects ComplexFloat)
# ---------------------------------------------------------------------------
class GlobalLinearAttention(nn.Module):
    """
    Global mixing via linear attention with ReLU+1 feature map.

    Standard attention: softmax(QK^T/√d) · V  →  O(T² × d)
    Linear attention:   φ(Q) · (φ(K)^T · V)   →  O(T × d²)

    Supports both CAUSAL and BIDIRECTIONAL modes:
    - Causal (pretraining): chunk-wise cumulative state, only sees past tokens
    - Bidirectional (agent mode): full K^T·V aggregation

    Feature map φ(x) = ReLU(x) + 1 ensures all values are positive.
    (ReLU used instead of ELU because aten::elu.out falls back to CPU on DirectML)

    DirectML safe: pure matmul + ReLU, no complex numbers, no scatter.
    """

    def __init__(self, d_model: int = 1024, n_heads: int = 4, chunk_size: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU+1 feature map: ensures positive values, fully GPU-native."""
        return F.relu(x) + 1.0

    def _bidirectional(self, Q, K, V):
        """Full bidirectional linear attention. O(T × d²)."""
        KV = torch.matmul(K.transpose(-2, -1), V)  # [B, H, d_h, d_h]
        QKV = torch.matmul(Q, KV)                  # [B, H, T, d_h]
        K_sum = K.sum(dim=-2, keepdim=True)         # [B, H, 1, d_h]
        Z = torch.matmul(Q, K_sum.transpose(-2, -1))  # [B, H, T, 1]
        return QKV / (Z + 1e-6)

    def _causal_chunked(self, Q, K, V):
        """Causal linear attention via chunk-wise cumulative state.
        Each chunk computes internal causal attention + carries state from prior chunks.
        O(T × d²) compute, O(chunk_size × d²) peak memory per chunk.
        """
        B, H, T, d_h = Q.shape
        C = self.chunk_size
        out_chunks = []

        # Running state from previous chunks
        S = torch.zeros(B, H, d_h, d_h, device=Q.device, dtype=Q.dtype)
        z = torch.zeros(B, H, 1, d_h, device=Q.device, dtype=Q.dtype)

        for t_start in range(0, T, C):
            t_end = min(t_start + C, T)
            Q_c = Q[:, :, t_start:t_end, :]  # [B, H, c, d_h]
            K_c = K[:, :, t_start:t_end, :]
            V_c = V[:, :, t_start:t_end, :]
            c = t_end - t_start

            # Within-chunk causal: build lower-triangular mask for small c×c
            # causal_mask[i,j] = 1 if i >= j, else 0
            causal = torch.ones(c, c, device=Q.device, dtype=Q.dtype).tril()  # [c, c]

            # Within-chunk attention: Q_c @ K_c^T masked → × V_c
            # attn_scores[i,j] = Q_c[i] · K_c[j] for j <= i
            attn_intra = torch.matmul(Q_c, K_c.transpose(-2, -1))  # [B, H, c, c]
            attn_intra = attn_intra * causal.unsqueeze(0).unsqueeze(0)  # mask future
            out_intra = torch.matmul(attn_intra, V_c)  # [B, H, c, d_h]

            # Normalizer for intra-chunk
            z_intra = attn_intra.sum(dim=-1, keepdim=True)  # [B, H, c, 1]

            # Cross-chunk: Q_c @ S (accumulated K^T·V from all previous chunks)
            out_cross = torch.matmul(Q_c, S)  # [B, H, c, d_h]
            z_cross = torch.matmul(Q_c, z.transpose(-2, -1))  # [B, H, c, 1]

            # Combine
            out_c = (out_intra + out_cross) / (z_intra + z_cross + 1e-6)
            out_chunks.append(out_c)

            # Update running state with this chunk
            S = S + torch.matmul(K_c.transpose(-2, -1), V_c)  # [B, H, d_h, d_h]
            z = z + K_c.sum(dim=-2, keepdim=True)              # [B, H, 1, d_h]

        return torch.cat(out_chunks, dim=2)  # [B, H, T, d_h]

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            causal: if True, each position only sees past positions (for pretraining)
                    if False, full bidirectional (for agent mode)
        Returns:
            [B, T, d_model]
        """
        B, T, D = x.shape

        # Project to Q, K, V with positive feature map
        Q = self._feature_map(self.q_proj(x))  # [B, T, D] — positive
        K = self._feature_map(self.k_proj(x))  # [B, T, D] — positive
        V = self.v_proj(x)                      # [B, T, D]

        # Reshape for multi-head: [B, H, T, d_h]
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if causal:
            out = self._causal_chunked(Q, K, V)
        else:
            out = self._bidirectional(Q, K, V)

        # Merge heads: [B, H, T, d_h] → [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Full Resonance Block: dual-path + gated fusion + FFN
# ---------------------------------------------------------------------------
class ResonanceBlock(nn.Module):
    """
    One V5 Resonance Block.

    Input → RMSNorm → [Path A (local) || Path B (global)]
         → Gated Fusion → Residual
         → RMSNorm → SwiGLU FFN → Residual
         → Output

    Both paths run in parallel. The gate learns to weight local vs global
    contributions per-position (expect ~60/40 split for code).
    """

    def __init__(self, d_model: int = 1024, n_heads: int = 8,
                 window_size: int = 128, d_ff: int = None,
                 max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4

        # Normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Path A: Local sliding window attention
        self.local_attn = SlidingWindowAttention(
            d_model=d_model, n_heads=n_heads,
            window_size=window_size, dropout=dropout
        )

        # Path B: Global linear attention (replaces FFT spectral mixing)
        self.spectral = GlobalLinearAttention(
            d_model=d_model, n_heads=n_heads
        )

        # Gated Fusion: learns to combine local + global per-position
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # Feed-forward
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            causal: if True, both paths use causal masking (for pretraining)
        Returns:
            [B, T, d_model]
        """
        # Pre-norm
        normed = self.norm1(x)

        # Dual path
        local_out = self.local_attn(normed)           # [B, T, d] — always causal
        global_out = self.spectral(normed, causal=causal)  # [B, T, d]

        # Gated fusion
        combined = torch.cat([local_out, global_out], dim=-1)  # [B, T, 2d]
        gate = stable_sigmoid(self.gate_proj(combined))  # [B, T, d]
        fused = gate * local_out + (1 - gate) * global_out

        # Residual connection
        x = x + fused

        # FFN with pre-norm + residual
        x = x + self.ffn(self.norm2(x))

        return x


class ResonanceStack(nn.Module):
    """Stack of N Resonance Blocks with optional gradient checkpointing."""

    def __init__(self, n_blocks: int = 8, d_model: int = 1024,
                 n_heads: int = 8, window_size: int = 128,
                 d_ff: int = None, max_seq_len: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.use_checkpoint = False
        self.blocks = nn.ModuleList([
            ResonanceBlock(
                d_model=d_model, n_heads=n_heads,
                window_size=window_size, d_ff=d_ff,
                max_seq_len=max_seq_len, dropout=dropout
            )
            for _ in range(n_blocks)
        ])
        self.final_norm = RMSNorm(d_model)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to trade compute for memory (~40% savings)."""
        self.use_checkpoint = True

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint as ckpt_fn
            for block in self.blocks:
                x = ckpt_fn(block, x, causal, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, causal=causal)
        return self.final_norm(x)
