"""
SNAP-C1 V6: Resonance Block with Dynamic Skip
==============================================
COMBINES V5's dual-path attention with V6's dynamic layer skipping.

V5 Resonance Block:
- Path A: Sliding Window Attention (local, O(n × window))
- Path B: Global Linear Attention (global, O(n × d²))
- Gated fusion

V6 Addition:
- Skip Router: predicts if layer is redundant
- Hard skip: identity function when redundant → 2x speedup
- Training: stochastic skip
- Inference: deterministic skip

V6.5 (Modern Attention):
- Path A: SlidingWindowGQA with QK-Norm + Grouped Query Attention
- Path B: DeltaNetAttention (linear attention with recurrences)
- MTP: Multi-Token Prediction head for speculative decoding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_core.architecture.dml_ops import RMSNorm, stable_sigmoid
from v6_core.architecture.advanced_attention import (
    SlidingWindowGQA,
    DeltaNetAttention,
    MultiTokenPrediction,
)


# ---------------------------------------------------------------------------
# Path A: Sliding Window Attention with GQA (UPGRADED from V6.5)
# ---------------------------------------------------------------------------
# NOW USES: SlidingWindowGQA with QK-Norm + Grouped Query Attention


# ---------------------------------------------------------------------------
# Path B: DeltaNetAttention (UPGRADED from V6.5)
# ---------------------------------------------------------------------------
# NOW USES: DeltaNet with linear attention + recurrences


# ---------------------------------------------------------------------------
# V6: Skip Router (NEW)
# ---------------------------------------------------------------------------
class SkipRouter(nn.Module):
    """
    Tiny MLP that predicts skip probability for a layer.
    Very small params - adds almost nothing to model size.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 8, bias=False),
            nn.GELU(),
            nn.Linear(d_model // 8, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_feat = x.mean(dim=1, keepdim=True)
        return self.net(global_feat)


# ---------------------------------------------------------------------------
# Full V6 Resonance Block with Dynamic Skip
# ---------------------------------------------------------------------------
class ResonanceBlock(nn.Module):
    """
    V6 Resonance Block with Dynamic Skip.

    Input → [Skip? Identity] → RMSNorm → [Local || Global] → Gate → FFN → Output

    Skip behavior:
    - Training: stochastic (torch.bernoulli)
    - Inference: deterministic (prob < 0.5 → skip)
    """
    def __init__(self, d_model: int = 1024, n_heads: int = 8,
                 window_size: int = 128, d_ff: int = None,
                 max_seq_len: int = 2048, dropout: float = 0.0,
                 use_skip: bool = True):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.use_skip = use_skip

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Skip router
        self.skip_router = SkipRouter(d_model) if use_skip else None

        # Path A: Local sliding window attention with GQA + QK-Norm (V6.5)
        n_kv_heads = max(1, n_heads // 8)
        self.local_attn = SlidingWindowGQA(
            d_model=d_model, n_heads=n_heads,
            window_size=window_size, n_kv_heads=n_kv_heads
        )

        # Path B: DeltaNetAttention for global context (V6.5)
        self.spectral = DeltaNetAttention(d_model=d_model, n_heads=n_heads)

        # Gated Fusion
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # FFN (SwiGLU)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        residual = x

        # Predict skip probability
        if self.use_skip:
            skip_prob = self.skip_router(x)
            
            if self.training:
                skip = torch.bernoulli(1 - skip_prob).bool()
            else:
                skip = skip_prob < 0.5
            
            if skip.all():
                return x  # All samples skip this layer

        # Pre-norm
        normed = self.norm1(x)

        # Dual path attention
        local_out = self.local_attn(normed)
        global_out = self.spectral(normed)

        # Gated fusion
        combined = torch.cat([local_out, global_out], dim=-1)
        gate = stable_sigmoid(self.gate_proj(combined))
        fused = gate * local_out + (1 - gate) * global_out

        # First residual
        x = x + fused

        # FFN with pre-norm + second residual
        ffn_gate = F.silu(self.w_gate(self.norm2(x)))
        ffn_up = self.w_up(self.norm2(x))
        ffn_out = self.w_down(ffn_gate * ffn_up)
        
        x = x + self.dropout(ffn_out)

        # Apply skip if needed
        if self.use_skip:
            skip_3d = skip.expand(-1, x.shape[1], x.shape[2])  # [B, T, D]
            x = torch.where(skip_3d, residual, x)

        return x


class ResonanceStack(nn.Module):
    """Stack of N Resonance Blocks with optional gradient checkpointing."""

    def __init__(self, n_blocks: int = 8, d_model: int = 1024,
                 n_heads: int = 8, window_size: int = 128,
                 d_ff: int = None, max_seq_len: int = 2048,
                 dropout: float = 0.0, use_skip: bool = True):
        super().__init__()
        self.use_checkpoint = False
        self.use_skip = use_skip

        self.blocks = nn.ModuleList([
            ResonanceBlock(
                d_model=d_model, n_heads=n_heads,
                window_size=window_size, d_ff=d_ff,
                max_seq_len=max_seq_len, dropout=dropout,
                use_skip=use_skip
            )
            for _ in range(n_blocks)
        ])
        self.final_norm = RMSNorm(d_model)

    def enable_gradient_checkpointing(self):
        self.use_checkpoint = True

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        total_skipped = 0

        if self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint as ckpt_fn
            for block in self.blocks:
                x = ckpt_fn(block, x, causal, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, causal=causal)

        return self.final_norm(x)
