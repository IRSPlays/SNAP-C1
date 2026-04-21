"""
Nexus-r V1: Foundational Layers
================================
Standard PyTorch primitives. No fused GRU.

Based on:
- Samsung TRM (layers.py) — CastedLinear, RoPE, SwiGLU, RMSNorm
- Moonshot AttnRes — Block Attention Residuals
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def scaled_dot_product_attention_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run SDPA with grouped-query attention when query and KV head counts differ."""
    if q.size(1) == k.size(1):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    if k.size(1) != v.size(1):
        raise ValueError("Key and value head counts must match for grouped attention")
    if q.size(1) % k.size(1) != 0:
        raise ValueError("Query head count must be divisible by KV head count")

    try:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)
    except TypeError:
        rep = q.size(1) // k.size(1)
        k = k.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(q.size(0), q.size(1), q.size(2), q.size(3))
        v = v.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(q.size(0), q.size(1), q.size(2), q.size(3))
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

# ============================================================================
# INIT UTILITIES
# ============================================================================

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal init (JAX-style, mathematically correct)."""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a, b = math.erf(-2.0 / sqrt2), math.erf(2.0 / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * 4.0)  # lower=-2
            pdf_l = c * math.exp(-0.5 * 4.0)  # upper=2
            comp_std = std / math.sqrt(1 - (2.0 * pdf_u - (-2.0) * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b).erfinv_().mul_(sqrt2 * comp_std)
            tensor.clip_(-2 * comp_std, 2 * comp_std)
    return tensor


# ============================================================================
# RMS NORM (no learnable params — weight-free for recursive reuse)
# ============================================================================

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype)


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.eps) * self.weight


# ============================================================================
# CASTED LINEAR (TRM-style, truncated normal init)
# ============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_features, in_features), std=1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


# ============================================================================
# ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. q,k: [B, T, H, D]. cos,sin: [T, D]."""
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    cos = cos.unsqueeze(-2)  # [T, 1, D]
    sin = sin.unsqueeze(-2)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(orig_dtype), k_out.to(orig_dtype)


# ============================================================================
# SwiGLU FFN
# ============================================================================

def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        inter = _find_multiple(round(expansion * d_model * 2 / 3), 256)
        self.gate_up = CastedLinear(d_model, inter * 2)
        self.down = CastedLinear(inter, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.down(F.silu(gate) * up))


# ============================================================================
# MULTI-HEAD ATTENTION (standard, via SDPA)
# ============================================================================

class Attention(nn.Module):
    """Standard multi-head attention using F.scaled_dot_product_attention."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads

        self.qkv = CastedLinear(d_model, (n_heads + 2 * self.n_kv_heads) * self.head_dim)
        self.out = CastedLinear(n_heads * self.head_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, self.n_heads + 2 * self.n_kv_heads, self.head_dim)
        q = qkv[:, :, :self.n_heads]
        k = qkv[:, :, self.n_heads:self.n_heads + self.n_kv_heads]
        v = qkv[:, :, self.n_heads + self.n_kv_heads:]

        if cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, *cos_sin)

        # BHSD for SDPA
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = scaled_dot_product_attention_gqa(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.drop(self.out(out))


# ============================================================================
# BLOCK ATTENTION RESIDUALS (Moonshot AttnRes — drop-in residual upgrade)
# ============================================================================

class BlockAttnRes(nn.Module):
    """
    Attention Residuals: replaces h = h + layer(h) with
    h = softmax_over_depth(block_outputs) weighted sum.
    One learned pseudo-query per application point.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Parameter(trunc_normal_init_(torch.empty(d_model), std=0.02))
        self.norm = RMSNorm(d_model)

    def forward(self, blocks: list, partial: torch.Tensor) -> torch.Tensor:
        """
        blocks: list of [B, T, D] tensors from completed blocks
        partial: [B, T, D] current in-progress block accumulation
        """
        if len(blocks) == 0:
            return partial

        all_reps = torch.stack(blocks + [partial], dim=0)  # [N+1, B, T, D]
        K = self.norm(all_reps)
        # Learned attention over depth dimension
        logits = torch.einsum('d, n b t d -> n b t', self.proj, K)
        weights = logits.softmax(dim=0)  # [N+1, B, T]
        h = torch.einsum('n b t, n b t d -> b t d', weights, all_reps)
        return h
