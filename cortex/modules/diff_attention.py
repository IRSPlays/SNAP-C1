"""Differential Attention (Diff Transformer) — ICLR 2025 Oral.

Microsoft Research, Oct 2024. arXiv:2410.05258.

Core formula:
    A1 = softmax(Q1·K1^T / √d)
    A2 = softmax(Q2·K2^T / √d)
    DiffAttn(X) = (A1 - λ·A2) · V

where λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init

The subtraction cancels common-mode noise, promoting sparse attention patterns.
Proven to reduce hallucination, improve in-context learning, reduce activation outliers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
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
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out.to(orig_dtype), k_out.to(orig_dtype)


def _multiple_of(n: int, m: int) -> int:
    return (-(n // -m)) * m


class DiffAttentionLayer(nn.Module):
    """Single layer: Differential Attention + SwiGLU FFN.

    Differential attention splits heads into two groups (A1, A2) and subtracts.
    λ is per-head learnable with depth-dependent init.
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_kv_heads: int = 4,
                 ffn_expansion: float = 8 / 3, dropout: float = 0.0, depth: int = 0,
                 max_depth: int = 4):
        super().__init__()
        head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.depth = depth

        total_q_heads = n_heads * 2
        total_kv_heads = n_kv_heads * 2
        self.attn_norm = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, total_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        lambda_init = 0.8 - 0.6 * math.exp(-depth / max(1, max_depth - 1))
        self.lambda_q1 = nn.Parameter(0.01 * torch.randn(n_heads))
        self.lambda_k1 = nn.Parameter(0.01 * torch.randn(n_heads))
        self.lambda_q2 = nn.Parameter(0.01 * torch.randn(n_heads))
        self.lambda_k2 = nn.Parameter(0.01 * torch.randn(n_heads))
        self.register_buffer('lambda_init', torch.tensor(lambda_init))

        self.ffn_norm = nn.RMSNorm(d_model)
        hidden = _multiple_of(round(d_model * ffn_expansion * 2 / 3), 256)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

        self.drop = nn.Dropout(dropout)

    def _compute_lambda(self) -> torch.Tensor:
        lam = (torch.exp(self.lambda_q1 * self.lambda_k1)
               - torch.exp(self.lambda_q2 * self.lambda_k2)
               + self.lambda_init)
        return lam.view(1, self.n_heads, 1, 1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                is_causal: bool = False) -> torch.Tensor:
        B, T, D = x.shape

        normed = self.attn_norm(x)
        q = self.q_proj(normed).view(B, T, self.n_heads * 2, self.head_dim)
        k = self.k_proj(normed).view(B, T, self.n_kv_heads * 2, self.head_dim)
        v = self.v_proj(normed).view(B, T, self.n_kv_heads * 2, self.head_dim)

        q, k = apply_rope(q, k, cos, sin)

        q1 = q[:, :, :self.n_heads]
        q2 = q[:, :, self.n_heads:]
        k1 = k[:, :, :self.n_kv_heads]
        k2 = k[:, :, self.n_kv_heads:]
        v1 = v[:, :, :self.n_kv_heads]
        v2 = v[:, :, self.n_kv_heads:]

        q1, k1, v1 = (t.transpose(1, 2) for t in (q1, k1, v1))
        q2, k2, v2 = (t.transpose(1, 2) for t in (q2, k2, v2))

        if q1.size(1) != k1.size(1):
            r = q1.size(1) // k1.size(1)
            k1e = k1.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, q1.size(1), T, self.head_dim)
            v1e = v1.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, q1.size(1), T, self.head_dim)
            k2e = k2.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, q1.size(1), T, self.head_dim)
            v2e = v2.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, q1.size(1), T, self.head_dim)
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                 torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                 torch.nn.attention.SDPBackend.MATH]
            ):
                a1 = F.scaled_dot_product_attention(q1, k1e, v1e, is_causal=is_causal)
                a2 = F.scaled_dot_product_attention(q2, k2e, v2e, is_causal=is_causal)
        else:
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                 torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                 torch.nn.attention.SDPBackend.MATH]
            ):
                a1 = F.scaled_dot_product_attention(q1, k1, v1, is_causal=is_causal)
                a2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=is_causal)

        lam = self._compute_lambda()
        diff_attn = (a1 - lam * a2) * (1.0 / (1.0 + lam.abs()))
        diff_attn = diff_attn.transpose(1, 2).reshape(B, T, D)

        x = x + self.drop(self.out_proj(diff_attn))

        normed = self.ffn_norm(x)
        gate = F.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        x = x + self.drop(self.down_proj(gate * up))

        return x
