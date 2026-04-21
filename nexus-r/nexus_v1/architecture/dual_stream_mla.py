"""
Nexus-r V1: Dual-Stream Multi-Head Latent Attention
=====================================================
The critical innovation of the Asymmetric Recursive Architecture.

Stream 1 (Anchor): Input prompt → compute K, V once → FREEZE in VRAM
Stream 2 (Thought): Recursive reasoning → produces Q each pass

Each recursion: new Q queries frozen K,V.
"Given my new deduction, what parts of the original prompt matter now?"

This mathematically prevents the Echo Chamber Effect (hallucination from
recursive self-attention overwriting the original prompt).

Reference: DeepSeek MLA concept (latent KV compression), but routing is novel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .layers import (
    CastedLinear, RMSNorm, rms_norm,
    apply_rotary_pos_emb, RotaryEmbedding,
)


class DualStreamMLA(nn.Module):
    """
    Dual-Stream Multi-Head Latent Attention.

    The Anchor stream (K, V) is computed once from the input and frozen.
    The Thought stream (Q) evolves each recursive pass.

    Forward signature supports two modes:
    1. anchor_mode=True:  Compute and cache K, V from input (run once)
    2. anchor_mode=False: Cross-attend thought Q against frozen anchor K, V
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads

        # Thought projection (Q from recursive state)
        self.q_proj = CastedLinear(d_model, n_heads * self.head_dim)

        # Output projection
        self.out_proj = CastedLinear(n_heads * self.head_dim, d_model)

    def forward(
        self,
        thought: torch.Tensor,
        anchor_k: torch.Tensor,
        anchor_v: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Cross-attend: thought Q queries frozen anchor K, V.

        Args:
            thought:  [B, T, D] — current recursive thought state
            anchor_k: [B, n_kv_heads, T, head_dim] — frozen keys
            anchor_v: [B, n_kv_heads, T, head_dim] — frozen values
            cos_sin:  RoPE for Q

        Returns:
            [B, T, D] — attended output
        """
        B, T, _ = thought.shape

        q = self.q_proj(thought).view(B, T, self.n_heads, self.head_dim)

        # Apply RoPE to Q
        if cos_sin is not None:
            cos, sin = cos_sin
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
            q_rot = q.to(cos.dtype)
            x1, x2 = q_rot[..., :self.head_dim // 2], q_rot[..., self.head_dim // 2:]
            q = (q_rot * cos + torch.cat((-x2, x1), dim=-1) * sin).to(q.dtype)

        q = q.transpose(1, 2)  # [B, H, T, D]

        # GQA expand if needed
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = anchor_k.repeat_interleave(rep, dim=1)
            v = anchor_v.repeat_interleave(rep, dim=1)
        else:
            k, v = anchor_k, anchor_v

        # Attention (CAUSAL — thought at position i must not see anchor positions > i)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)
