"""Eidos Encoder — stack of Differential Attention layers with number value embedding.

Maps input tokens → d-dimensional embeddings.
Adds a numeric magnitude signal so numbers are represented as quantities, not just tokens.
Uses Diff Transformer (ICLR 2025) for noise-resistant representations.

Also supports Forward-Forward training (Hinton 2022):
    Positive pass: real data → maximize activation norm per layer
    Negative pass: corrupted data → minimize activation norm per layer
    Each layer trains INDEPENDENTLY via local loss (no cross-layer gradients).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
from .diff_attention import DiffAttentionLayer, RotaryEmbedding


class EidosEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_kv_heads: int = 4, n_layers: int = 4,
                 max_seq_len: int = 512, dropout: float = 0.0,
                 embed_weights: Optional[torch.Tensor] = None,
                 num_values: Optional[torch.Tensor] = None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if embed_weights is not None:
            with torch.no_grad():
                self.token_emb.weight.copy_(embed_weights)
        else:
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        # Number value embedding: adds magnitude signal to token embeddings
        self.num_proj = None
        if num_values is not None:
            self.register_buffer('num_values', num_values)
            self.num_proj = nn.Linear(1, d_model, bias=False)
            nn.init.normal_(self.num_proj.weight, std=0.05)

        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

        self.layers = nn.ModuleList([
            DiffAttentionLayer(d_model, n_heads, n_kv_heads, depth=i,
                                max_depth=n_layers, dropout=dropout)
            for i in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)

        # FF goodness thresholds (per layer)
        self.register_buffer('ff_threshold', torch.ones(n_layers) * 1.0)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.token_emb(input_ids)

        # Inject numeric magnitude signal
        if self.num_proj is not None:
            vals = self.num_values[input_ids].unsqueeze(-1)  # [B, T, 1]
            x = x + self.num_proj(vals)

        B, T, _ = x.shape
        cos, sin = self.rope(T)

        for layer in self.layers:
            x = layer(x, cos, sin, is_causal=True)

        pooled = x.mean(dim=1)
        return self.final_norm(x), pooled

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embedding only (no layers). Used by FF training."""
        x = self.token_emb(input_ids)
        if self.num_proj is not None:
            vals = self.num_values[input_ids].unsqueeze(-1)
            x = x + self.num_proj(vals)
        return x

    def forward_with_goodness(self, x: torch.Tensor, cos: torch.Tensor,
                               sin: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning per-layer activation norms (goodness)."""
        goodness = []
        for layer in self.layers:
            x = layer(x, cos, sin, is_causal=True)
            # Goodness = mean squared activation (Hinton 2022 convention)
            goodness.append(x.pow(2).mean(dim=-1).mean(dim=-1))  # [B]
        return goodness

    def ff_loss(self, input_ids: torch.Tensor, negative_ids: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward-Forward loss for all encoder layers.

        Positive pass: real data → high goodness
        Negative pass: corrupted data (or auto-generated) → low goodness
        Loss per layer = -log(sigmoid(goodness_pos - threshold))
                         -log(sigmoid(threshold - goodness_neg))

        Returns (total_loss, per_layer_losses).
        """
        # Positive pass
        x_pos = self._embed(input_ids)
        _, T, _ = x_pos.shape
        cos, sin = self.rope(T)
        good_pos = self.forward_with_goodness(x_pos, cos, sin)
        good_pos = torch.stack(good_pos)  # [L, B]

        # Negative pass (shuffle tokens if no explicit negative provided)
        if negative_ids is None:
            idx_shuffle = torch.randperm(T, device=input_ids.device)
            negative_ids = input_ids[:, idx_shuffle]
        x_neg = self._embed(negative_ids)
        cos_n, sin_n = self.rope(x_neg.shape[1])
        good_neg = self.forward_with_goodness(x_neg, cos_n, sin_n)
        good_neg = torch.stack(good_neg)  # [L, B]

        # Per-layer FF loss
        thresholds = self.ff_threshold.unsqueeze(1)  # [L, 1]
        layer_losses = (
            F.softplus(thresholds - good_pos) +  # minimize for positive
            F.softplus(good_neg - thresholds)    # minimize for negative
        )  # [L, B]
        loss = layer_losses.mean()

        return loss, layer_losses.mean(dim=1)  # per-layer avg
