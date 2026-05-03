"""Eidos Encoder — stack of Differential Attention layers with number value embedding.

Maps input tokens → d-dimensional embeddings.
Adds a numeric magnitude signal so numbers are represented as quantities, not just tokens.
Uses Diff Transformer (ICLR 2025) for noise-resistant representations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
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
