"""Token Encoder — BPE embedding + sinusoidal positional encoding."""

import math
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, vocab_size: int, d_model: int = 256, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding (no learnable params)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] long tensor

        Returns:
            embeddings: [B, T, d_model]
        """
        T = token_ids.size(1)
        x = self.token_emb(token_ids) * math.sqrt(self.d_model)
        x = x + self.pe[:, :T, :]
        x = self.dropout(self.norm(x))
        return x
