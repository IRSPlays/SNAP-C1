"""Output Decoder — projects hidden state to vocabulary logits.

Supports weight tying with encoder embedding.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, d_model: int, vocab_size: int, tie_weights: nn.Embedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        if tie_weights is not None:
            # Weight tying: share embedding matrix
            self.proj = None
            self.tied_weight = tie_weights.weight  # [vocab_size, d_model]
        else:
            self.proj = nn.Linear(d_model, vocab_size, bias=False)
            self.tied_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.norm(x)
        if self.tied_weight is not None:
            return torch.matmul(x, self.tied_weight.t())
        return self.proj(x)
