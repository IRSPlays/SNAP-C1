"""Predictive Coder — top-down prediction of next embedding AND number value.

Predicts z_{t+1} from z_t AND global pooled context, and also predicts
the numeric value of the next token (for arithmetic error detection).

Outputs:
    z_hat: predicted next embedding
    error: L2 distance between predicted and actual embedding
    cosine_dist: 1 − cos(z_hat, z_actual) — surprise signal
    value_pred: predicted numeric magnitude at each position [B, T]

Uses sequence-level pooled context so prediction is meaningful (not single-token blind).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PredictiveCoder(nn.Module):
    def __init__(self, d_model: int = 512, expansion: float = 4.0):
        super().__init__()
        hidden = int(d_model * expansion)
        self.d_model = d_model

        self.norm = nn.RMSNorm(d_model)
        self.pooled_norm = nn.RMSNorm(d_model)

        self.fc1 = nn.Linear(d_model * 2, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.value_head = nn.Linear(hidden, 1, bias=False)

        nn.init.normal_(self.fc1.weight, std=1.0 / (d_model * 2) ** 0.5)
        nn.init.normal_(self.fc2.weight, std=1.0 / hidden ** 0.5)
        nn.init.normal_(self.value_head.weight, std=0.01)

    def forward(self, z_prev: torch.Tensor,
                z_next: torch.Tensor,
                z_pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z_prev.shape

        # Concatenate local prev token with global pooled context
        z_prev_norm = self.norm(z_prev)
        z_pooled_expanded = self.pooled_norm(z_pooled).unsqueeze(1).expand(-1, T, -1)
        combined = torch.cat([z_prev_norm, z_pooled_expanded], dim=-1)

        hidden = F.gelu(self.fc1(combined))
        z_hat = self.fc2(hidden)
        value_pred = self.value_head(hidden).squeeze(-1)  # [B, T]

        error = torch.norm(z_next.float() - z_hat.float(), dim=-1)

        z_hat_norm = F.normalize(z_hat.float(), dim=-1)
        z_next_norm = F.normalize(z_next.float(), dim=-1)
        cosine_dist = 1.0 - (z_hat_norm * z_next_norm).sum(dim=-1)

        return z_hat, error, cosine_dist, value_pred
