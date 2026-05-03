"""Neuromodulator — deterministic brain-inspired global modulation.

Outputs 4 signals from prediction error, memory match, and problem complexity:

    δ = sigmoid(z_score(ε))              dopamine → memory write gate
    ν = 2 + clamp(num_count // 3, 0, 14)  norepinephrine → LTC iteration budget
    σ = sigmoid((memory_match − 0.5) * 5) serotonin → memory vs cortex blend (soft)
    α = 0                                 acetylcholine → unused

Running statistics track the distribution of prediction errors so
z-scores are centered across training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Neuromodulator(nn.Module):
    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.register_buffer('error_mean', torch.tensor(0.0))
        self.register_buffer('error_std', torch.tensor(1.0))
        self.momentum = momentum

    def forward(self, error: torch.Tensor,
                memory_match: torch.Tensor,
                num_count: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.training:
                batch_mean = error.mean()
                batch_std = error.std() + 1e-8
                self.error_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                self.error_std.mul_(self.momentum).add_(batch_std, alpha=1 - self.momentum)

        z_score = (error - self.error_mean) / (self.error_std + 1e-8)

        # δ: dopamine — write gate, high for surprising tokens
        dopamine = torch.sigmoid(z_score)

        # ν: norepinephrine — LTC iterations from problem complexity
        if num_count is not None:
            norepi = torch.clamp(2 + num_count // 3, 2, 16)
        else:
            per_sample_error = error.mean(dim=-1)
            norepi_raw = 1.0 + per_sample_error * 4.0
            norepi = torch.clamp(norepi_raw.round().long(), 1, 8)

        # σ: serotonin — memory trust with softer temperature
        serotonin = torch.sigmoid((memory_match - 0.5) * 5.0)

        acetylcholine = torch.zeros_like(serotonin)

        return dopamine, norepi, serotonin, acetylcholine
