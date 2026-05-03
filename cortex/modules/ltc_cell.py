"""Liquid Time-Constant cells — adaptive computation speed neural network.

Based on: "Liquid Time-constant Networks" (Hasani et al., 2020, arXiv:2006.04439)

Core equation:
    dh/dt = -h / τ(h, x) + f(h, x)
    τ(h, x) = σ(W_τ·h + U_τ·x + b_τ)

Discretized (Euler):
    h_{t+1} = h_t + Δt · (-h_t/τ + f(h_t, x_t))

Properties:
- Novel input → small τ → fast reaction
- Familiar input → large τ → stable, energy-efficient
- Bounded states (proven)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LTCCell(nn.Module):
    def __init__(self, d_model: int, dt: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dt = dt

        self.input_proj = nn.Linear(d_model, d_model)
        self.tau_linear = nn.Linear(d_model * 2, d_model)
        self.hidden_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        for m in [self.input_proj, self.hidden_proj]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(self.d_model))
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.tau_linear.weight, std=0.02)
        nn.init.normal_(self.tau_linear.bias, mean=2.0, std=0.1)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None,
                iterations: int = 1) -> torch.Tensor:
        B = x.shape[0]
        if h is None:
            h = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)

        x_proj = self.input_proj(x)
        for _ in range(iterations):
            combined = torch.cat([h, x_proj], dim=-1)
            tau = F.softplus(self.tau_linear(combined)) + 1e-4
            f_out = torch.tanh(self.hidden_proj(h) + x_proj)
            dh = -h / tau + f_out
            h = h + self.dt * dh

        return h


class LTCBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, dt: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cell = LTCCell(d_model, dt=dt)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None,
                iterations: int = 1) -> torch.Tensor:
        h = self.cell(self.norm(x), h=h, iterations=iterations)
        return x + self.drop(h)
