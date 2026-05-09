"""LTC Cortex — Liquid Time-Constant recurrent network with adaptive iterations.

Accepts optional memory context (h_mem) that it fuses with input before recurrence.
Memory-augmented: LTC processes z ⊕ h_mem, where h_mem is retrieved relevant knowledge.

Based on: "Liquid Time-constant Networks" (Hasani et al., 2020, arXiv:2006.04439)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LTCCortex(nn.Module):
    def __init__(self, d_model: int = 512, expansion: float = 4.0,
                 dt: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.dt_per_step = dt

        self.norm_in = nn.RMSNorm(d_model)
        self.mem_fuse = nn.Linear(d_model * 2, d_model, bias=False)

        self.tau_linear = nn.Linear(d_model * 2, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.hidden_proj = nn.Linear(d_model, d_model)
        self.norm_mid = nn.RMSNorm(d_model)

        hidden = int(d_model * expansion)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mem_fuse.weight, std=1.0 / math.sqrt(self.d_model * 2))
        for m in [self.input_proj, self.hidden_proj, self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(self.d_model))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.tau_linear.weight, std=0.02)
        nn.init.normal_(self.tau_linear.bias, mean=2.0, std=0.1)

    def _ltc_step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h, x], dim=-1)
        tau = F.softplus(self.tau_linear(combined).float()) + 1e-4
        tau = tau.to(h.dtype)
        dh_val = -h / tau + torch.tanh(self.hidden_proj(h) + self.input_proj(x))
        return h + self.dt_per_step * dh_val

    def forward(self, x: torch.Tensor, iterations: int = 1,
                memory: Optional[torch.Tensor] = None,
                h: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        # Fuse current input with retrieved memory
        if memory is not None:
            fused = self.mem_fuse(torch.cat([x, memory], dim=-1))
        else:
            fused = x

        normed = self.norm_in(fused)

        if h is None:
            h = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)

        # TD learning: track per-iteration error for self-improvement signal
        td_errors = []
        for _ in range(iterations):
            h_prev = h.clone()
            h = self._ltc_step(normed, h)
            # TD error: how much did this iteration change the state?
            td_errors.append(F.mse_loss(h, h_prev, reduction='none').mean(dim=-1))  # [B, T]

        gate = F.silu(self.gate_proj(self.norm_mid(h)))
        up = self.up_proj(self.norm_mid(h))
        out = h + self.drop(self.down_proj(gate * up))
        return out
