"""Liquid Time-Constant RNN Cell (Cortex)

Neural ODE with input-dependent time constants.
Adaptive dt controlled by norepinephrine signal.

Math:
    tau_i = tau_base_i + softplus(W_tau @ x)
    f = sigmoid(W_x @ x + W_h @ h + b)
    dh/dt = (-h + f) / tau
    h_new = h + dt * dh

Reference: Hasani et al. "Liquid Time-constant Networks" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """Single step of a Liquid Time-Constant RNN."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_tau = nn.Linear(input_size, hidden_size)

        # Learnable base time constant (initialized > 0 via softplus)
        self.tau_base = nn.Parameter(torch.ones(hidden_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_tau.weight)
        nn.init.zeros_(self.W_x.bias)
        nn.init.zeros_(self.W_tau.bias)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """One LTC step.

        Args:
            x: [B, input_size]
            h: [B, hidden_size]
            dt: time step (modulated by norepinephrine in Phase 2)

        Returns:
            h_new: [B, hidden_size]
        """
        # Input-dependent time constant (always positive)
        tau = F.softplus(self.tau_base) + F.softplus(self.W_tau(x))  # [B, hidden_size]

        # Activation gate
        f = torch.sigmoid(self.W_x(x) + self.W_h(h))  # [B, hidden_size]

        # ODE step: dh/dt = (-h + f) / tau
        dh = (-h + f) / tau
        h_new = h + dt * dh

        return h_new


class LTCRNN(nn.Module):
    """Unrolled LTC over a sequence."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LTCCell(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process sequence through LTC-RNN.

        Args:
            x: [B, T, input_size]
            h0: [B, hidden_size] initial state (zeros if None)
            dt: time step size

        Returns:
            outputs: [B, T, hidden_size] — hidden state at each step
            h_final: [B, hidden_size] — last hidden state
        """
        B, T, _ = x.shape
        device = x.device

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=device)
        else:
            h = h0

        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h, dt=dt)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # [B, T, hidden_size]
        return outputs, h
