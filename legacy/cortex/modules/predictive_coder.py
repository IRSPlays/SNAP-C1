"""Predictive Coding Module

Single-pass approximation of predictive coding:
Each layer computes a representation, then predicts its own input.
The prediction error (what was surprising) propagates to the next layer alongside
the representation via a skip connection.

This forces each layer to focus compute on the unpredicted parts of the input.

Math:
    r_l = GELU(W_up @ input_l)          # bottom-up representation
    pred_l = W_down @ r_l                # top-down prediction of input
    error_l = input_l - pred_l           # prediction error
    input_{l+1} = r_l + error_l          # skip + error to next layer

    prediction_error = mean(||error_l||^2) across layers  (scalar for neuromodulator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveCodingLayer(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.encode = nn.Linear(d_model, d_model)
        self.predict = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            output: [B, T, d_model] — representation + error (skip connection)
            error_sq: [B, T] — squared error norm per position
        """
        r = F.gelu(self.encode(x))        # bottom-up
        pred = self.predict(r)             # top-down prediction of input
        error = x - pred                   # what wasn't predicted

        # Squared error per position (for neuromodulator)
        error_sq = (error ** 2).mean(dim=-1)  # [B, T]

        # Output: representation + error signal
        output = self.norm(r + error)

        return output, error_sq


class PredictiveCoder(nn.Module):

    def __init__(self, d_model: int = 256, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(d_model) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            output: [B, T, d_model] — processed representation
            prediction_error: [B, T] — mean squared error across layers per position
        """
        total_error = torch.zeros(x.size(0), x.size(1), device=x.device)

        for layer in self.layers:
            x, error_sq = layer(x)
            total_error = total_error + error_sq

        # Average across layers
        prediction_error = total_error / len(self.layers)

        return x, prediction_error
