"""Eidos Decoder — thin wrapper, actual work done by MultiTokenPredictor."""

import torch
import torch.nn as nn


class EidosDecoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
