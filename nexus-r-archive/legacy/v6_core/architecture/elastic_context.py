"""
SNAP-C1 V5: Elastic Hierarchical Context (Component 3)
======================================================
Multi-resolution context compression with proportional boundaries.
Copied from V5 - working DirectML-safe implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_core.architecture.dml_ops import RMSNorm, stable_sigmoid


class ElasticContext(nn.Module):
    def __init__(self, d_model: int = 1024,
                 max_seq_len: int = 8192,
                 level_boundaries: list = None,
                 strides: list = None):
        super().__init__()
        self.d_model = d_model
        self.boundaries = level_boundaries or [
            max_seq_len // 2,
            max_seq_len * 3 // 4,
            max_seq_len
        ]
        self.strides = strides or [1, 4, 16]
        self.n_levels = len(self.boundaries)

        assert len(self.strides) == self.n_levels

        self.level_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(self.n_levels)])
        self.level_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(self.n_levels)
        ])

        self.downsamplers = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        for i, stride in enumerate(self.strides):
            if stride > 1:
                self.downsamplers[str(i)] = nn.Conv1d(
                    d_model, d_model,
                    kernel_size=stride, stride=stride,
                    bias=True
                )
                self.gates[str(i)] = nn.Linear(d_model, d_model)

        self.level_weights = nn.Parameter(torch.ones(self.n_levels) / self.n_levels)

    def _compress_with_gate(self, x_full: torch.Tensor, level_idx: int) -> torch.Tensor:
        key = str(level_idx)

        x_down = self.downsamplers[key](
            x_full.transpose(1, 2)
        ).transpose(1, 2)

        T_compressed = x_down.shape[1]
        x_interp = F.interpolate(
            x_full.transpose(1, 2),
            size=T_compressed,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        gate = stable_sigmoid(self.gates[key](x_interp))
        return gate * x_interp + (1 - gate) * x_down

    def forward(self, tokens_embedded: torch.Tensor) -> torch.Tensor:
        B, T_full, D = tokens_embedded.shape
        processed = []
        level_weights = F.softmax(self.level_weights, dim=0)

        prev_boundary = 0
        for i in range(self.n_levels):
            end = min(self.boundaries[i], T_full)
            if prev_boundary >= end:
                break

            chunk = tokens_embedded[:, prev_boundary:end, :]

            if self.strides[i] > 1 and chunk.shape[1] >= self.strides[i]:
                chunk = self._compress_with_gate(chunk, i)

            chunk = self.level_projs[i](self.level_norms[i](chunk))
            chunk = chunk * level_weights[i]

            processed.append(chunk)
            prev_boundary = end

        return torch.cat(processed, dim=1)

    def get_slot_count(self, seq_len: int) -> int:
        total = 0
        prev = 0
        for i in range(self.n_levels):
            end = min(self.boundaries[i], seq_len)
            if prev >= end:
                break
            chunk_len = end - prev
            if self.strides[i] > 1:
                total += chunk_len // self.strides[i]
            else:
                total += chunk_len
            prev = end
        return total
