"""
SNAP-C1 V5: Elastic Hierarchical Context (Component 3)
=======================================================
Multi-resolution context compression with proportional boundaries.

Boundaries default to [max_seq_len//2, max_seq_len*3//4, max_seq_len],
giving 50% stride-1 (full loss signal), 25% stride-4, 25% stride-16.

Example for max_seq_len=4096:
  Level 0 (recent):   tokens[0:2048]      → full resolution (2048 slots)
  Level 1 (medium):   tokens[2048:3072]   → Conv1d stride=4 + gated residual (256 slots)
  Level 2 (distant):  tokens[3072:4096]   → Conv1d stride=16 + gated residual (64 slots)
  Total: 2368 slots

Example for max_seq_len=8192:
  Level 0 (recent):   tokens[0:4096]      → full resolution (4096 slots)
  Level 1 (medium):   tokens[4096:6144]   → Conv1d stride=4 + gated residual (512 slots)
  Level 2 (distant):  tokens[6144:8192]   → Conv1d stride=16 + gated residual (128 slots)
  Total: 4736 slots

Why Conv1d instead of avg_pool:
  avg_pool treats all tokens equally → destroys precision-critical tokens like "!", "==", "return".
  Conv1d LEARNS which tokens matter during compression.
  Gated residual lets critical tokens punch through: gate * x_full + (1-gate) * x_compressed.

DirectML safety:
  - nn.Conv1d ✓ (backward is matmul-based)
  - F.interpolate(mode='linear') ✓
  - StableSigmoid ✓
  - torch.cat ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_core.utils.dml_ops import RMSNorm, stable_sigmoid


class ElasticContext(nn.Module):
    """
    Multi-resolution context compression with learnable downsampling.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length (used to compute default boundaries)
        level_boundaries: Token boundaries for each level.
            Default: [max_seq_len//2, max_seq_len*3//4, max_seq_len]
        strides: Compression stride per non-full-res level. Default: [1, 4, 16]
            Level 0 has stride=1 (full resolution), level 1 = 4:1, level 2 = 16:1
    """

    def __init__(self, d_model: int = 1024,
                 max_seq_len: int = 8192,
                 level_boundaries: list = None,
                 strides: list = None):
        super().__init__()
        self.d_model = d_model
        # Boundaries proportional to max_seq_len: 50% stride-1, 25% stride-4, 25% stride-16
        # This ensures the loss signal covers half the sequence at full resolution
        self.boundaries = level_boundaries or [
            max_seq_len // 2,
            max_seq_len * 3 // 4,
            max_seq_len
        ]
        self.strides = strides or [1, 4, 16]
        self.n_levels = len(self.boundaries)

        assert len(self.strides) == self.n_levels

        # Per-level processing
        self.level_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(self.n_levels)])
        self.level_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(self.n_levels)
        ])

        # Learnable downsampling convolutions for compressed levels
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

        # Learnable scale weights for each level
        self.level_weights = nn.Parameter(torch.ones(self.n_levels) / self.n_levels)

    def _compress_with_gate(self, x_full: torch.Tensor, level_idx: int) -> torch.Tensor:
        """Compress with learnable Conv1d downsampling + gated residual.

        Args:
            x_full: [B, T_chunk, d_model] — full resolution chunk
            level_idx: which level (to look up the right downsampler/gate)

        Returns:
            [B, T_compressed, d_model]
        """
        key = str(level_idx)

        # Learnable downsampling: Conv1d needs [B, d, T] layout
        x_down = self.downsamplers[key](
            x_full.transpose(1, 2)
        ).transpose(1, 2)  # [B, T_compressed, d]

        # Interpolate full-res to match compressed length for gated residual
        T_compressed = x_down.shape[1]
        x_interp = F.interpolate(
            x_full.transpose(1, 2),          # [B, d, T_chunk]
            size=T_compressed,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [B, T_compressed, d]

        # Gated residual: critical tokens can punch through compression
        gate = stable_sigmoid(self.gates[key](x_interp))  # [B, T_compressed, d]
        return gate * x_interp + (1 - gate) * x_down

    def forward(self, tokens_embedded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens_embedded: [B, T_full, d_model] where T_full can be up to 8192.
                For shorter sequences, missing levels are skipped.

        Returns:
            [B, total_slots, d_model] — concatenated multi-resolution context.
                For full 8192: [B, 1024+512+320, d_model] = [B, 1856, d_model]
        """
        B, T_full, D = tokens_embedded.shape
        processed = []
        level_weights = F.softmax(self.level_weights, dim=0)  # compute once

        prev_boundary = 0
        for i in range(self.n_levels):
            end = min(self.boundaries[i], T_full)
            if prev_boundary >= end:
                break  # Sequence shorter than this level

            chunk = tokens_embedded[:, prev_boundary:end, :]  # [B, chunk_len, d]

            # Compress if needed
            if self.strides[i] > 1 and chunk.shape[1] >= self.strides[i]:
                chunk = self._compress_with_gate(chunk, i)

            # Normalize and project
            chunk = self.level_projs[i](self.level_norms[i](chunk))

            # Apply learnable scale weight
            chunk = chunk * level_weights[i]

            processed.append(chunk)
            prev_boundary = end

        # Concatenate all levels
        return torch.cat(processed, dim=1)  # [B, total_slots, d_model]

    def get_slot_count(self, seq_len: int) -> int:
        """Compute how many context slots a given sequence length produces."""
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
