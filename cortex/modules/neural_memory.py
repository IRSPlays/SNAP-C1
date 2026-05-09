"""Neural Long-Term Memory — fully differentiable write+read with low-rank compression.

Google Research, Dec 2024. arXiv:2501.00663 (Titans).

Features:
  - Surprise-gated writes: only patterns with surprise > threshold get stored
  - Low-rank memory: dim_mem = d_model // 2 for 4× compression
  - Momentum Hebbian update with per-batch adaptive α
  - NaN-safe reads and writes

Single differentiable forward pass:
    k = key_proj(x)           [B, T, dim_mem]
    v = value_proj(x)         [B, T, dim_mem]
    q = query_proj(x)         [B, T, dim_mem]
    M_new = (1−α)·M + α·Σ(w·K⊗V)
    h_mem_raw = M·q^T / √dim_mem     [B, T, dim_mem]
    h_mem = out_proj(h_mem_raw)       [B, T, d_model]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class NeuralMemory(nn.Module):
    def __init__(self, d_model: int = 512, momentum_init: float = 0.95,
                 mem_ratio: float = 0.5, write_threshold: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.dim_mem = max(int(d_model * mem_ratio), 64)
        self.write_threshold = write_threshold

        self.key_proj = nn.Linear(d_model, self.dim_mem, bias=False)
        self.value_proj = nn.Linear(d_model, self.dim_mem, bias=False)
        self.query_proj = nn.Linear(d_model, self.dim_mem, bias=False)
        self.out_proj = nn.Linear(self.dim_mem, d_model, bias=False)

        self.momentum_proj = nn.Linear(d_model * 2, 1)

        self.register_buffer('M', torch.zeros(self.dim_mem, self.dim_mem))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self._write_count = 0
        self._skip_count = 0

        self.momentum_init = momentum_init

        self._init_weights()

    def _init_weights(self):
        for m in [self.key_proj, self.value_proj, self.query_proj, self.out_proj, self.momentum_proj]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(self.d_model))

    def _compute_momentum(self, x: torch.Tensor, surprise: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        surprise_pooled = surprise.mean(dim=1)
        if surprise_pooled.dim() == 0:
            surprise_pooled = surprise_pooled.unsqueeze(0)
        surprise_pooled = surprise_pooled.unsqueeze(-1).expand(-1, pooled.size(-1))
        combined = torch.cat([pooled, surprise_pooled], dim=-1)
        alpha = torch.sigmoid(self.momentum_proj(combined))
        return alpha.clamp(min=0.05, max=self.momentum_init)

    def forward(self, x: torch.Tensor, surprise: torch.Tensor,
                write_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D_in = x.shape
        D = self.dim_mem

        k = self.key_proj(x)
        v = self.value_proj(x)
        q = self.query_proj(x)

        alpha = self._compute_momentum(x, surprise)

        # Surprise-gated write: only store patterns above threshold
        effective_surprise = surprise
        if write_gate is not None:
            effective_surprise = surprise * write_gate
        # Zero out low-surprise tokens (they don't get stored — saving memory bandwidth)
        surprise_mask = (effective_surprise > self.write_threshold).float()
        effective_surprise = effective_surprise * surprise_mask
        w = F.softmax(effective_surprise, dim=1)
        w_k = (w.unsqueeze(-1) * k).transpose(1, 2)
        batch_updates = w_k @ v

        M_current = self.M.detach().clone()
        has_memory = M_current.abs().sum() > 1e-8

        if has_memory:
            a = alpha.view(B, 1, 1)
            M_per_item = (1 - a) * M_current.unsqueeze(0) + a * batch_updates
            M_new_raw = M_per_item.mean(dim=0)
        else:
            M_new_raw = batch_updates.mean(dim=0)

        scale = M_new_raw.float().norm().detach() + 1e-8
        M_new = M_new_raw / scale.to(M_new_raw.dtype)

        if not torch.isfinite(M_new).all():
            M_new = torch.zeros_like(M_new)

        scores = torch.einsum('btd,de->bte', q.to(M_new.dtype), M_new) / math.sqrt(D)
        h_mem_raw = F.normalize(scores.float(), dim=-1).to(scores.dtype)
        h_mem = self.out_proj(h_mem_raw)

        if self.training and torch.isfinite(M_new).all():
            self.M[:] = M_new.to(self.M.dtype)
            self._write_count += 1
        elif self.training:
            self._skip_count += 1

        return h_mem

    @property
    def write_stats(self):
        total = self._write_count + self._skip_count
        if total == 0:
            return "no writes yet"
        return f"{self._write_count} writes, {self._skip_count} skips ({100*self._write_count/total:.0f}% success)"

    def reset(self):
        self.M.data.zero_()
        self.step_count.zero_()
        self._write_count = 0
        self._skip_count = 0
