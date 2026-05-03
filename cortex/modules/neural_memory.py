"""Neural Long-Term Memory — fully differentiable write+read with low-rank compression.

Google Research, Dec 2024. arXiv:2501.00663 (Titans).

Projects to a lower-dimensional memory space for efficiency:
    dim_in = d_model (512) — full dimension for input/output
    dim_mem = d_model // 2 (256) — compressed memory dimension
    M: [dim_mem, dim_mem] — 4× smaller than [d_model, d_model]

Single differentiable forward pass:
    k = key_proj(x)           [B, T, dim_mem]
    v = value_proj(x)         [B, T, dim_mem]
    q = query_proj(x)         [B, T, dim_mem]
    M_new = (1−α)·M + α·Σ(w·K⊗V)
    h_mem_raw = M·q^T / √dim_mem     [B, T, dim_mem]
    h_mem = out_proj(h_mem_raw)       [B, T, d_model]

Gradient flows through all projections. Memory persists across batches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class NeuralMemory(nn.Module):
    def __init__(self, d_model: int = 512, momentum_init: float = 0.95,
                 mem_ratio: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.dim_mem = max(int(d_model * mem_ratio), 64)

        # Project to/from compressed memory space
        self.key_proj = nn.Linear(d_model, self.dim_mem, bias=False)
        self.value_proj = nn.Linear(d_model, self.dim_mem, bias=False)
        self.query_proj = nn.Linear(d_model, self.dim_mem, bias=False)
        self.out_proj = nn.Linear(self.dim_mem, d_model, bias=False)

        self.momentum_proj = nn.Linear(d_model * 2, 1)

        self.register_buffer('M', torch.zeros(self.dim_mem, self.dim_mem))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

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

        # ── Learnable projections to compressed memory space ──
        k = self.key_proj(x)    # [B, T, D]
        v = self.value_proj(x)  # [B, T, D]
        q = self.query_proj(x)  # [B, T, D]

        # ── Per-batch-item momentum ──
        alpha = self._compute_momentum(x, surprise)  # [B, 1]

        # ── Weighted batch updates (bmm is faster than einsum on CUDA) ──
        effective_surprise = surprise
        if write_gate is not None:
            effective_surprise = surprise * write_gate
        w = F.softmax(effective_surprise, dim=1)  # [B, T]
        w_k = (w.unsqueeze(-1) * k).transpose(1, 2)  # [B, D, T]
        batch_updates = w_k @ v  # [B, D, D]

        # ── Momentum Hebbian update ──
        M_prev = self.M.detach()  # [D, D]
        has_memory = M_prev.abs().sum() > 1e-8

        if has_memory:
            a = alpha.view(B, 1, 1)  # [B, 1, 1]
            M_per_item = (1 - a) * M_prev.unsqueeze(0) + a * batch_updates
            M_new_raw = M_per_item.mean(dim=0)
        else:
            M_new_raw = batch_updates.mean(dim=0)

        scale = M_new_raw.norm().detach() + 1e-8
        M_new = M_new_raw / scale

        # ── Read from memory ──
        scores = torch.einsum('btd,de->bte', q.to(M_new.dtype), M_new) / math.sqrt(D)
        h_mem_raw = F.normalize(scores, dim=-1)  # [B, T, D]

        # ── Project back to model dimension ──
        h_mem = self.out_proj(h_mem_raw)  # [B, T, d_model]

        # ── Persist ──
        if self.training:
            if torch.isfinite(M_new).all():
                self.M.data = M_new.detach()
            else:
                self.M.data.zero_()

        return h_mem

    def reset(self):
        self.M.data.zero_()
        self.step_count.zero_()
