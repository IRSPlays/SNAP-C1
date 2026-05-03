"""Modern Hopfield Memory — one-shot Hebbian storage and content-based retrieval.

Based on: "Hopfield Networks is All You Need" (Ramsauer et al., 2020, arXiv:2008.02217)

Core equations:
    Energy:    E(ξ) = -lse(β, X^T·ξ) + ½‖ξ‖²
    Retrieval: ξ_new = X · softmax(β · X^T · ξ)
    Write:     M ← M + δ · k ⊗ v^T   (Hebbian outer product)

The retrieval is mathematically identical to transformer attention (softmax(QK^T)V)
but with PERSISTENT keys and values stored across inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HopfieldMemory(nn.Module):
    def __init__(self, key_dim: int = 512, value_dim: int = 256, beta: float = 8.0,
                 match_threshold: float = 0.85, max_memories: int = 10000):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        self.match_threshold = match_threshold
        self.max_memories = max_memories

        self.keys = nn.Parameter(torch.empty(0, key_dim), requires_grad=False)
        self.values = nn.Parameter(torch.empty(0, value_dim), requires_grad=False)
        self.access_counts = nn.Parameter(torch.empty(0), requires_grad=False)

        self.key_proj = nn.Linear(value_dim, key_dim, bias=False)

    def read(self, queries: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = queries.shape
        if self.keys.shape[0] == 0:
            return torch.zeros(B, self.value_dim, device=queries.device, dtype=queries.dtype), \
                   torch.zeros(B, device=queries.device, dtype=queries.dtype)

        q = self.key_proj(queries)
        scores = F.cosine_similarity(q.unsqueeze(1), self.keys.unsqueeze(0), dim=-1)

        if top_k > 0 and self.keys.shape[0] > top_k:
            top_vals, top_idx = scores.topk(min(top_k, self.keys.shape[0]), dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_idx, 1.0)
            scores = scores * mask

        weights = F.softmax(self.beta * scores, dim=-1)
        retrieved = weights @ self.values

        max_scores, _ = scores.max(dim=-1)

        with torch.no_grad():
            if self.keys.shape[0] > 0:
                self.access_counts.scatter_add_(0, top_idx.reshape(-1),
                    torch.ones_like(top_idx.reshape(-1), dtype=self.access_counts.dtype))

        return retrieved, max_scores

    def write(self, keys_to_store: torch.Tensor, values_to_store: torch.Tensor,
              importance: torch.Tensor) -> int:
        B = keys_to_store.shape[0]
        written = 0
        for i in range(B):
            if importance[i] <= 0:
                continue
            k = keys_to_store[i:i+1]
            v = values_to_store[i:i+1]

            if self.keys.shape[0] > 0:
                sims = F.cosine_similarity(k, self.keys, dim=-1)
                best_idx = sims.argmax().item()
                if sims[best_idx] > self.match_threshold:
                    alpha = importance[i].item()
                    self.values.data[best_idx] = (1 - alpha) * self.values[best_idx] + alpha * v.squeeze(0)
                    self.keys.data[best_idx] = (1 - alpha) * self.keys[best_idx] + alpha * k.squeeze(0)
                    self.access_counts.data[best_idx] += 1
                    written += 1
                    continue

            if self.keys.shape[0] >= self.max_memories:
                if self.access_counts.shape[0] > 0:
                    oldest = self.access_counts.argmin().item()
                    self.keys.data[oldest] = k.squeeze(0)
                    self.values.data[oldest] = v.squeeze(0)
                    self.access_counts.data[oldest] = 1
                written += 1
                continue

            self.keys.data = torch.cat([self.keys.data, k], dim=0)
            self.values.data = torch.cat([self.values.data, v], dim=0)
            self.access_counts.data = torch.cat([
                self.access_counts.data,
                torch.ones(1, device=self.access_counts.device, dtype=self.access_counts.dtype)
            ])
            written += 1
        return written

    @property
    def num_memories(self) -> int:
        return self.keys.shape[0]

    def forget(self, frac: float = 0.1):
        if self.keys.shape[0] == 0:
            return
        n_keep = max(1, int(self.keys.shape[0] * (1 - frac)))
        keep_idx = self.access_counts.topk(n_keep).indices
        keep_idx = keep_idx.sort().values
        self.keys.data = self.keys[keep_idx]
        self.values.data = self.values[keep_idx]
        self.access_counts.data = self.access_counts[keep_idx]
