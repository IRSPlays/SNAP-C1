"""Modern Hopfield Network Memory (Hippocampus)

One-shot storage via outer product writes. No backprop needed for memory storage.
Retrieval uses softmax attention over stored keys.

Math:
    Write:  K ← cat(K, normalize(k)),  V ← cat(V, v)   if δ > threshold
    Read:   attn = softmax(β · K · q),  output = attn^T · V
    Confidence = max(attn)  (how strongly one memory matched)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldMemory(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        d_key: int = 512,
        d_val: int = 256,
        max_memories: int = 10_000,
        beta: float = 8.0,
        similarity_threshold: float = 0.85,
        write_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.max_memories = max_memories
        self.beta = beta
        self.similarity_threshold = similarity_threshold
        self.write_threshold = write_threshold

        # Projections from model space to memory space
        self.key_proj = nn.Linear(d_model, d_key)
        self.val_proj = nn.Linear(d_model, d_val)
        self.query_proj = nn.Linear(d_model, d_key)
        self.out_proj = nn.Linear(d_val, d_model)

        # Memory banks (not parameters — managed manually)
        self.register_buffer('keys', torch.zeros(0, d_key))
        self.register_buffer('values', torch.zeros(0, d_val))
        self.register_buffer('access_counts', torch.zeros(0, dtype=torch.long))
        self.register_buffer('num_memories', torch.tensor(0, dtype=torch.long))

    @property
    def size(self) -> int:
        return self.num_memories.item()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    @torch.no_grad()
    def write(self, x: torch.Tensor, dopamine: torch.Tensor):
        """Write to memory if dopamine exceeds threshold.

        Args:
            x: [B, d_model] — input to store
            dopamine: [B] — write gate signal
        """
        # Project to key/value space
        keys = self._normalize(self.key_proj(x))      # [B, d_key]
        values = self.val_proj(x)                       # [B, d_val]

        for i in range(x.size(0)):
            if dopamine[i].item() < self.write_threshold:
                continue

            k = keys[i]  # [d_key]
            v = values[i]  # [d_val]

            if self.size > 0:
                # Check for collision (similar key already exists)
                sims = torch.mv(self.keys, k)  # [N]
                max_sim, max_idx = sims.max(dim=0)

                if max_sim.item() > self.similarity_threshold:
                    # Update existing memory (weighted average)
                    alpha = 0.3
                    self.values[max_idx] = (1 - alpha) * self.values[max_idx] + alpha * v
                    self.access_counts[max_idx] += 1
                    continue

            # Evict least-accessed if full
            if self.size >= self.max_memories:
                min_idx = self.access_counts.argmin()
                self.keys[min_idx] = k
                self.values[min_idx] = v
                self.access_counts[min_idx] = 1
            else:
                # Append new memory
                self.keys = torch.cat([self.keys, k.unsqueeze(0)], dim=0)
                self.values = torch.cat([self.values, v.unsqueeze(0)], dim=0)
                self.access_counts = torch.cat([
                    self.access_counts,
                    torch.ones(1, dtype=torch.long, device=x.device)
                ])
                self.num_memories += 1

    def read(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read from memory using attention.

        Args:
            x: [B, d_model] — query input

        Returns:
            output: [B, d_model] — retrieved value projected back
            confidence: [B] — max attention weight (0-1)
        """
        if self.size == 0:
            return (
                torch.zeros(x.size(0), x.size(-1), device=x.device),
                torch.zeros(x.size(0), device=x.device),
            )

        query = self._normalize(self.query_proj(x))  # [B, d_key]

        # Attention over stored keys
        logits = self.beta * torch.mm(query, self.keys.t())  # [B, N]
        attn = F.softmax(logits, dim=-1)                     # [B, N]

        # Retrieve values
        retrieved = torch.mm(attn, self.values)  # [B, d_val]
        output = self.out_proj(retrieved)         # [B, d_model]

        # Confidence = max attention weight
        confidence = attn.max(dim=-1).values  # [B]

        # Update access counts
        with torch.no_grad():
            top_indices = attn.argmax(dim=-1)  # [B]
            for idx in top_indices:
                self.access_counts[idx] += 1

        return output, confidence

    @torch.no_grad()
    def clear(self):
        """Reset all memories."""
        device = self.keys.device
        self.keys = torch.zeros(0, self.d_key, device=device)
        self.values = torch.zeros(0, self.d_val, device=device)
        self.access_counts = torch.zeros(0, dtype=torch.long, device=device)
        self.num_memories.zero_()
