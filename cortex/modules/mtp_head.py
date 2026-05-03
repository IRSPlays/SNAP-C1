"""Multi-Token Predictor — predicts 4 future tokens with shared projection.

Based on: DeepSeek V3 (Dec 2024, arXiv:2412.19437)

4 heads project from hidden state at different offsets:
    Head 0: next token      (main head, tied to embedding)
    Head 1: token + 2       (shared projection, separate RMSNorm)
    Head 2: token + 3       (shared projection, separate RMSNorm)
    Head 3: token + 4       (shared projection, separate RMSNorm)

Extra heads share ONE Linear(vocab) matrix for parameter efficiency.
Each head gets its own RMSNorm for per-horizon specialization.
Saves 2 × d_model × vocab_size params (~66% of extra head params).

Total: L = CE_0 + 0.3·(CE_1 + CE_2 + CE_3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiTokenPredictor(nn.Module):
    def __init__(self, d_model: int = 512, vocab_size: int = 4096,
                 n_extra_heads: int = 3, mtp_weight: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_extra_heads = n_extra_heads
        self.mtp_weight = mtp_weight

        self.norm = nn.RMSNorm(d_model)
        self.main_head = nn.Linear(d_model, vocab_size, bias=False)

        # Shared projection for all extra heads (DeepSeek V3 style)
        # Each head gets its own RMSNorm for per-horizon specialization
        self.shared_head = nn.Linear(d_model, vocab_size, bias=False)
        self.extra_norms = nn.ModuleList([
            nn.RMSNorm(d_model) for _ in range(n_extra_heads)
        ])

    def tie_main_weight(self, embedding: nn.Embedding):
        self.main_head.weight = embedding.weight

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        x = self.norm(x)
        logits_main = self.main_head(x)

        result = {'logits': logits_main}

        if labels is not None:
            B, T, V = logits_main.shape
            losses = []
            # Main head: logits[t] predicts labels[t] (next token from input[t])
            # labels is already shifted by 1 (labels[k] = input[k+1])
            losses.append(F.cross_entropy(
                logits_main[:, :-1].reshape(-1, V),
                labels[:, :-1].reshape(-1),
                ignore_index=-100,
            ))

            for head_idx, norm in enumerate(self.extra_norms):
                offset = head_idx + 2  # predict token at offset steps ahead
                logits_extra = self.shared_head(norm(x[:, :-offset]))
                # labels[k] = token[k+1], so we need labels[offset-1 : T-1]
                labels_extra = labels[:, offset - 1: -1]
                if labels_extra.numel() > 0:
                    losses.append(self.mtp_weight * F.cross_entropy(
                        logits_extra.reshape(-1, V),
                        labels_extra.reshape(-1),
                        ignore_index=-100,
                    ))

            result['loss'] = sum(losses)
            result['ce_loss'] = losses[0]
            result['mtp_losses'] = losses[1:] if len(losses) > 1 else []

        return result
