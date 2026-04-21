"""
SNAP-C1 V5: Outcome Predictor (Component 6)
=============================================
Predicts P(success) BEFORE executing an action.

This is V5's strongest validated component (per stress test review).
It enables:
  1. Cost-aware decisions: don't run expensive tools if likely to fail
  2. Self-calibration: compare predicted vs actual outcomes → train on delta
  3. Active learning: seek tasks where prediction is uncertain (high info gain)

The predictor sees the hidden state (what the model understands) and the
proposed action (what it plans to do), and outputs a probability.

Training signal: binary cross-entropy against actual outcome (success/failure).
Over time, the predictor learns which situations are "easy" vs "hard" for
the model, creating an implicit difficulty curriculum.

DirectML safety: Pure nn.Linear + StableSigmoid. ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_core.utils.dml_ops import RMSNorm, stable_sigmoid


class OutcomePredictor(nn.Module):
    """
    Predicts probability of success for a proposed action.

    Args:
        d_model: Model hidden dimension
        n_tools: Number of tool types (for action embedding)
        d_hidden: Hidden dimension of the predictor MLP
    """

    def __init__(self, d_model: int = 1024, n_tools: int = 8,
                 d_hidden: int = 256):
        super().__init__()

        # Encode the proposed tool as a small embedding (scatter-free)
        self.tool_table = nn.Parameter(torch.randn(n_tools, d_hidden) * 0.02)
        self.n_tools = n_tools

        # MLP: hidden_state + tool_embed → P(success)
        self.predictor = nn.Sequential(
            nn.Linear(d_model + d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )
        self.norm = RMSNorm(d_model)

    def _tool_embed(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """Scatter-free tool embedding lookup.
        tool_ids: [B] long tensor → [B, d_hidden]
        """
        arange = torch.arange(self.n_tools, device=tool_ids.device)
        one_hot = (arange == tool_ids.unsqueeze(-1)).float()  # [B, n_tools]
        return one_hot @ self.tool_table  # [B, d_hidden]

    def forward(self, hidden_state: torch.Tensor,
                tool_ids: torch.Tensor) -> dict:
        """
        Args:
            hidden_state: [B, d_model] — pooled resonance output
            tool_ids: [B] — proposed tool to execute

        Returns:
            dict with 'p_success' (sigmoid applied) and 'logit' (raw, for loss)
        """
        h = self.norm(hidden_state)
        tool_emb = self._tool_embed(tool_ids)  # [B, d_hidden]

        combined = torch.cat([h, tool_emb], dim=-1)  # [B, d_model + d_hidden]
        logit = self.predictor(combined).squeeze(-1)  # [B]
        return {
            'p_success': stable_sigmoid(logit),  # for display/thresholding
            'logit': logit,                       # for loss computation
        }

    def loss(self, logit: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss using raw logits (numerically stable).
        logit: [B] — raw logit from forward()['logit']
        actual: [B] — 1.0 for success, 0.0 for failure
        """
        return F.binary_cross_entropy_with_logits(logit, actual.float())
