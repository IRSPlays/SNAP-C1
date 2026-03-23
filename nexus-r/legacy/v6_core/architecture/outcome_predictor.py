"""
SNAP-C1 V6: Outcome Predictor
=============================
Copied from V5 - predicts P(success) before acting.
"""

import torch
import torch.nn as nn

from v6_core.architecture.dml_ops import RMSNorm, stable_sigmoid


class OutcomePredictor(nn.Module):
    """
    Predicts whether an action will succeed BEFORE executing.
    
    Returns:
    - p_success: probability of success
    - outcome_logits: raw logits for loss computation
    """

    def __init__(self, d_model: int = 1024, n_tools: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_tools = n_tools

        # Encode tool one-hot + hidden state
        self.encoder = nn.Sequential(
            nn.Linear(d_model + n_tools, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )

        # Predict 3 outcomes: SUCCESS, PARTIAL, FAILURE
        self.outcome_head = nn.Linear(d_model // 2, 3)

        # Also predict continuous score
        self.score_head = nn.Sequential(
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_state: torch.Tensor,
                tool_id: torch.Tensor) -> dict:
        """
        Args:
            hidden_state: [B, d_model] - pooled resonance output
            tool_id: [B] - selected tool ID

        Returns:
            dict with 'p_success', 'outcome_logits', 'score'
        """
        B = hidden_state.shape[0]

        # One-hot encode tool
        tool_onehot = torch.zeros(B, self.n_tools, device=hidden_state.device)
        tool_onehot.scatter_(1, tool_id.unsqueeze(1), 1)

        # Encode tool + hidden
        encoded = self.encoder(torch.cat([hidden_state, tool_onehot], dim=-1))

        # Outcome logits
        outcome_logits = self.outcome_head(encoded)  # [B, 3]
        
        # P(success) = softmax on first class
        p_success = torch.softmax(outcome_logits, dim=-1)[:, 0]

        # Continuous score
        score = self.score_head(encoded).squeeze(-1)

        return {
            'p_success': p_success,
            'outcome_logits': outcome_logits,
            'score': score,
        }
