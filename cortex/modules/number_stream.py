"""Number-as-Voltage (NAV) — differentiable arithmetic circuit.

Numbers are NOT tokens. They are analog voltage signals processed
through a small learned arithmetic circuit that can ADD, SUBTRACT,
MULTIPLY through weight configurations, then cross-attend to the
main text stream.

Components:
    - NumberEncoder: maps number values to embeddings
    - ArithmeticCircuit: pair-wise learned +, -, × operations
    - NumberCrossAttention: cross-attends number stream → text stream

Total params: ~2M (tiny addition to the architecture).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NumberStream(nn.Module):
    """Processes number values through arithmetic circuits, cross-attends to text."""

    def __init__(self, d_model: int = 512, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model

        # Encode scalar number values → d_model space
        self.num_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Pairwise arithmetic: given embedding of a and b, predict a+b and a*b
        self.add_circuit = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.mul_circuit = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.sub_circuit = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Cross-attention: text stream attends to number stream
        self.num_to_text_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )

        # Output projection
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.num_encoder[0], self.num_encoder[2],
                  self.add_circuit[0], self.add_circuit[2],
                  self.mul_circuit[0], self.mul_circuit[2],
                  self.sub_circuit[0], self.sub_circuit[2]]:
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, num_values: torch.Tensor, num_mask: torch.Tensor,
                text_stream: torch.Tensor) -> torch.Tensor:
        """Process number values and cross-attend to text.

        Args:
            num_values: [B, T] — scalar number values at each position (0 for non-numbers)
            num_mask: [B, T] — bool mask, True where num_values != 0
            text_stream: [B, T, d_model] — main text encoder output

        Returns:
            num_context [B, T, d_model] — number-augmented text representations
        """
        B, T, D = text_stream.shape

        # Encode numbers (only at number positions)
        num_embeds = self.num_encoder(num_values.unsqueeze(-1))  # [B, T, D]

        # Pairwise arithmetic on adjacent number pairs
        # For each adjacent pair of number positions (i, i+1), compute a+b, a-b, a*b
        pair_add = self.add_circuit(torch.cat([
            num_embeds[:, :-1], num_embeds[:, 1:]], dim=-1
        ))  # [B, T-1, D]

        pair_sub = self.sub_circuit(torch.cat([
            num_embeds[:, :-1], num_embeds[:, 1:]], dim=-1))

        pair_mul = self.mul_circuit(torch.cat([
            num_embeds[:, :-1], num_embeds[:, 1:]], dim=-1))

        # Combine arithmetic results (pad to match T)
        arith_out = pair_add + pair_sub + pair_mul  # [B, T-1, D]
        arith_out = F.pad(arith_out, (0, 0, 0, 1))  # [B, T, D]

        # Combine with original number embeddings
        num_stream = num_embeds + arith_out  # [B, T, D]

        # Cross-attention: text attends to number stream
        num_context, _ = self.num_to_text_attn(
            text_stream, num_stream, num_stream,
            key_padding_mask=(~num_mask.bool())
        )  # [B, T, D]

        # Fuse with original text
        fused = self.out_proj(torch.cat([text_stream, num_context], dim=-1))
        return fused
