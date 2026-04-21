"""Cortex V1 — Full Model Assembly

Brain-inspired language model combining:
    1. Encoder (BPE embeddings + positional encoding)
    2. Predictive Coder (only prediction errors propagate)
    3. LTC-RNN Cortex (adaptive time-constant reasoning)
    4. Neuromodulator (deterministic: dopamine, norepinephrine, serotonin)
    5. Hopfield Memory (one-shot hippocampal storage)
    6. Decoder (logits with weight tying)

Phase 1: Encoder → PredictiveCoder → LTCRNN → Decoder (no memory)
Phase 2: Full pipeline with Neuromodulator + HopfieldMemory
"""

import torch
import torch.nn as nn

from .modules import (
    Encoder,
    PredictiveCoder,
    LTCRNN,
    Neuromodulator,
    HopfieldMemory,
    Decoder,
)


class CortexV1(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_key: int = 512,
        n_pc_layers: int = 3,
        max_seq_len: int = 512,
        max_memories: int = 10_000,
        dropout: float = 0.1,
        use_memory: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_memory = use_memory

        # Core modules (Phase 1)
        self.encoder = Encoder(vocab_size, d_model, max_seq_len, dropout)
        self.pc = PredictiveCoder(d_model, n_pc_layers)
        self.cortex = LTCRNN(d_model, d_model)
        self.decoder = Decoder(d_model, vocab_size, tie_weights=self.encoder.token_emb)

        # Memory modules (Phase 2 — initialized but not used in Phase 1)
        self.neuromod = Neuromodulator()
        self.memory = HopfieldMemory(d_model, d_key, d_model, max_memories)

    def forward(
        self,
        token_ids: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            token_ids: [B, T] input token IDs
            h0: [B, d_model] optional initial hidden state

        Returns:
            dict with:
                logits: [B, T, vocab_size]
                prediction_error: [B, T] — PC error per position
                h_final: [B, d_model] — last hidden state
        """
        # 1. Encode tokens
        embeddings = self.encoder(token_ids)  # [B, T, d_model]

        # 2. Predictive coding — extract surprising features
        pc_out, pred_error = self.pc(embeddings)  # [B, T, d_model], [B, T]

        if not self.use_memory:
            # Phase 1: simple forward pass
            cortex_out, h_final = self.cortex(pc_out, h0)  # [B, T, d_model]
            logits = self.decoder(cortex_out)               # [B, T, vocab_size]

            return {
                'logits': logits,
                'prediction_error': pred_error,
                'h_final': h_final,
            }

        # Phase 2: full pipeline with neuromodulator + memory
        B, T, D = pc_out.shape
        device = pc_out.device

        h = h0 if h0 is not None else torch.zeros(B, D, device=device)
        outputs = []

        for t in range(T):
            x_t = pc_out[:, t, :]          # [B, d_model]
            eps_t = pred_error[:, t]        # [B]

            # Read from memory
            mem_out, mem_conf = self.memory.read(h)  # [B, d_model], [B]

            # Compute neuromodulatory signals
            signals = self.neuromod(eps_t, mem_conf)

            # LTC step with norepinephrine modulating dt
            # High nu → more cortex effort → larger time step (more state change)
            dt = 0.5 + signals['nu'].mean().item()  # scalar dt for the batch
            h = self.cortex.cell(x_t, h, dt=dt)

            # Write to memory if dopamine is high
            self.memory.write(h, signals['delta'])

            # Blend memory and cortex output
            sigma = signals['sigma'].unsqueeze(-1)  # [B, 1]
            blended = sigma * mem_out + (1 - sigma) * h  # [B, d_model]
            outputs.append(blended)

        outputs = torch.stack(outputs, dim=1)  # [B, T, d_model]
        logits = self.decoder(outputs)          # [B, T, vocab_size]

        return {
            'logits': logits,
            'prediction_error': pred_error,
            'h_final': h,
        }

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per module."""
        counts = {}
        for name, module in [
            ('encoder', self.encoder),
            ('predictive_coder', self.pc),
            ('cortex_ltc', self.cortex),
            ('decoder', self.decoder),
            ('neuromodulator', self.neuromod),
            ('memory', self.memory),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['total_unique'] = sum(
            p.numel() for p in {id(p): p for p in self.parameters()}.values()
        )
        return counts
