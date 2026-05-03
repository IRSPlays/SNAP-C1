"""Eidos V1 — Differential Predictive Memory Transformer.

Architecture flow (FIXED — fully differentiable end-to-end):
    tokens → EidosEncoder (Diff Attention x 4 layers)
        ├→ z (per-token embeddings, [B, T, D])
        ├→ pooled (sequence summary, [B, D])
        │
        ├→ PredictiveCoder: z_hat from z_prev ⊕ pooled; outputs ε, cosine_dist
        │
        ├→ Neuromodulator: [δ, ν, σ, α] = f(ε, memory_match)
        │   ├→ δ → DA/NE precursor (logged, not directly used since memory
        │   │       uses cosine_dist directly as surprise)
        │   ├→ ν → LTC iteration budget per batch
        │   └→ σ → direct-memory vs cortex-reasoning blend
        │
        ├→ NeuralMemory (DIFFERENTIABLE): write(z, cos_dist) + read(z) → h_mem
        │   Gradient flows through all projections.
        │   M_new = (1−α)·M_prev + α·Σ(σ(cos_dist)·K⊗V)
        │   h_mem = normalize(M_new·Q^T / √D)
        │
        ├→ LTCCortex: iter = ν; h_cortex = LTC(z ⊕ h_mem, iter)
        │   Memory-augmented reasoning: LTC sees current + remembered
        │
        ├→ Integrator: out = σ·h_mem + (1−σ)·h_cortex
        │   σ→1: fast memory recall
        │   σ→0: fresh reasoning with memory context
        │
        └→ MultiTokenPredictor: 4 heads → logits, CE + 0.3·MTP

Based on: Diff Transformer (ICLR 2025) + Titans (Dec 2024) + DeepSeek V3 (Dec 2024) + LTC (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .modules.encoder import EidosEncoder
from .modules.predictive_coder import PredictiveCoder
from .modules.neuromodulator import Neuromodulator
from .modules.neural_memory import NeuralMemory
from .modules.ltc_cortex import LTCCortex
from .modules.mtp_head import MultiTokenPredictor


class EidosV1(nn.Module):
    def __init__(self, vocab_size: int = 4096, d_model: int = 512,
                 n_heads: int = 8, n_kv_heads: int = 4, n_layers: int = 4,
                 max_seq_len: int = 512, dropout: float = 0.0,
                 memory_mode: str = 'momentum',
                 embed_weights: Optional[torch.Tensor] = None,
                 num_values: Optional[torch.Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.memory_mode = memory_mode

        self.encoder = EidosEncoder(vocab_size, d_model, n_heads, n_kv_heads,
                                    n_layers, max_seq_len, dropout, embed_weights,
                                    num_values)

        self.predictive_coder = PredictiveCoder(d_model)

        self.neuromodulator = Neuromodulator()

        self.neural_memory = NeuralMemory(d_model)

        self.ltc_cortex = LTCCortex(d_model)

        self.mtp = MultiTokenPredictor(d_model, vocab_size)
        self.mtp.tie_main_weight(self.encoder.token_emb)

        self.num_head = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.num_head.weight, std=0.01)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                answer_values: Optional[torch.Tensor] = None) -> Dict:
        B, T = input_ids.shape

        # ── 1. Differential encoding ──
        z, pooled = self.encoder(input_ids)

        # ── 1a. Problem complexity: count numbers in prompt ──
        num_count = torch.zeros(B, device=input_ids.device, dtype=torch.long)
        if self.encoder.num_proj is not None:
            is_number = (self.encoder.num_values[input_ids] != 0).float()  # [B, T]
            num_count = is_number.sum(dim=1).long()

        # ── 2. Predictive coding: embed prediction + value prediction ──
        z_shifted = torch.cat([
            z[:, :1, :].detach(),
            z[:, :-1, :]
        ], dim=1)
        z_hat, error, cosine_dist, value_pred = self.predictive_coder(z_shifted, z, pooled)

        # ── 3. Differentiable memory: δ-gated write then read ──
        h_mem = self.neural_memory(z, surprise=cosine_dist)

        # ── 4. Memory relevance signal ──
        memory_match = F.cosine_similarity(
            h_mem.reshape(B, -1), z.reshape(B, -1), dim=-1
        )
        memory_match = memory_match.unsqueeze(-1)

        # ── 5. Neuromodulation: δ, ν from complexity, σ with soft blend ──
        delta, norepi, serotonin, acetyl = self.neuromodulator(
            cosine_dist, memory_match, num_count=num_count
        )

        # ── 6. LTC with memory-augmented input (proper iteration depth) ──
        max_iters = int(norepi.max().item()) if norepi.numel() > 0 else 2
        max_iters = max(2, min(max_iters, 16))
        h_cortex = self.ltc_cortex(z, iterations=max_iters, memory=h_mem)

        # ── 7. Blend: smooth interpolation between recall and reasoning ──
        sigma = serotonin.view(B, 1, 1)
        out = sigma * h_mem + (1 - sigma) * h_cortex

        # ── 8. Multi-token prediction ──
        result = self.mtp(out, labels=labels)

        # ── 9. Auxiliary losses ──
        if labels is not None and 'loss' in result:
            result['loss'] = result['loss'] + 0.01 * cosine_dist.mean()
            result['pred_aux_loss'] = 0.01 * cosine_dist.mean()

        # ── 10. Per-position number value prediction ──
        if self.encoder.num_proj is not None:
            actual_vals = self.encoder.num_values[input_ids]  # [B, T]
            is_num = is_number  # [B, T]

            # num_head predicts next token's numeric value at each position
            num_pred = self.num_head(out).squeeze(-1)  # [B, T]
            num_pred_aligned = num_pred[:, :-1]  # predict next
            target_vals = actual_vals[:, 1:]     # value of next token
            target_mask = is_num[:, 1:]          # only where next token is a number

            num_loss_raw = F.mse_loss(num_pred_aligned, target_vals, reduction='none')
            n_num = target_mask.sum()
            if n_num > 0:
                num_loss = 0.1 * (num_loss_raw * target_mask).sum() / n_num
                result['loss'] = result.get('loss', 0.0) + num_loss
                result['num_loss'] = num_loss
                result['num_pred'] = num_pred

            # PredictiveCoder value prediction loss
            val_pred_loss_raw = F.mse_loss(value_pred, actual_vals, reduction='none')
            n_val = is_num.sum()
            if n_val > 0:
                val_loss = 0.05 * (val_pred_loss_raw * is_num).sum() / n_val
                result['loss'] = result.get('loss', 0.0) + val_loss
                result['val_pred_loss'] = val_loss

        result['thought'] = out
        result['prediction_error'] = error
        result['cosine_dist'] = cosine_dist
        result['memory_match'] = memory_match
        result['dopamine'] = delta
        result['serotonin'] = serotonin
        result['iterations'] = max_iters

        return result

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        embed = self.encoder.token_emb.weight.numel()
        return {
            'total': total,
            'unique': total - embed,
            'embedding': embed,
            'encoder': sum(p.numel() for p in self.encoder.layers.parameters()),
            'predictive_coder': sum(p.numel() for p in self.predictive_coder.parameters()),
            'neural_memory': sum(p.numel() for p in self.neural_memory.parameters()
                                 if p.requires_grad),
            'ltc_cortex': sum(p.numel() for p in self.ltc_cortex.parameters()),
            'mtp_heads': sum(p.numel() for p in self.mtp.parameters()) - embed,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 eos_token_id: Optional[int] = None,
                 enable_self_verify: bool = True,
                 enable_self_consistency: bool = True,
                 enable_skip_ltc: bool = True,
                 verify_threshold: float = 0.5,
                 verify_retries: int = 1) -> torch.Tensor:
        """Generate with self-verification, self-consistency, and adaptive compute.

        - Self-verification: if prediction error after generation is high, re-generate
          with more LTC iterations (up to 16).
        - Self-consistency: extra MTP heads vote on the next token. If majority
          disagrees with main head, use the consensus token.
        - Skip LTC: if problem has < 3 numbers, skip LTC entirely (fast memory recall).
        """
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            input_ids = input_ids.to(device)
            B = input_ids.size(0)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            # 7d: Check problem complexity — skip LTC on easy problems
            if enable_skip_ltc and self.encoder.num_proj is not None:
                num_count = (self.encoder.num_values[input_ids] != 0).sum().item()
                skip_ltc = num_count < 3
            else:
                skip_ltc = False

            for _ in range(max_new_tokens):
                idx = input_ids[:, -min(input_ids.size(1), 512):]

                if skip_ltc:
                    # Fast path: encoder + memory only, no LTC
                    z, pooled = self.encoder(idx)
                    h_mem = self.neural_memory(z, surprise=torch.zeros_like(z[:, :, 0]))
                    result = self.mtp(h_mem)
                    logits = result['logits'][:, -1, :]
                else:
                    out = self.forward(idx)
                    logits = out['logits'][:, -1, :]

                    # 7b: Self-consistency — MTP heads vote on next token
                    if enable_self_consistency and 'thought' in out:
                        thought = out['thought']
                        main_token = torch.argmax(logits, dim=-1)
                        # Extra heads predict at their trained offsets from the last position
                        extra_tokens = []
                        T = thought.size(1)
                        for head_idx, norm in enumerate(self.mtp.extra_norms):
                            offset = head_idx + 2
                            if T > offset:
                                extra_logits = self.mtp.shared_head(norm(thought[:, -(offset+1):-offset]))
                                extra_tokens.append(torch.argmax(extra_logits[:, 0], dim=-1))
                        if extra_tokens:
                            votes = torch.stack([main_token] + extra_tokens)  # [N_heads, B]
                            for b in range(B):
                                b_votes = votes[:, b]
                                unique, counts = torch.unique(b_votes, return_counts=True)
                                majority = unique[torch.argmax(counts)]
                                if counts.max() > 1:  # at least 2 agree
                                    logits[b, :] = float('-inf')
                                    logits[b, majority] = 0.0

                if temperature <= 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-5)
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                for b in range(B):
                    if eos_token_id is not None and int(next_token[b, 0].item()) == eos_token_id:
                        finished[b] = True

                input_ids = torch.cat([input_ids, next_token], dim=1)
                if finished.all():
                    break

            # 7a: Self-verification — check prediction error, re-generate if high
            if enable_self_verify and verify_retries > 0 and self.encoder.num_proj is not None and not skip_ltc:
                idx = input_ids[:, -min(input_ids.size(1), 512):]
                z, pooled = self.encoder(idx)
                z_shifted = torch.cat([z[:, :1, :].detach(), z[:, :-1, :]], dim=1)
                _, _, cosine_dist, _ = self.predictive_coder(z_shifted, z, pooled)
                mean_surprise = cosine_dist.mean().item()

                if mean_surprise > verify_threshold:
                    # Model is uncertain — re-generate with max LTC iterations
                    longer_ids = input_ids[:, :1]  # keep only prompt
                    finished[:] = False
                    for _ in range(max_new_tokens):
                        idx = longer_ids[:, -min(longer_ids.size(1), 512):]
                        z, pooled = self.encoder(idx)
                        h_mem_verify = self.neural_memory(
                            z, surprise=torch.zeros_like(z[:, :, 0])
                        )
                        h_cortex_verify = self.ltc_cortex(z, iterations=16, memory=h_mem_verify)
                        out_verify = h_cortex_verify  # full reasoning, no memory blend
                        result_verify = self.mtp(out_verify)
                        v_logits = result_verify['logits'][:, -1, :]
                        if temperature <= 0:
                            nt = torch.argmax(v_logits, dim=-1, keepdim=True)
                        else:
                            v_logits = v_logits / max(temperature, 1e-5)
                            if top_k > 0:
                                tv, _ = torch.topk(v_logits, min(top_k, v_logits.size(-1)))
                                v_logits = v_logits.masked_fill(v_logits < tv[:, [-1]], float('-inf'))
                            v_probs = F.softmax(v_logits, dim=-1)
                            nt = torch.multinomial(v_probs, num_samples=1)
                        for b in range(B):
                            if eos_token_id is not None and int(nt[b, 0].item()) == eos_token_id:
                                finished[b] = True
                        longer_ids = torch.cat([longer_ids, nt], dim=1)
                        if finished.all():
                            break
                    input_ids = longer_ids

            return input_ids
        finally:
            self.train(was_training)
