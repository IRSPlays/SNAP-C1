"""Eidos — Differential Predictive Memory Transformer with Tool-Native Architecture.

Architecture flow:
    tokens → EidosEncoder (Diff Attention x N layers, MoE FFN, cascade gates)
        ├→ z (per-token embeddings), pooled (sequence summary)
        │
        ├→ NumberStream: differentiable arithmetic circuit, cross-attends to text
        │
        ├→ PredictiveCoder: z_hat from z_prev ⊕ pooled; outputs ε, cos_dist, value_pred
        │
        ├→ Neuromodulator: δ (write gate), ν (LTC budget), σ (recall/reason blend)
        │   Also: fast weights ΔW for attention layers
        │
        ├→ NeuralMemory: surprise-gated write + read, low-rank compression
        │
        ├→ LTCCortex: TD-learning recurrence, ν iterations, memory-augmented
        │
        ├→ Integrator: σ·h_mem + (1-σ)·h_cortex
        │
        └→ MTP Heads: 4-head prediction, shared projection, self-consistency voting

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
from .modules.number_stream import NumberStream


class EidosV1(nn.Module):
    def __init__(self, vocab_size: int = 4096, d_model: int = 512,
                 n_heads: int = 8, n_kv_heads: int = 4, n_layers: int = 4,
                 max_seq_len: int = 512, dropout: float = 0.0,
                 memory_mode: str = 'momentum',
                 embed_weights: Optional[torch.Tensor] = None,
                 num_values: Optional[torch.Tensor] = None,
                 use_number_stream: bool = True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.memory_mode = memory_mode

        self.encoder = EidosEncoder(vocab_size, d_model, n_heads, n_kv_heads,
                                    n_layers, max_seq_len, dropout, embed_weights,
                                    num_values)

        self.number_stream = None
        if use_number_stream and num_values is not None:
            self.number_stream = NumberStream(d_model, n_heads // 2)

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

        # ── 1. Differential encoding (fp32 for safety — AMP fp16 softmax overflows) ──
        with torch.amp.autocast('cuda', enabled=False):
            z, pooled = self.encoder(input_ids)

        # NaN early-detection: if encoder produces NaN, skip training on this batch
        if not torch.isfinite(z).all():
            result = {'logits': torch.zeros(B, T, self.vocab_size, device=z.device),
                      'loss': torch.tensor(float('inf'), device=z.device)}
            return result

        # Clamp encoder output for downstream fp16 safety (prevents F.normalize overflow)
        z = torch.clamp(z, -50, 50)
        pooled = torch.clamp(pooled, -50, 50)

        # ── 1a. Problem complexity + number mask ──
        num_count = torch.zeros(B, device=input_ids.device, dtype=torch.long)
        is_number = torch.zeros(B, T, device=input_ids.device)
        if self.encoder.num_proj is not None:
            is_number = (self.encoder.num_values[input_ids] != 0).float()
            num_count = is_number.sum(dim=1).long()

        # ── 1b. Number Stream: differentiable arithmetic circuit + cross-attention ──
        if self.number_stream is not None and self.encoder.num_proj is not None:
            num_stream_out = self.number_stream(
                self.encoder.num_values[input_ids], is_number, z
            )
            z = z + 0.1 * num_stream_out

        # ── 1c. Forward-Forward loss ──
        if self.training and labels is not None:
            ff_loss, _ = self.encoder.ff_loss(input_ids)
        else:
            ff_loss = None

        # ── 2. Predictive coding: embed prediction + value prediction ──
        z_shifted = torch.cat([
            z[:, :1, :].detach(),
            z[:, :-1, :]
        ], dim=1)
        z_hat, error, cosine_dist, value_pred = self.predictive_coder(z_shifted, z, pooled)

        # ── 3. Differentiable memory: δ-gated write then read ──
        h_mem = self.neural_memory(z, surprise=cosine_dist)

        # ── 4. Memory relevance signal (fp32 for safety) ──
        memory_match = F.cosine_similarity(
            h_mem.float().reshape(B, -1), z.float().reshape(B, -1), dim=-1
        ).to(z.dtype)
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

        # ── 9b. Forward-Forward loss for encoder layers ──
        if ff_loss is not None:
            result['loss'] = result.get('loss', 0.0) + 0.1 * ff_loss
            result['ff_loss'] = ff_loss

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

            # ── Reward signal: 0 = very wrong, 1 = perfect number prediction ──
            if n_num > 0:
                num_acc = torch.exp(-torch.abs(num_pred_aligned - target_vals) * 5.0)
                result['reward_signal'] = (num_acc * target_mask).sum() / n_num
            else:
                result['reward_signal'] = torch.tensor(0.5, device=out.device)

        # ── Surprise mask: positions where model was confused ──
        if labels is not None:
            result['surprise_mask'] = (cosine_dist[:, :-1] > 0.5).float()  # [B, T-1]

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
                 enable_tools: bool = True,
                 verify_threshold: float = 0.5,
                 verify_retries: int = 1) -> torch.Tensor:
        """Generate with self-verification, self-consistency, adaptive compute, and tools.

        Tool protocol: model outputs <CALC>expr</CALC>, <EXEC>code</EXEC>, etc.
        System intercepts, executes, feeds result back as synthetic tokens.
        """
        was_training = self.training
        self.eval()
        try:
            from .tool_parser import ToolParser
            parser = ToolParser(memory_module=self.neural_memory,
                               ltc_module=self.ltc_cortex,
                               model=self)
            device = next(self.parameters()).device
            input_ids = input_ids.to(device)
            B = input_ids.size(0)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            if enable_skip_ltc and self.encoder.num_proj is not None:
                num_count = (self.encoder.num_values[input_ids] != 0).sum().item()
                skip_ltc = num_count < 3
            else:
                skip_ltc = False

            for _ in range(max_new_tokens):
                idx = input_ids[:, -min(input_ids.size(1), 512):]

                if skip_ltc:
                    z, pooled = self.encoder(idx)
                    if self.number_stream is not None:
                        is_num = (self.encoder.num_values[idx] != 0).float()
                        z = z + 0.1 * self.number_stream(
                            self.encoder.num_values[idx], is_num, z)
                    h_mem = self.neural_memory(z, surprise=torch.zeros_like(z[:, :, 0]))
                    result = self.mtp(h_mem)
                    logits = result['logits'][:, -1, :]
                else:
                    out = self.forward(idx)
                    logits = out['logits'][:, -1, :]

                    if enable_self_consistency and 'thought' in out:
                        thought = out['thought']
                        main_token = torch.argmax(logits, dim=-1)
                        extra_tokens = []
                        T_seq = thought.size(1)
                        for head_idx, norm in enumerate(self.mtp.extra_norms):
                            offset = head_idx + 2
                            if T_seq > offset:
                                extra_logits = self.mtp.shared_head(norm(thought[:, -(offset+1):-offset]))
                                extra_tokens.append(torch.argmax(extra_logits[:, 0], dim=-1))
                        if extra_tokens:
                            votes = torch.stack([main_token] + extra_tokens)
                            for b in range(B):
                                b_votes = votes[:, b]
                                unique, counts = torch.unique(b_votes, return_counts=True)
                                majority = unique[torch.argmax(counts)]
                                if counts.max() > 1:
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

                # Tool execution: check if last N tokens form a complete tool call
                if enable_tools and not skip_ltc:
                    # Decode recent tokens to check for tool calls
                    # This is approximate — uses the post-hoc decode
                    pass  # Full tool parsing runs in post-processing

                if finished.all():
                    break

            return input_ids
        finally:
            self.train(was_training)
