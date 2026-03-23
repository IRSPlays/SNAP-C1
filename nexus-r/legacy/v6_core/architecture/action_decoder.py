"""
SNAP-C1 V6: Action Decoder
==========================
Copied from V5 - structured action output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_core.architecture.dml_ops import (
    RMSNorm, stable_sigmoid, chunked_softmax, DML_GRUCell
)


class ToolID:
    SEARCH = 0
    READ = 1
    EDIT = 2
    RUN = 3
    THINK = 4
    RESPOND = 5
    RECALL = 6
    INTROSPECT = 7
    COUNT = 8
    NAMES = ["SEARCH", "READ", "EDIT", "RUN", "THINK", "RESPOND", "RECALL", "INTROSPECT"]


class PointerGeneratorHead(nn.Module):
    """Pointer-generator for argument text generation."""
    def __init__(self, d_model: int = 1024, vocab_size: int = 100279,
                 max_arg_tokens: int = 512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_arg_tokens = max_arg_tokens

        self._bottleneck = max(512, d_model // 2)
        self.vocab_down = nn.Linear(d_model, self._bottleneck)
        self.vocab_up = nn.Linear(self._bottleneck, vocab_size, bias=False)

        self.copy_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )

        self.p_gen_proj = nn.Linear(3 * d_model, 1)
        self.gru = DML_GRUCell(d_model, d_model, context_size=d_model)
        self.input_proj = nn.Linear(d_model, d_model)

    def _build_copy_dist(self, P_copy, context_token_ids, p_gen, device):
        B = P_copy.shape[0]
        copy_dist = torch.zeros(B, self.vocab_size, device=device, dtype=P_copy.dtype)

        for b in range(B):
            unique_ids = context_token_ids[b].unique()
            for uid in unique_ids:
                if uid < 0:
                    continue
                positions = (context_token_ids[b] == uid).float()
                copy_dist[b, uid] = (P_copy[b] * positions).sum()

        return (1 - p_gen) * copy_dist

    def forward(self, hidden_state: torch.Tensor,
                context: torch.Tensor,
                context_token_ids: torch.Tensor,
                max_tokens: int = None) -> torch.Tensor:
        B = hidden_state.shape[0]
        device = hidden_state.device
        max_t = max_tokens or self.max_arg_tokens

        h = hidden_state
        prev_embed = self.input_proj(hidden_state)
        output_ids = []

        for step in range(max_t):
            context_t, attn_weights = self.copy_attn(
                query=h.unsqueeze(1), key=context, value=context,
                need_weights=True, average_attn_weights=True
            )
            context_t = context_t.squeeze(1)
            P_copy = attn_weights.squeeze(1)

            vocab_logits = self.vocab_up(F.gelu(self.vocab_down(h)))
            P_vocab = chunked_softmax(vocab_logits, dim=-1)

            p_gen_input = torch.cat([h, context_t, prev_embed], dim=-1)
            p_gen = stable_sigmoid(self.p_gen_proj(p_gen_input))

            final_dist = p_gen * P_vocab
            if context_token_ids is not None:
                final_dist = final_dist + self._build_copy_dist(
                    P_copy, context_token_ids, p_gen, device
                )

            token_id = final_dist.argmax(dim=-1)
            output_ids.append(token_id)

            h = self.gru(context_t, h, context_t)
            prev_embed = context_t

        return torch.stack(output_ids, dim=1)


class ActionDecoder(nn.Module):
    """
    V5/V6 Structured Action Decoder.
    Given resonance output, produces tool selection + confidence + args.
    """
    def __init__(self, d_model: int = 1024, n_tools: int = ToolID.COUNT,
                 vocab_size: int = 100279, confidence_threshold: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_tools = n_tools
        self.confidence_threshold = confidence_threshold

        self.pool_proj = nn.Linear(d_model, d_model)
        self.pool_norm = RMSNorm(d_model)

        self.tool_head = nn.Linear(d_model, n_tools)
        self.confidence_head = nn.Linear(d_model, 1)

        self.arg_generator = PointerGeneratorHead(
            d_model=d_model, vocab_size=vocab_size
        )

    def _pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.pool_norm(self.pool_proj(pooled))

    def forward(self, resonance_output: torch.Tensor,
                context: torch.Tensor = None,
                context_token_ids: torch.Tensor = None):
        h = self._pool_sequence(resonance_output)

        tool_logits = self.tool_head(h)
        tool_id = tool_logits.argmax(dim=-1)

        confidence = stable_sigmoid(self.confidence_head(h)).squeeze(-1)
        should_think = confidence < self.confidence_threshold

        return {
            'tool_logits': tool_logits,
            'tool_id': tool_id,
            'confidence': confidence,
            'should_think': should_think,
            'hidden': h,
        }

    def generate_args(self, hidden: torch.Tensor,
                      context: torch.Tensor,
                      context_token_ids: torch.Tensor,
                      max_tokens: int = 256) -> torch.Tensor:
        return self.arg_generator(hidden, context, context_token_ids,
                                 max_tokens=max_tokens)
