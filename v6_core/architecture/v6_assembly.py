"""
SNAP-C1 V6: Master Assembly
============================
Wires all components into a single forward pass.

V5 components (kept):
  1. ObservationEncoder - tokenize + embed + compress context
  2. ResonanceStack - 8× dual-path blocks with dynamic skip (V6 innovation)
  3. ActionDecoder - tool selection + confidence + arg generation
  4. OutcomePredictor - P(success) before acting

V6 NEW innovations:
  - Dynamic Layer Skipping: ~45% layers auto-skipped → 2x speedup
  - Holographic Embedding: content-modulated distributed embedding

Full model (d=1024, 8 blocks): ~220M params, 100% trainable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_core.architecture.observation_encoder import ObservationEncoder, SegmentType
from v6_core.architecture.resonance_block import ResonanceStack
from v6_core.architecture.action_decoder import ActionDecoder, ToolID
from v6_core.architecture.outcome_predictor import OutcomePredictor
from v6_core.architecture.dml_ops import RMSNorm


class V6ResonanceModel(nn.Module):
    """
    V6 WHORMHOLE — Master Assembly.
    
    Key difference from V5:
    - ResonanceStack uses Dynamic Layer Skipping (2x speedup)
    - Holographic Embedding with content modulation

    Args:
        d_model: Hidden dimension (1024)
        n_blocks: Number of resonance blocks (8)
        n_heads: Attention heads (8)
        window_size: Sliding window size for local attention
        d_ff: FFN hidden dimension
        max_seq_len: Maximum input sequence length
        vocab_size: BPE vocabulary size
        K_hash: Number of hash functions for embedding
        d_hash: Dimension per hash table
        n_tools: Number of tools
        max_think_steps: Maximum internal THINK loops
        confidence_threshold: Below this, model loops back to THINK
        dropout: Dropout rate
        use_skip: Enable dynamic layer skipping (default: True)
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_blocks: int = 8,
        n_heads: int = 8,
        window_size: int = 128,
        d_ff: int = None,
        max_seq_len: int = 2048,
        vocab_size: int = 100279,
        K_hash: int = 8,
        d_hash: int = 128,
        n_tools: int = ToolID.COUNT,
        max_think_steps: int = 3,
        confidence_threshold: float = 0.5,
        dropout: float = 0.0,
        use_skip: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_think_steps = max_think_steps
        self.vocab_size = vocab_size
        self.use_skip = use_skip

        # Component 1: Observation Encoder with Holographic Embedding
        self.encoder = ObservationEncoder(
            d_model=d_model, K=K_hash, d_hash=d_hash,
            max_seq_len=max_seq_len
        )

        # Component 2: Resonance Stack with Dynamic Skip
        self.resonance = ResonanceStack(
            n_blocks=n_blocks, d_model=d_model,
            n_heads=n_heads, window_size=window_size,
            d_ff=d_ff, max_seq_len=max_seq_len,
            dropout=dropout, use_skip=use_skip
        )

        # Component 3: Action Decoder
        self.action_decoder = ActionDecoder(
            d_model=d_model, n_tools=n_tools,
            vocab_size=vocab_size,
            confidence_threshold=confidence_threshold
        )

        # Component 4: Outcome Predictor
        self.outcome_predictor = OutcomePredictor(
            d_model=d_model, n_tools=n_tools
        )

        # LM Head for next-token prediction
        self._lm_bottleneck = max(512, d_model // 2)
        self.lm_down = nn.Linear(d_model, self._lm_bottleneck)
        self.lm_up = nn.Linear(self._lm_bottleneck, vocab_size, bias=False)

    def forward_pretrain(self, token_ids: torch.Tensor,
                         type_ids: torch.Tensor = None,
                         labels: torch.Tensor = None):
        """
        Pre-training forward pass: next-token prediction.
        """
        context = self.encoder(token_ids, type_ids)
        hidden = self.resonance(context, causal=True)

        stride1_boundary = self.encoder.elastic.boundaries[0]
        stride1_slots = min(stride1_boundary, hidden.shape[1])
        hidden_s1 = hidden[:, :stride1_slots, :]

        lm_logits = self.lm_up(F.gelu(self.lm_down(hidden_s1)))

        result = {'logits': lm_logits}

        if labels is not None:
            B, S, V = lm_logits.shape
            labels_s1 = labels[:, :S]

            loss = F.cross_entropy(
                lm_logits.reshape(-1, V),
                labels_s1.reshape(-1),
                ignore_index=-100
            )
            result['loss'] = loss

        return result

    def forward_agent(self, token_ids: torch.Tensor,
                      type_ids: torch.Tensor = None):
        """
        Agent forward pass: produce an action decision.
        """
        context = self.encoder(token_ids, type_ids)

        slot_token_ids = self._build_slot_token_ids(token_ids)

        hidden = self.resonance(context, causal=False)

        action = self.action_decoder(hidden, context, slot_token_ids)

        think_steps = 0
        while action['should_think'].any() and think_steps < self.max_think_steps:
            with torch.no_grad():
                hidden = self.resonance(hidden, causal=False)
            action = self.action_decoder(hidden, context, slot_token_ids)
            think_steps += 1

        outcome = self.outcome_predictor(action['hidden'], action['tool_id'])

        return {
            'tool_id': action['tool_id'],
            'tool_logits': action['tool_logits'],
            'confidence': action['confidence'],
            'p_success': outcome['p_success'],
            'outcome_logit': outcome['outcome_logits'],
            'hidden': action['hidden'],
            'context': context,
            'slot_token_ids': slot_token_ids,
            'think_steps': think_steps,
        }

    def _build_slot_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Build token ID mapping for each context slot."""
        B, T = token_ids.shape
        elastic = self.encoder.elastic
        slot_ids = []

        prev = 0
        for i in range(elastic.n_levels):
            end = min(elastic.boundaries[i], T)
            if prev >= end:
                break

            chunk_ids = token_ids[:, prev:end]
            stride = elastic.strides[i]

            if stride > 1 and chunk_ids.shape[1] >= stride:
                n_groups = chunk_ids.shape[1] // stride
                chunk_ids = chunk_ids[:, :n_groups * stride:stride]

            slot_ids.append(chunk_ids)
            prev = end

        return torch.cat(slot_ids, dim=1)

    def generate_args(self, hidden: torch.Tensor,
                      context: torch.Tensor,
                      slot_token_ids: torch.Tensor,
                      max_tokens: int = 256) -> torch.Tensor:
        return self.action_decoder.generate_args(
            hidden, context, slot_token_ids, max_tokens
        )

    def count_parameters(self) -> dict:
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder': _count(self.encoder),
            'resonance': _count(self.resonance),
            'action_decoder': _count(self.action_decoder),
            'outcome_predictor': _count(self.outcome_predictor),
            'lm_head': (_count(self.lm_down) + _count(self.lm_up)),
            'total': total,
            'trainable': trainable,
            'utilization': f"{trainable / total * 100:.1f}%",
        }

    def get_skip_rate(self) -> float:
        """Get average layer skip rate from last forward pass."""
        if hasattr(self.resonance, 'get_skip_rate'):
            return self.resonance.get_skip_rate()
        return 0.0


def build_v6_local(use_skip: bool = True, dropout: float = 0.0) -> V6ResonanceModel:
    """Build V6 for local inference/training on RX 7600 (d=1024, 8 blocks)."""
    return V6ResonanceModel(
        d_model=1024, n_blocks=8, n_heads=8,
        window_size=128, max_seq_len=2048,
        vocab_size=100279, K_hash=8, d_hash=128,
        dropout=dropout, use_skip=use_skip,
    )


def build_v6_small(use_skip: bool = True, dropout: float = 0.0) -> V6ResonanceModel:
    """Build smaller V6 for testing (d=512, 4 blocks)."""
    return V6ResonanceModel(
        d_model=512, n_blocks=4, n_heads=4,
        window_size=64, max_seq_len=1024,
        vocab_size=100279, K_hash=8, d_hash=64,
        dropout=dropout, use_skip=use_skip,
    )


def build_v6_rtx6000(use_skip: bool = True, dropout: float = 0.0) -> V6ResonanceModel:
    """
    Build V6 optimized for RTX 6000 Ada (48GB VRAM).
    
    d_model=2048, n_blocks=16 → ~800M params
    Fits with: fp16 + gradient checkpointing + batch 8-16
    """
    return V6ResonanceModel(
        d_model=2048, n_blocks=16, n_heads=16,
        window_size=256, max_seq_len=2048,
        vocab_size=50257, K_hash=8, d_hash=256,
        dropout=dropout, use_skip=use_skip,
    )


def build_v6_large(use_skip: bool = True, dropout: float = 0.0) -> V6ResonanceModel:
    """
    Build large V6 for high-end GPUs (RTX 6000 Ada 48GB, A100 80GB).
    
    d_model=2560, n_blocks=20 → ~1B params
    """
    return V6ResonanceModel(
        d_model=2560, n_blocks=20, n_heads=20,
        window_size=256, max_seq_len=2048,
        vocab_size=50257, K_hash=8, d_hash=256,
        dropout=dropout, use_skip=use_skip,
    )
