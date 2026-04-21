"""
SNAP-C1 V5: Master Assembly
=============================
Wires all 6 neural components into a single forward pass:

  1. ObservationEncoder  — tokenize + embed + compress context
  2. ResonanceStack      — 8× dual-path spectral/local blocks
  3. ActionDecoder       — tool selection + confidence + arg generation
  4. OutcomePredictor    — P(success) before acting

The THINK loop is internal: if confidence < threshold, the model
re-processes through the resonance blocks with its own "reasoning"
appended to context (up to max_think_steps times).

Full model at local scale (d=1024, 8 blocks):  ~216M params, 100% trainable
Full model at RunPod scale (d=1536, 12 blocks): ~1.38B params, 100% trainable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_core.architecture.observation_encoder import ObservationEncoder, SegmentType
from v5_core.architecture.resonance_block import ResonanceStack
from v5_core.architecture.action_decoder import ActionDecoder, ToolID
from v5_core.architecture.outcome_predictor import OutcomePredictor
from v5_core.utils.dml_ops import RMSNorm, chunked_softmax


class V5ResonanceModel(nn.Module):
    """
    V5 Living Model — Master Assembly.

    Args:
        d_model: Hidden dimension (1024 local, 1536 RunPod)
        n_blocks: Number of resonance blocks (8 local, 12 RunPod)
        n_heads: Attention heads (8 local, 12 RunPod)
        window_size: Sliding window size for local attention
        d_ff: FFN hidden dimension (default: 4 × d_model)
        max_seq_len: Maximum input sequence length
        vocab_size: BPE vocabulary size (tiktoken cl100k_base = 100279)
        K_hash: Number of hash functions for embedding
        d_hash: Dimension per hash table
        n_tools: Number of tools
        max_think_steps: Maximum internal THINK loops before forced action
        confidence_threshold: Below this, model loops back to THINK
        dropout: Dropout rate
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
    ):
        super().__init__()
        self.d_model = d_model
        self.max_think_steps = max_think_steps
        self.vocab_size = vocab_size

        # Component 1+3+4: Observation Encoder (embedding + elastic context)
        self.encoder = ObservationEncoder(
            d_model=d_model, K=K_hash, d_hash=d_hash,
            max_seq_len=max_seq_len
        )

        # Component 2: Resonance Blocks
        self.resonance = ResonanceStack(
            n_blocks=n_blocks, d_model=d_model,
            n_heads=n_heads, window_size=window_size,
            d_ff=d_ff, max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Component 5: Action Decoder
        self.action_decoder = ActionDecoder(
            d_model=d_model, n_tools=n_tools,
            vocab_size=vocab_size,
            confidence_threshold=confidence_threshold
        )

        # Component 6: Outcome Predictor
        self.outcome_predictor = OutcomePredictor(
            d_model=d_model, n_tools=n_tools
        )

        # LM Head for next-token prediction (pre-training)
        # Factored: d_model → bottleneck → vocab_size
        # Bottleneck scales with d_model (min 512 for small configs)
        self._lm_bottleneck = max(512, d_model // 2)
        self.lm_down = nn.Linear(d_model, self._lm_bottleneck)
        self.lm_up = nn.Linear(self._lm_bottleneck, vocab_size, bias=False)

    def forward_pretrain(self, token_ids: torch.Tensor,
                         type_ids: torch.Tensor,
                         labels: torch.Tensor = None):
        """
        Pre-training forward pass: next-token prediction.

        IMPORTANT: Loss is computed ONLY on stride-1 (uncompressed) slots.
        Compressed slots represent groups of 4-16 tokens and can't map
        cleanly to single next-token labels.

        Args:
            token_ids: [B, T] — input token IDs
            type_ids: [B, T] — segment types
            labels: [B, T] — target token IDs for loss (shifted by 1)

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Encode (includes elastic context compression)
        context = self.encoder(token_ids, type_ids)  # [B, slots, d]

        # Process through resonance blocks (causal for pretraining)
        hidden = self.resonance(context, causal=True)  # [B, slots, d]

        # LM head — only compute on stride-1 (uncompressed) slots
        # Stride-1 slots have 1:1 correspondence with original positions;
        # compressed slots can't map to single next-token labels.
        # Computing LM head only on stride-1 saves ~3-4 GB VRAM.
        stride1_boundary = self.encoder.elastic.boundaries[0]
        stride1_slots = min(stride1_boundary, hidden.shape[1])
        hidden_s1 = hidden[:, :stride1_slots, :]  # [B, s1, d]

        lm_logits = self.lm_up(F.gelu(self.lm_down(hidden_s1)))  # [B, s1, vocab]

        result = {'logits': lm_logits}

        if labels is not None:
            B, S, V = lm_logits.shape
            labels_s1 = labels[:, :S]  # [B, s1]

            loss = F.cross_entropy(
                lm_logits.reshape(-1, V),
                labels_s1.reshape(-1),
                ignore_index=-100
            )
            result['loss'] = loss

        return result

    def forward_agent(self, token_ids: torch.Tensor,
                      type_ids: torch.Tensor):
        """
        Agent forward pass: produce an action decision.
        Uses bidirectional global attention (no causality needed —
        we're making a single decision based on the full context,
        not generating autoregressively).

        Args:
            token_ids: [B, T] — input token IDs (user msg + tool outputs + memory)
            type_ids: [B, T] — segment types for each token

        Returns:
            dict with tool selection, confidence, outcome prediction, etc.
        """
        # Encode
        context = self.encoder(token_ids, type_ids)  # [B, slots, d]

        # Build slot→token_id mapping for copy mechanism
        # Stride-1 slots map directly; compressed slots use first token in group
        slot_token_ids = self._build_slot_token_ids(token_ids)  # [B, slots]

        # Process through resonance blocks (bidirectional for agent)
        hidden = self.resonance(context, causal=False)  # [B, slots, d]

        # Action decision
        action = self.action_decoder(hidden, context, slot_token_ids)

        # Internal THINK loop: if confidence too low, re-process
        # Use torch.no_grad for THINK iterations to avoid 4× memory
        think_steps = 0
        while action['should_think'].any() and think_steps < self.max_think_steps:
            with torch.no_grad():
                hidden = self.resonance(hidden, causal=False)
            action = self.action_decoder(hidden, context, slot_token_ids)
            think_steps += 1

        # Predict outcome (returns dict with 'p_success' and 'logit')
        outcome = self.outcome_predictor(action['hidden'], action['tool_id'])

        return {
            'tool_id': action['tool_id'],
            'tool_logits': action['tool_logits'],
            'confidence': action['confidence'],
            'p_success': outcome['p_success'],
            'outcome_logit': outcome['logit'],
            'hidden': action['hidden'],
            'context': context,
            'slot_token_ids': slot_token_ids,
            'think_steps': think_steps,
        }

    def _build_slot_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Build token ID mapping for each context slot.

        Stride-1 slots: direct 1:1 mapping to original token IDs.
        Stride-4/16 slots: use the FIRST token in each compressed group.

        Returns: [B, total_slots] long tensor
        """
        B, T = token_ids.shape
        elastic = self.encoder.elastic
        slot_ids = []

        prev = 0
        for i in range(elastic.n_levels):
            end = min(elastic.boundaries[i], T)
            if prev >= end:
                break

            chunk_ids = token_ids[:, prev:end]  # [B, chunk_len]
            stride = elastic.strides[i]

            if stride > 1 and chunk_ids.shape[1] >= stride:
                # Take first token from each group of `stride` tokens
                n_groups = chunk_ids.shape[1] // stride
                chunk_ids = chunk_ids[:, :n_groups * stride:stride]  # [B, n_groups]

            slot_ids.append(chunk_ids)
            prev = end

        return torch.cat(slot_ids, dim=1)  # [B, total_slots]

    def generate_args(self, hidden: torch.Tensor,
                      context: torch.Tensor,
                      slot_token_ids: torch.Tensor,
                      max_tokens: int = 256) -> torch.Tensor:
        """Generate text arguments for the selected tool action.
        Use slot_token_ids from forward_agent() output, NOT raw token_ids.
        """
        return self.action_decoder.generate_args(
            hidden, context, slot_token_ids, max_tokens
        )

    def count_parameters(self) -> dict:
        """Count parameters by component."""
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


def build_v5_local(dropout: float = 0.0) -> V5ResonanceModel:
    """Build V5 for local inference/training on RX 7600 (d=1024, 8 blocks)."""
    return V5ResonanceModel(
        d_model=1024, n_blocks=8, n_heads=8,
        window_size=128, max_seq_len=2048,
        vocab_size=100279, K_hash=8, d_hash=128,
        dropout=dropout,
    )


def build_v5_runpod(dropout: float = 0.1) -> V5ResonanceModel:
    """Build V5 for RunPod A100 pre-training (d=1536, 12 blocks)."""
    return V5ResonanceModel(
        d_model=1536, n_blocks=12, n_heads=12,
        window_size=128, max_seq_len=8192,
        vocab_size=100279, K_hash=8, d_hash=192,
        dropout=dropout,
    )
