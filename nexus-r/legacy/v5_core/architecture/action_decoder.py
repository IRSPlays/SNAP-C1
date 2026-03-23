"""
SNAP-C1 V5: Action Decoder (Component 5)
==========================================
Structured action output — NOT text completion.

Unlike GPT-4/Claude which generate arbitrary text and hope a framework
parses tool calls from it, the V5 Action Decoder outputs structured
decisions directly from neural network heads:

  Head 1: tool_head     — which tool to use (softmax over tool IDs)
  Head 2: confidence    — P(ready to act) — if low, loops back to THINK
  Head 3: arg_generator — pointer-generator for text arguments (EDIT, RESPOND)

The argument generator reuses the proven pointer-generator mechanism from V4:
copy attention over context (for exact variable names, file paths) combined
with vocabulary generation (for novel text).

DirectML safety:
  - nn.Linear (all heads) ✓
  - F.softmax (small dim, 8-way) ✓
  - chunked_softmax (vocab projection) ✓
  - MultiheadAttention (copy attention) ✓
  - StableSigmoid (confidence) ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_core.utils.dml_ops import (
    RMSNorm, stable_sigmoid, chunked_softmax, DML_GRUCell
)


# Tool IDs — registered tools the model can call
class ToolID:
    SEARCH = 0      # Search files/code
    READ = 1        # Read file contents
    EDIT = 2        # Edit a file
    RUN = 3         # Run a terminal command
    THINK = 4       # Internal reasoning (loop back)
    RESPOND = 5     # Generate response to user
    RECALL = 6      # Query Fast Brain (episodic memory)
    INTROSPECT = 7  # Read/modify own code via Code Introspector

    COUNT = 8
    NAMES = ["SEARCH", "READ", "EDIT", "RUN", "THINK", "RESPOND", "RECALL", "INTROSPECT"]


class PointerGeneratorHead(nn.Module):
    """
    V5 Pointer-Generator for argument text generation.
    Generates text by combining:
      1. Vocabulary distribution (novel generation)
      2. Copy distribution over context (exact reproduction of names, paths, etc.)
      3. p_gen gate (how much to generate vs. copy)

    Operates autoregressively: one token at a time with GRU hidden state.
    """

    def __init__(self, d_model: int = 1024, vocab_size: int = 100279,
                 max_arg_tokens: int = 512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_arg_tokens = max_arg_tokens

        # Factored vocabulary projection (bottleneck scales with d_model)
        self._bottleneck = max(512, d_model // 2)
        self.vocab_down = nn.Linear(d_model, self._bottleneck)
        self.vocab_up = nn.Linear(self._bottleneck, vocab_size, bias=False)

        # Copy attention over context (8-head)
        self.copy_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8,
            batch_first=True
        )

        # p_gen gate: decoder_hidden + context + input_embed → scalar
        self.p_gen_proj = nn.Linear(3 * d_model, 1)

        # Autoregressive decoder cell (DirectML-safe GRU)
        self.gru = DML_GRUCell(d_model, d_model, context_size=d_model)

        # Input projection for previously generated token
        self.input_proj = nn.Linear(d_model, d_model)

    def _build_copy_dist(self, P_copy, context_token_ids, p_gen, device):
        """Build copy distribution efficiently.
        Instead of O(slots × vocab) per step, uses scatter-free vectorized approach:
        groups attention weights by token ID using unique() to skip empty vocab entries.
        """
        B = P_copy.shape[0]
        copy_dist = torch.zeros(B, self.vocab_size, device=device, dtype=P_copy.dtype)

        for b in range(B):
            unique_ids = context_token_ids[b].unique()
            for uid in unique_ids:
                if uid < 0:
                    continue  # skip padding
                positions = (context_token_ids[b] == uid).float()  # [slots]
                copy_dist[b, uid] = (P_copy[b] * positions).sum()

        return (1 - p_gen) * copy_dist  # [B, vocab]

    def forward(self, hidden_state: torch.Tensor,
                context: torch.Tensor,
                context_token_ids: torch.Tensor,
                max_tokens: int = None) -> torch.Tensor:
        """
        Generate argument tokens autoregressively (inference mode).

        Args:
            hidden_state: [B, d_model] — pooled resonance block output
            context: [B, context_slots, d_model] — elastic context tensors
            context_token_ids: [B, context_slots] — token IDs in context (for copy)
            max_tokens: max tokens to generate (default: self.max_arg_tokens)

        Returns:
            [B, max_tokens] — generated token IDs
        """
        B = hidden_state.shape[0]
        device = hidden_state.device
        max_t = max_tokens or self.max_arg_tokens

        # Initialize GRU hidden state from pooled resonance output
        h = hidden_state  # [B, d_model]
        prev_embed = self.input_proj(hidden_state)  # [B, d_model]
        output_ids = []

        for step in range(max_t):
            # Copy attention: query with decoder state, keys/values from context
            context_t, attn_weights = self.copy_attn(
                query=h.unsqueeze(1), key=context, value=context,
                need_weights=True, average_attn_weights=True
            )
            context_t = context_t.squeeze(1)      # [B, d]
            P_copy = attn_weights.squeeze(1)       # [B, slots]

            # Vocabulary distribution
            vocab_logits = self.vocab_up(F.gelu(self.vocab_down(h)))  # [B, vocab]
            P_vocab = chunked_softmax(vocab_logits, dim=-1)            # [B, vocab]

            # p_gen: probability of generating from vocab vs copying from context
            p_gen_input = torch.cat([h, context_t, prev_embed], dim=-1)  # [B, 3d]
            p_gen = stable_sigmoid(self.p_gen_proj(p_gen_input))  # [B, 1]

            # Combined distribution
            final_dist = p_gen * P_vocab
            if context_token_ids is not None:
                final_dist = final_dist + self._build_copy_dist(
                    P_copy, context_token_ids, p_gen, device
                )

            # Greedy decode (argmax)
            token_id = final_dist.argmax(dim=-1)  # [B]
            output_ids.append(token_id)

            # Update GRU state
            h = self.gru(context_t, h, context_t)
            prev_embed = context_t

        return torch.stack(output_ids, dim=1)  # [B, max_tokens]

    def forward_train(self, hidden_state: torch.Tensor,
                      context: torch.Tensor,
                      context_token_ids: torch.Tensor,
                      target_ids: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced training for argument generation.
        Uses ground-truth previous tokens instead of argmax.
        Returns per-step cross-entropy loss (differentiable).

        Args:
            hidden_state: [B, d_model]
            context: [B, context_slots, d_model]
            context_token_ids: [B, context_slots]
            target_ids: [B, max_tokens] — ground truth token IDs

        Returns:
            scalar loss (mean cross-entropy over steps)
        """
        B = hidden_state.shape[0]
        device = hidden_state.device
        max_t = target_ids.shape[1]

        h = hidden_state
        prev_embed = self.input_proj(hidden_state)
        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        for step in range(max_t):
            context_t, attn_weights = self.copy_attn(
                query=h.unsqueeze(1), key=context, value=context,
                need_weights=True, average_attn_weights=True
            )
            context_t = context_t.squeeze(1)
            P_copy = attn_weights.squeeze(1)

            # Vocab logits (raw, not softmaxed — use log_softmax for loss)
            vocab_logits = self.vocab_up(F.gelu(self.vocab_down(h)))  # [B, vocab]

            # p_gen gate
            p_gen_input = torch.cat([h, context_t, prev_embed], dim=-1)
            p_gen = stable_sigmoid(self.p_gen_proj(p_gen_input))  # [B, 1]

            # Build combined log distribution for loss
            P_vocab = chunked_softmax(vocab_logits, dim=-1)  # [B, vocab]
            final_dist = p_gen * P_vocab
            if context_token_ids is not None:
                final_dist = final_dist + self._build_copy_dist(
                    P_copy, context_token_ids, p_gen, device
                )

            # Cross-entropy loss against target
            target = target_ids[:, step]  # [B]
            # Clamp to avoid log(0)
            log_probs = torch.log(final_dist + 1e-10)  # [B, vocab]
            step_loss = F.nll_loss(log_probs, target, ignore_index=-100)

            if target.ne(-100).any():
                total_loss = total_loss + step_loss
                n_valid += 1

            # Teacher forcing: use ground truth for next step
            h = self.gru(context_t, h, context_t)
            prev_embed = context_t

        return total_loss / max(n_valid, 1)


class ActionDecoder(nn.Module):
    """
    V5 Structured Action Decoder.

    Given the resonance block output, produces:
      1. Which tool to use (8-way classification)
      2. Confidence level (should I act or think more?)
      3. Text arguments if needed (via pointer-generator)

    Args:
        d_model: Model dimension
        n_tools: Number of available tools
        vocab_size: BPE vocabulary size for argument generation
        confidence_threshold: Below this, model loops back to THINK
    """

    def __init__(self, d_model: int = 1024, n_tools: int = ToolID.COUNT,
                 vocab_size: int = 100279, confidence_threshold: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_tools = n_tools
        self.confidence_threshold = confidence_threshold

        # Pooling projection: sequence → single vector
        self.pool_proj = nn.Linear(d_model, d_model)
        self.pool_norm = RMSNorm(d_model)

        # Head 1: Tool selection
        self.tool_head = nn.Linear(d_model, n_tools)

        # Head 2: Confidence
        self.confidence_head = nn.Linear(d_model, 1)

        # Head 3: Argument generator (for EDIT, RESPOND, SEARCH, RUN, etc.)
        self.arg_generator = PointerGeneratorHead(
            d_model=d_model, vocab_size=vocab_size
        )

    def _pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Mean pool across sequence dimension, then project.
        x: [B, T, d] → [B, d]
        """
        pooled = x.mean(dim=1)  # [B, d]
        return self.pool_norm(self.pool_proj(pooled))

    def forward(self, resonance_output: torch.Tensor,
                context: torch.Tensor = None,
                context_token_ids: torch.Tensor = None):
        """
        Args:
            resonance_output: [B, T, d_model] — output from ResonanceStack
            context: [B, context_slots, d_model] — elastic context (for copy attention)
            context_token_ids: [B, context_slots] — token IDs in context

        Returns:
            dict with keys:
                'tool_logits': [B, n_tools] — raw logits for tool selection
                'tool_id': [B] — argmax tool selection
                'confidence': [B] — P(ready to act)
                'should_think': [B] — bool, True if confidence < threshold
                'hidden': [B, d_model] — pooled hidden state (for arg generation)
        """
        # Pool sequence to single vector
        h = self._pool_sequence(resonance_output)  # [B, d]

        # Tool selection
        tool_logits = self.tool_head(h)  # [B, n_tools]
        tool_id = tool_logits.argmax(dim=-1)  # [B]

        # Confidence
        confidence = stable_sigmoid(self.confidence_head(h)).squeeze(-1)  # [B]
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
        """Generate text arguments for the selected tool.

        Args:
            hidden: [B, d_model] — pooled hidden state from forward()
            context, context_token_ids: for copy attention

        Returns:
            [B, max_tokens] — generated token IDs
        """
        return self.arg_generator(hidden, context, context_token_ids,
                                  max_tokens=max_tokens)
