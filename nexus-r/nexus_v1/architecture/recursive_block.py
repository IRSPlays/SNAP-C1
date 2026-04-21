"""
Nexus-r V1: Recursive Reasoning Block
=======================================
Implements the TRM-style recursive loop with dual-stream attention.

Architecture per recursive pass:
  1. Thought = Thought + Anchor_injection (residual bypass)
  2. Thought → RMSNorm → DualStreamMLA(Q=Thought, K,V=Anchor) → AttnRes
  3. Thought → RMSNorm → SwiGLU FFN → AttnRes
  4. Repeat for L_cycles

The key insight from Samsung TRM:
  - H_cycles-1 passes run with torch.no_grad() (saves massive memory)
  - Only the LAST H_cycle pass has gradients
  - This makes recursive backprop O(1) in memory, not O(K)

The key insight from our spec:
  - K,V are frozen from input (Anchor stream)
  - Q evolves each cycle (Thought stream)
  - Residual bypass feeds directly into Add&Norm before the block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .layers import RMSNorm, SwiGLU, rms_norm
from .dual_stream_mla import DualStreamMLA


class RecursiveBlock(nn.Module):
    """
    One layer of the recursive reasoning stack.
    Contains: DualStreamMLA + SwiGLU + AttnRes hooks.
    Weight-tied: the same block is reused across all recursive passes.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, ffn_expansion: float = 8/3, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mla = DualStreamMLA(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_expansion, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        thought: torch.Tensor,
        anchor_k: torch.Tensor,
        anchor_v: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        One forward pass through the recursive block.

        Args:
            thought:  [B, T, D] current thought state
            anchor_k: [B, H_kv, T, D_h] frozen keys
            anchor_v: [B, H_kv, T, D_h] frozen values
            cos_sin:  RoPE tensors

        Returns:
            thought: [B, T, D] refined thought state
        """
        # MLA: Thought queries Anchor (cross-attention)
        h = self.attn_norm(thought)
        attn_out = self.mla(h, anchor_k, anchor_v, cos_sin)
        thought = thought + self.drop(attn_out)  # Post-norm residual

        # FFN
        h = self.ffn_norm(thought)
        ffn_out = self.ffn(h)
        thought = thought + ffn_out

        return thought


class RecursiveReasoner(nn.Module):
    """
    The full recursive reasoning loop.

    Implements the TRM H-cycle / L-cycle pattern:
    - L_layers: number of RecursiveBlock layers per L-cycle (weight-tied stack)
    - L_cycles: how many times to loop through L_layers per H-step
    - H_cycles: outer reasoning steps (H-1 with no_grad, last 1 with grad)

    Plus Attention Residuals across recursive depths.
    Plus Dynamic Halting via cosine similarity confidence gate.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        L_layers: int = 2,
        L_cycles: int = 4,
        H_cycles: int = 3,
        halt_threshold: float = 0.001,
        max_halt_steps: int = 8,
        step_bias_scale: float = 0.1,
        ffn_expansion: float = 8/3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles
        self.halt_threshold = halt_threshold
        self.max_halt_steps = max_halt_steps
        self.step_bias_scale = step_bias_scale

        # Shared-weight recursive blocks (weight-tied across all cycles)
        self.layers = nn.ModuleList([
            RecursiveBlock(d_model, n_heads, n_kv_heads, ffn_expansion, dropout=dropout)
            for _ in range(L_layers)
        ])

        # Step embeddings: break the fixed-point attractor by giving each
        # H-cycle a unique identity. Without this, weight-tied blocks
        # converge to f(f(f(x))) ~ f(x) because same function + same input.
        self.step_embeds = nn.Embedding(H_cycles, d_model)
        nn.init.normal_(self.step_embeds.weight, std=0.02)

        # R12: Annealed repulsion tau (set by training loop)
        self.register_buffer('_repulsion_tau', torch.tensor(0.50))

    def _apply_step_bias(self, thought: torch.Tensor, h_step: int) -> torch.Tensor:
        step_bias = self.step_embeds.weight[h_step].to(dtype=thought.dtype, device=thought.device)
        return thought + self.step_bias_scale * step_bias.view(1, 1, -1)

    @staticmethod
    def _mean_cosine_similarity(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(
            current.float().flatten(1),
            previous.float().flatten(1),
            dim=-1,
        ).mean()

    def _run_L_cycle(
        self,
        thought: torch.Tensor,
        anchor_k: torch.Tensor,
        anchor_v: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """One full L-cycle: pass through all L_layers. No input injection —
        thought already starts as encoded input, and cross-attention to
        anchor K,V provides ongoing access to input information."""
        for layer in self.layers:
            thought = layer(thought, anchor_k, anchor_v, cos_sin)
        return thought

    def forward(
        self,
        input_embeddings: torch.Tensor,
        anchor_k: torch.Tensor,
        anchor_v: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full recursive reasoning loop.

        Follows the TRM memory pattern: run exploratory H-cycles without
        gradients, then do a single trainable refinement step from the final
        exploratory state. This keeps recursive memory use flat in H.

        Args:
            input_embeddings: [B, T, D] — encoded input (used as injection)
            anchor_k, anchor_v: KV from anchor stream
            cos_sin: RoPE

        Returns:
            thought: [B, T, D] — final stabilized thought vector
            info: dict with recursion diagnostics
        """
        B, T, D = input_embeddings.shape

        # Initialize thought from input embeddings
        thought = input_embeddings

        # Track for halting + auxiliary repulsion loss
        total_steps = 0
        halt_similarities = []
        step_similarities_tensor = []  # keep differentiable sims for aux loss
        prev_thought = None
        halted_on_similarity = False
        max_h_cycles = max(1, min(self.H_cycles, self.max_halt_steps))
        h_cycles_used = 0

        # Exploratory passes without gradient retention.
        for h_step in range(max(0, max_h_cycles - 1)):
            with torch.no_grad():
                thought = self._apply_step_bias(thought, h_step)
                for _ in range(self.L_cycles):
                    thought = self._run_L_cycle(thought, anchor_k, anchor_v, cos_sin)

            total_steps += self.L_cycles
            h_cycles_used = h_step + 1

            if prev_thought is not None:
                sim = self._mean_cosine_similarity(thought, prev_thought)
                sim_value = float(sim.detach())
                halt_similarities.append(sim_value)
                if sim_value > (1.0 - self.halt_threshold):
                    halted_on_similarity = True
                    prev_thought = thought
                    break

            prev_thought = thought

        # Final refinement step keeps the recursive loop trainable without
        # retaining graphs for every exploratory H-cycle.
        grad_step_index = min(h_cycles_used, max_h_cycles - 1)
        thought = thought.detach()
        thought = self._apply_step_bias(thought, grad_step_index)
        for _ in range(self.L_cycles):
            thought = self._run_L_cycle(thought, anchor_k, anchor_v, cos_sin)
        total_steps += self.L_cycles
        h_cycles_used = grad_step_index + 1

        if prev_thought is not None:
            sim = self._mean_cosine_similarity(thought, prev_thought.detach())
            sim_value = float(sim.detach())
            halt_similarities.append(sim_value)
            step_similarities_tensor.append(sim)

        # === Auxiliary Repulsion Loss ===
        # tau=0.5: consecutive thought vectors should share at most 50% direction
        # This is aggressive but self-reducing — as sim drops, penalty vanishes
        repulsion_tau = self._repulsion_tau.to(input_embeddings.device, dtype=thought.dtype)
        if len(step_similarities_tensor) > 0:
            penalties = []
            for sim_t in step_similarities_tensor:
                penalties.append(F.relu(sim_t - repulsion_tau))
            diversity_loss = torch.stack(penalties).sum()
        else:
            diversity_loss = torch.tensor(0.0, device=input_embeddings.device)

        info = {
            'total_recursive_steps': total_steps,
            'h_cycles_used': h_cycles_used,
            'halt_similarities': halt_similarities,
            'converged_early': halted_on_similarity and h_cycles_used < self.H_cycles,
            'diversity_loss': diversity_loss,
        }

        return thought, info
