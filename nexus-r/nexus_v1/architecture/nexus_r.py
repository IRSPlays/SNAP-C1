"""
Nexus-R V1: Full Model Assembly
=================================
The "First Heartbeat" — Asymmetric Recursive Architecture.

Architecture flow:
  Input IDs → Embedding → RoPE → Anchor Encoder (K,V freeze)
                                ↘ Engram (thought init)
                                ↘ RecursiveReasoner(Q evolves, K,V frozen)
                                → Final Norm → LM Head

Model sizes:
  - build_nexus_tiny():  ~7M params  (for smoke tests)
  - build_nexus_small(): ~30M params (for real training on RX 7600)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .layers import RMSNorm, CastedLinear, RotaryEmbedding, Attention, rms_norm
from .dual_stream_mla import DualStreamMLA
from .recursive_block import RecursiveReasoner


@dataclass
class NexusConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4       # GQA: fewer KV heads
    n_anchor_layers: int = 2  # Layers to encode anchor K,V
    L_layers: int = 2         # Recursive block depth
    L_cycles: int = 4         # Inner loop repeats
    H_cycles: int = 3         # Outer reasoning steps
    ffn_expansion: float = 8/3
    max_seq_len: int = 512
    halt_threshold: float = 0.001
    dropout: float = 0.0      # Off for small models


class AnchorEncoder(nn.Module):
    """
    Encodes input embeddings into frozen Anchor K,V tensors.
    Uses standard self-attention layers (NOT recursive).
    These K,V become the "ground truth" that the thought stream queries.
    """
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        self.drop = nn.Dropout(cfg.dropout)
        for _ in range(cfg.n_anchor_layers):
            self.layers.append(nn.ModuleDict({
                'attn_norm': RMSNorm(cfg.d_model),
                'attn': Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, dropout=cfg.dropout),
                'ffn_norm': RMSNorm(cfg.d_model),
                'ffn': nn.Sequential(
                    CastedLinear(cfg.d_model, int(cfg.d_model * cfg.ffn_expansion)),
                    nn.SiLU(),
                    CastedLinear(int(cfg.d_model * cfg.ffn_expansion), cfg.d_model),
                ),
            }))

        # Project to K,V for the dual-stream MLA
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.k_proj = CastedLinear(cfg.d_model, cfg.n_kv_heads * self.head_dim)
        self.v_proj = CastedLinear(cfg.d_model, cfg.n_kv_heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] input embeddings
            cos_sin: RoPE

        Returns:
            encoded: [B, T, D] anchor-encoded hidden state
            anchor_k: [B, n_kv_heads, T, head_dim]
            anchor_v: [B, n_kv_heads, T, head_dim]
        """
        h = x
        for layer in self.layers:
            # Self-attention (CAUSAL — must not see future tokens)
            normed = layer['attn_norm'](h)
            h = h + layer['attn'](normed, cos_sin, is_causal=True)
            # FFN
            normed = layer['ffn_norm'](h)
            h = h + self.drop(layer['ffn'](normed))

        B, T, D = h.shape
        # Project to K, V
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to keys only (values stay position-free)
        cos, sin = cos_sin
        # RoPE: k shape is [B, H, T, D_h]
        k = _apply_rope_to_kv(k, cos, sin)

        return h, k, v


def _apply_rope_to_kv(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to key tensor [B, H, T, D_h]."""
    # cos, sin are [T, D_h] or [1, 1, T, D_h]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D_h]
        sin = sin.unsqueeze(0).unsqueeze(0)
    k_rot = k[..., : cos.shape[-1]]
    k_pass = k[..., cos.shape[-1]:]
    # Rotate
    k1 = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat([k1, k_pass], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class NexusR(nn.Module):
    """
    Nexus-R V1: Asymmetric Recursive Language Model.

    The "First Heartbeat" architecture:
    - Embedding → AnchorEncoder → RecursiveReasoner → LM Head
    - Anchor stream freezes K,V from input
    - Thought stream (Q) evolves recursively
    - Dynamic halting via cosine similarity
    - Memory-efficient via TRM no_grad pattern
    """
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Positional encoding (RoPE)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.max_seq_len)

        # Anchor encoder (produces frozen K,V)
        self.anchor_encoder = AnchorEncoder(cfg)

        # Recursive reasoning core
        self.reasoner = RecursiveReasoner(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            L_layers=cfg.L_layers,
            L_cycles=cfg.L_cycles,
            H_cycles=cfg.H_cycles,
            halt_threshold=cfg.halt_threshold,
            ffn_expansion=cfg.ffn_expansion,
            dropout=cfg.dropout,
        )

        # Final norm + LM head
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = CastedLinear(cfg.d_model, cfg.vocab_size)

        # Weight tying (embedding ↔ LM head)
        self.lm_head.weight = self.embed.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Only init embedding — CastedLinear already has correct 1/sqrt(fan_in) init."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: [B, T] token IDs
            labels: [B, T] target IDs for cross-entropy (shifted internally)

        Returns:
            dict with 'logits', 'loss' (if labels), 'recursion_info'
        """
        B, T = input_ids.shape

        # Embed
        x = self.embed(input_ids)  # [B, T, D]

        # Embedding-space noise: annealed by training loop (Fix 2)
        if self.training:
            noise_scale = getattr(self, '_current_noise_scale', 0.1)
            x = x + torch.randn_like(x) * noise_scale

        # RoPE
        cos_sin = self.rope(T)

        # Anchor encode (produce K,V for the recursive loop)
        # K,V are computed once and reused across H-cycles, but gradients
        # MUST flow through k_proj/v_proj for the model to learn attention.
        encoded, anchor_k, anchor_v = self.anchor_encoder(x, cos_sin)

        # Recursive reasoning
        thought, recursion_info = self.reasoner(
            encoded, anchor_k, anchor_v, cos_sin
        )

        # Project to vocab
        thought = self.final_norm(thought)
        logits = self.lm_head(thought)  # [B, T, vocab_size]

        result = {
            'logits': logits,
            'recursion_info': recursion_info,
        }

        # Loss (causal LM: predict next token)
        # NOTE: labels are ALREADY shifted by the dataset (input=chunk[:-1], target=chunk[1:])
        # Do NOT shift again here -- just compute cross-entropy directly.
        if labels is not None:
            # R12b: Final-step-only CE (best config: eval=0.639)
            label_smoothing = getattr(self, '_current_label_smoothing', 0.02)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

            # R12b: Progression loss — encourage consecutive intermediates to differ
            # Each H-step should produce a genuinely different thought vector
            intermediates = recursion_info.get('intermediates', [])
            prog_loss = torch.tensor(0.0, device=logits.device)
            if len(intermediates) > 1:
                for j in range(1, len(intermediates)):
                    cos_dist = 1.0 - F.cosine_similarity(
                        intermediates[j].detach().flatten(1),
                        intermediates[j-1].detach().flatten(1),
                        dim=-1
                    ).mean()
                    prog_loss = prog_loss + cos_dist
                prog_loss = prog_loss / (len(intermediates) - 1)

            aux_loss = recursion_info.get('diversity_loss', 0.0)
            aux_coeff = 0.1  # R12b config
            loss = ce_loss + aux_coeff * aux_loss + 0.1 * prog_loss
            result['loss'] = loss
            result['ce_loss'] = ce_loss
            result['aux_loss'] = aux_loss
            result['prog_loss'] = prog_loss

        return result

    def count_params(self) -> dict:
        """Count parameters by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        total = sum(p.numel() for p in self.parameters())
        # Weight-tied head doesn't add new params
        unique = total - self.lm_head.weight.numel()

        return {
            'total': total,
            'unique (excl tied head)': unique,
            'embed': self.embed.weight.numel(),
            'anchor_encoder': _count(self.anchor_encoder),
            'reasoner': _count(self.reasoner),
            'final_norm': _count(self.final_norm),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        """Autoregressive generation with repetition penalty."""
        generated_counts = {}
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]
            out = self.forward(idx_cond)
            logits = out['logits'][:, -1, :]  # Last position

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id, count in generated_counts.items():
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Temperature + top-k sampling
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tid = next_token.item()
            generated_counts[tid] = generated_counts.get(tid, 0) + 1
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================================
# Builder functions
# ============================================================

def build_nexus_tiny() -> NexusR:
    """~7M params — smoke test config."""
    cfg = NexusConfig(
        vocab_size=8192,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,
        n_anchor_layers=2,
        L_layers=2,
        L_cycles=2,
        H_cycles=2,
        max_seq_len=256,
    )
    return NexusR(cfg)


def build_nexus_small() -> NexusR:
    """~30M params — real training config."""
    cfg = NexusConfig(
        vocab_size=32000,
        d_model=512,
        n_heads=16,
        n_kv_heads=8,
        n_anchor_layers=3,
        L_layers=3,
        L_cycles=4,
        H_cycles=3,
        max_seq_len=512,
    )
    return NexusR(cfg)
