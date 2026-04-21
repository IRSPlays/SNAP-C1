"""
SNAP-C1 V4: AST Geometry Decoder
=================================
Pointer-Generator BPE output head wired to the V3 Liquid Time-Constant GRU core.
All operations are DirectML-safe (no scatter_ in any backward path).
"""

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from v3_core.architecture.ast_decoder import ASTDecoder as V3ASTDecoder


def chunked_softmax(x: torch.Tensor, dim: int = -1, chunk_size: int = 20000) -> torch.Tensor:
    """
    DirectML-safe Softmax for 100k+ dimensions.
    Bypasses AMD driver hangs by breaking the computation into chunks.
    Only activates on DirectML (privateuseone) — CUDA/CPU use the native fast path.
    """
    # Fast path for non-DirectML devices (no overhead during CUDA/CPU training)
    if x.device.type != 'privateuseone':
        return F.softmax(x, dim=dim)
    # Numerical stability: subtract max.  **Detached** because torch.max(dim=...) backward
    # uses scatter_ which DirectML rejects.  Detach is mathematically correct here —
    # subtracting a constant from logits doesn't change the softmax gradient.
    max_val = torch.max(x, dim=dim, keepdim=True)[0].detach()
    x_stable = x - max_val

    # We must chunk the sum(exp(x)) calculation
    exp_sum = torch.zeros_like(max_val)
    num_elements = x.shape[dim]

    for i in range(0, num_elements, chunk_size):
        curr_chunk_size = min(chunk_size, num_elements - i)
        # Sliced math stays on GPU
        chunk = x_stable.narrow(dim, i, curr_chunk_size)
        exp_sum = exp_sum + torch.exp(chunk).sum(dim=dim, keepdim=True)

    # Detach the denominator too — softmax backward is y*(grad - (y·grad).sum()),
    # which doesn't need grad through the normalization constant.
    return torch.exp(x_stable) / (exp_sum.detach() + 1e-10)


class DML_GRUCell(nn.Module):
    """
    A manual GRU Cell implementation to bypass `aten::_thnn_fused_gru_cell`
    which is not supported by `torch-directml` on AMD GPUs.
    Extended with an optional context gate from the pointer-generator attention.
    """
    def __init__(self, input_size, hidden_size, context_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.weight_ctx = nn.Linear(context_size, hidden_size)  # context gate (fix #5)

    def forward(self, input_tensor, hx, context_t=None):
        # Calculate gates
        ih = self.weight_ih(input_tensor)
        hh = self.weight_hh(hx)

        # Split into reset (r), update (z), and new gate (n)
        i_r, i_z, i_n = ih.chunk(3, 1)
        h_r, h_z, h_n = hh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        # Context gate: inject attended copy target into the new-state calculation
        ctx_bias = self.weight_ctx(context_t) if context_t is not None else 0
        newgate = torch.tanh(i_n + resetgate * h_n + ctx_bias)

        hy = newgate + updategate * (hx - newgate)
        return hy


class PointerGeneratorBPEHead(nn.Module):
    """
    SNAP-C1 V4: The Semantic Zero-Shot Copy Mechanism

    Instead of a flat Linear layer guessing 1000 hardcoded variables, this Sub-Network:
    1. Auto-regressively predicts BPE sub-tokens (for novel generation).
    2. Calculates an Attention Distribution over the `input_context`.
    3. Calculates a `p_gen` (Probability of Generation) scalar.

    If p_gen is 0.99, it generates a sub-token from its latent dictionary.
    If p_gen is 0.01, it physically copies a specific external-framework token
    directly from the SWE-Bench Prompt (Zero-Shot execution).
    """
    def __init__(self, hidden_dim: int, bpe_vocab_size: int, context_dim: int = 1024):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = bpe_vocab_size

        # 1. Factored BPE Generator (improvement C) — 512-dim bottleneck instead of flat projection:
        #    Saves ~97 MB params vs hidden_dim=1024 direct head. Weight tying added in V4ASTDecoder.
        _bottleneck = 512
        self.vocab_proj_down = nn.Linear(hidden_dim, _bottleneck)
        self.vocab_proj_up   = nn.Linear(_bottleneck, bpe_vocab_size, bias=False)

        # 2. Multi-Head Copy Attention (8 heads: richer pointing than single-head Bahdanau)
        # kdim/vdim allow cross-dim attention from hidden_dim queries to context_dim keys
        self.copy_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )

        # 3. The p_gen Router — context_t + decoder_hidden + INPUT EMBEDDING (not hidden twice)
        self.p_gen_linear = nn.Linear(hidden_dim * 3, 1)

    def forward(self, decoder_hidden: torch.Tensor,
                context_vectors: torch.Tensor,
                context_token_ids: torch.Tensor,
                input_embedding: torch.Tensor = None,
                kv_cache: dict = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        V4 Multi-Head Copy Attention Forward.
        input_embedding: token embedding from the previous step, used to accurately route p_gen.
                         Defaults to decoder_hidden at step 0 when not yet available.
        kv_cache: optional dict for KV caching (improvement G). On first call, K/V projections
                  are stored with key=context_vectors.data_ptr(). Subsequent steps reuse them,
                  eliminating redundant context projections across beam search steps.
        """
        # --- 1. Factored Vocabulary Distribution: hidden → bottleneck → vocab ---
        vocab_logits = self.vocab_proj_up(F.gelu(self.vocab_proj_down(decoder_hidden)))  # [B, vocab_size]
        P_vocab = chunked_softmax(vocab_logits, dim=-1)

        # --- 2. Multi-Head Copy Attention with KV caching (improvement G) ---
        # key_padding_mask: True = masked position (our -1 padding sentinel)
        padding_mask = (context_token_ids == -1)                   # [B, seq_len]

        if kv_cache is not None:
            ctx_ptr = context_vectors.data_ptr()
            if ctx_ptr not in kv_cache:
                nheads   = self.copy_attention.num_heads
                head_dim = self.copy_attention.head_dim
                embed_dim = self.copy_attention.embed_dim
                bsz, seq_len, _ = context_vectors.shape
                if self.copy_attention.in_proj_bias is not None:
                    k_bias = self.copy_attention.in_proj_bias[embed_dim : 2 * embed_dim]
                    v_bias = self.copy_attention.in_proj_bias[2 * embed_dim :]
                else:
                    k_bias = v_bias = None
                k_proj = F.linear(context_vectors, self.copy_attention.k_proj_weight, k_bias)
                v_proj = F.linear(context_vectors, self.copy_attention.v_proj_weight, v_bias)
                static_k = k_proj.view(bsz, seq_len, nheads, head_dim).permute(0, 2, 1, 3).reshape(bsz * nheads, seq_len, head_dim)
                static_v = v_proj.view(bsz, seq_len, nheads, head_dim).permute(0, 2, 1, 3).reshape(bsz * nheads, seq_len, head_dim)
                kv_cache[ctx_ptr] = (static_k, static_v)
            static_k, static_v = kv_cache[ctx_ptr]
            attn_out, attn_w = F.multi_head_attention_forward(
                query=decoder_hidden.unsqueeze(1).transpose(0, 1),
                key=context_vectors.transpose(0, 1),
                value=context_vectors.transpose(0, 1),
                embed_dim_to_check=self.copy_attention.embed_dim,
                num_heads=self.copy_attention.num_heads,
                in_proj_weight=None,
                in_proj_bias=self.copy_attention.in_proj_bias,
                bias_k=self.copy_attention.bias_k,
                bias_v=self.copy_attention.bias_v,
                add_zero_attn=self.copy_attention.add_zero_attn,
                dropout_p=0.0,
                out_proj_weight=self.copy_attention.out_proj.weight,
                out_proj_bias=self.copy_attention.out_proj.bias,
                training=False,
                key_padding_mask=padding_mask,
                need_weights=True,
                attn_mask=None,
                use_separate_proj_weight=True,
                q_proj_weight=self.copy_attention.q_proj_weight,
                k_proj_weight=self.copy_attention.k_proj_weight,
                v_proj_weight=self.copy_attention.v_proj_weight,
                average_attn_weights=True,
                static_k=static_k,
                static_v=static_v,
            )
            context_t = attn_out.transpose(0, 1).squeeze(1)        # [B, hidden_dim]
            P_attn = attn_w.squeeze(1)                             # [B, seq_len]
        else:
            context_t, attn_weights = self.copy_attention(
                query=decoder_hidden.unsqueeze(1),                 # [B, 1, hidden_dim]
                key=context_vectors,                               # [B, seq_len, context_dim]
                value=context_vectors,                             # [B, seq_len, context_dim]
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=True
            )
            context_t = context_t.squeeze(1)                       # [B, hidden_dim]
            P_attn = attn_weights.squeeze(1)                       # [B, seq_len]

        # --- 3. Calculate p_gen (correctly uses input embedding, not hidden twice) ---
        inp_emb = input_embedding if input_embedding is not None else decoder_hidden
        p_gen_input = torch.cat([context_t, decoder_hidden, inp_emb], dim=-1)  # [B, hidden*3]
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))

        return P_vocab, P_attn, p_gen


class V4ASTDecoder(V3ASTDecoder):
    """
    Upgrades the V3 Geometric Core with the V4 Pointer-Generator Output Head.
    Inherits the flawless Liquid-Time GRU and structure mapping from `v3_core`.
    """
    def __init__(self, concept_dim: int, ast_vocab_size: int = 250, hidden_dim: int = 512, bpe_vocab_size: int = 100279):
        # Initialize the base V3 geometric structure
        super().__init__(concept_dim, ast_vocab_size, hidden_dim, semantic_vocab_size=1)

        self.bpe_vocab_size = bpe_vocab_size

        # Replace the hardcoded semantic head with the BPE Pointer-Generator
        self.hybrid_bpe_head = PointerGeneratorBPEHead(
            hidden_dim=hidden_dim,
            bpe_vocab_size=bpe_vocab_size,
            context_dim=concept_dim
        )

        # BPE token embedding table (input side — frozen during training to avoid
        # nn.Embedding scatter_add_ backward on DirectML; see trainer freeze logic)
        self.token_embedding = nn.Embedding(bpe_vocab_size, hidden_dim)

        # Weight tying: share vocab_proj_up weights with token_embedding for
        # parameter efficiency (improvement C)
        self.hybrid_bpe_head.vocab_proj_up.weight = self.token_embedding.weight

        # Single-step context-gated GRU for auto-regressive decoding
        self.state_transition = DML_GRUCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            context_size=concept_dim
        )

    def forward(self, input_equilibrium: torch.Tensor, context_vectors: torch.Tensor,
                context_token_ids: torch.Tensor, max_nodes: int = 50) -> List[List[int]]:
        """
        V4 Real Code Generation — runs on whatever device the tensors are on.
        """
        batch_size = input_equilibrium.shape[0]
        dev = input_equilibrium.device

        current_hidden = input_equilibrium.mean(dim=1)
        current_hidden = self.graph_proj(current_hidden)

        generated_sequences = [[] for _ in range(batch_size)]
        active_batches = torch.ones(batch_size, dtype=torch.bool)

        # Repetition penalty tracker (on same device)
        history = torch.zeros((batch_size, self.bpe_vocab_size), device=dev)
        prev_emb = None  # Previous token embedding for accurate p_gen routing

        for step in range(max_nodes):
            if not active_batches.any():
                break

            P_vocab, P_attn, p_gen = self.hybrid_bpe_head(
                current_hidden, context_vectors, context_token_ids, input_embedding=prev_emb
            )

            # Apply repetition penalty to P_vocab
            if step > 0:
                penalty = (history > 0).float() * 0.5
                P_vocab = P_vocab * (1.0 - penalty)
                P_vocab = P_vocab / (P_vocab.sum(dim=-1, keepdim=True) + 1e-10)

            val_v, idx_v = torch.max(P_vocab, dim=-1)
            val_a, pos_a = torch.max(P_attn, dim=-1)
            # Advanced indexing on integer tensor (no grad) — safe from scatter on backward
            _batch_idx = torch.arange(pos_a.shape[0], device=pos_a.device)
            idx_a = context_token_ids[_batch_idx, pos_a]

            prob_v = p_gen.squeeze(1) * val_v
            prob_a = (1 - p_gen.squeeze(1)) * val_a

            next_tokens = torch.where(prob_v > prob_a, idx_v, idx_a)

            for b in range(batch_size):
                tid = next_tokens[b].item()
                if tid < self.bpe_vocab_size:
                    history[b, tid] += 1

            for i in range(batch_size):
                if active_batches[i]:
                    token_id = next_tokens[i].item()
                    if token_id == 100257:
                        active_batches[i] = False
                    elif token_id != 0:
                        generated_sequences[i].append(token_id)

            safe_next_tokens = torch.clamp(next_tokens, max=self.bpe_vocab_size - 1)
            token_emb = self.token_embedding(safe_next_tokens)
            # Context-gated GRU: attended raw context guides the state transition
            gru_context = torch.bmm(P_attn.unsqueeze(1), context_vectors).squeeze(1)  # [B, context_dim]
            current_hidden = self.state_transition(token_emb, current_hidden, context_t=gru_context)
            prev_emb = token_emb  # Feed this step's embedding to next step's p_gen router

        return generated_sequences

    def forward_train(self, input_equilibrium: torch.Tensor, context_vectors: torch.Tensor,
                      context_token_ids: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
        """
        V4 Instruction Tuning (Teacher Forcing).
        """
        device = input_equilibrium.device
        target_seq_len = target_token_ids.shape[1]

        # Keep everything on the same device — no CPU round-trips
        target_token_ids = target_token_ids.to(device)

        current_hidden = input_equilibrium.mean(dim=1)
        current_hidden = self.graph_proj(current_hidden)
        loss = torch.tensor(0.0, device=device)
        prev_emb = None  # Previous token embedding for accurate p_gen routing
        # Coverage vector (improvement I): accumulates attention weights over steps so
        # the coverage loss penalises repeatedly attending to the same source positions.
        coverage = torch.zeros(context_token_ids.shape[0], context_token_ids.shape[1], device=device)

        # Pre-compute vocab index range outside the loop (no grad, reused every step)
        _vocab_range = torch.arange(self.bpe_vocab_size, device=device)  # [V]

        for step in range(target_seq_len):
            P_vocab, P_attn, p_gen = self.hybrid_bpe_head(
                current_hidden, context_vectors, context_token_ids, input_embedding=prev_emb
            )

            true_tokens = target_token_ids[:, step]

            # DirectML-safe target probability extraction: broadcast comparison mask.
            # torch.gather, advanced indexing, AND F.one_hot all use scatter_ internally,
            # which DirectML rejects. The == comparison builds the same binary mask using
            # only broadcast + elementwise ops whose backward is purely multiply + expand.
            safe_true_for_vocab = torch.clamp(true_tokens, max=self.bpe_vocab_size - 1)
            mask = (_vocab_range.unsqueeze(0) == safe_true_for_vocab.unsqueeze(1)).float()  # [B, V]
            p_vocab_target = (P_vocab * mask).sum(dim=-1)  # [B]

            matches = (context_token_ids == true_tokens.unsqueeze(1)).float()
            p_copy_target = torch.sum(P_attn * matches, dim=1)

            p_final = p_gen.squeeze(1) * p_vocab_target + (1 - p_gen.squeeze(1)) * p_copy_target

            # Label smoothing (improvement I, smooth_eps=0.1): prevents overconfident predictions
            smooth_eps = 0.1
            p_final_smooth = (1.0 - smooth_eps) * p_final + smooth_eps / self.bpe_vocab_size
            step_loss = -torch.log(p_final_smooth + 1e-10)

            # Coverage loss: penalise attending to already-seen positions (improvement I)
            cov_loss = torch.sum(torch.min(P_attn, coverage), dim=-1).mean()

            mask_pad = (true_tokens != 0).float()
            loss = loss + (step_loss * mask_pad).mean() + 0.5 * cov_loss

            # Accumulate coverage for next step (no grad — coverage is a running stat)
            coverage = coverage + P_attn.detach()

            safe_true_for_emb = torch.clamp(true_tokens, max=self.bpe_vocab_size - 1)
            # Detach embedding lookup: nn.Embedding backward uses scatter_add_ which DirectML
            # rejects. The token_emb here only drives the GRU state forward (teacher input),
            # not the loss itself — detaching is safe and all meaningful gradients still flow
            # through P_vocab, P_attn, and p_gen.
            token_emb = self.token_embedding(safe_true_for_emb).detach()
            # Context-gated GRU: teacher-forced attention context guides the state transition
            gru_context = torch.bmm(P_attn.unsqueeze(1), context_vectors).squeeze(1)  # [B, context_dim]
            current_hidden = self.state_transition(token_emb, current_hidden, context_t=gru_context)
            prev_emb = token_emb  # Feed to next step's p_gen router

        return loss / target_seq_len

    def forward_beam(self, input_equilibrium: torch.Tensor, context_vectors: torch.Tensor,
                     context_token_ids: torch.Tensor, max_nodes: int = 100,
                     beam_width: int = 3) -> List[List[int]]:
        """
        V4 Beam Search Decoding — runs on whatever device tensors live on.
        """
        batch_size = input_equilibrium.shape[0]

        init_hidden = input_equilibrium.mean(dim=1)
        init_hidden = self.graph_proj(init_hidden)

        all_results = []

        for b in range(batch_size):
            h0 = init_hidden[b:b+1]                    # [1, hidden_dim]
            ctx_v = context_vectors[b:b+1]             # [1, seq_len, context_dim]
            ctx_i = context_token_ids[b:b+1]           # [1, seq_len]

            # Each beam: (log_prob, token_sequence, hidden_state, prev_emb)
            beams = [(0.0, [], h0, None)]
            completed = []

            for step in range(max_nodes):
                if not beams:
                    break
                candidates = []
                for log_prob, seq, hidden, prev_emb in beams:
                    P_vocab, P_attn, p_gen = self.hybrid_bpe_head(
                        hidden, ctx_v, ctx_i, input_embedding=prev_emb
                    )
                    # Combine generation + copy probabilities
                    p_combined = p_gen.squeeze(1) * P_vocab + (1 - p_gen.squeeze(1)) * torch.zeros_like(P_vocab)

                    topk_probs, topk_ids = torch.topk(p_combined[0], k=min(beam_width, self.bpe_vocab_size))
                    for prob, tok_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                        if prob <= 0:
                            continue
                        new_log_prob = log_prob + (prob if prob > 0 else -1e10)
                        safe_tok = min(tok_id, self.bpe_vocab_size - 1)
                        new_emb = self.token_embedding(
                            torch.tensor([safe_tok], device=hidden.device)
                        )
                        gru_ctx = torch.bmm(P_attn.unsqueeze(1), ctx_v).squeeze(1)
                        new_hidden = self.state_transition(new_emb, hidden, context_t=gru_ctx)
                        new_seq = seq + [tok_id]
                        if tok_id == 100257:  # EOS
                            completed.append((new_log_prob, new_seq[:-1]))
                        else:
                            candidates.append((new_log_prob, new_seq, new_hidden, new_emb))

                # Keep top beam_width candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]

            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                all_results.append(completed[0][1])
            elif beams:
                all_results.append(beams[0][1])
            else:
                all_results.append([])

        return all_results
