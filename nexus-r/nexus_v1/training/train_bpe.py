"""Nexus-R V1 Training Script -- BPE Edition (V14 - Round 15)
============================================================================
Uses tiktoken GPT-2 BPE with RESTRICTED VOCAB (only tokens seen in data).

Round 15: Restore R12b config (the actual peak: eval=0.639)
  - EMA decay=0.999
  - Noise annealing: sigma 0.15->0.03 cosine
  - Label smoothing annealing: 0.10->0.02 cosine
  - Tau annealing: 0.50->0.20 cosine
  - Knowledge oversampling (1.4x)
  - Final-step-only CE + progression loss
  - aux_coeff=0.1
  - min_count=1, H=5, d=256, dropout=0.2
  - Answer-only masking

Usage:
    cd nexus-r
    python -m nexus_v1.training.train_bpe
"""

import json
import os
import re
import sys
import time
import math
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tiktoken

from nexus_v1.architecture import NexusR, NexusConfig


# ============================================================
# Data loading — all sources
# ============================================================

def load_all_data(base_data_dir: str):
    """Load diverse QA data with knowledge oversampling (R12b config)."""
    train_texts = []
    eval_texts = []

    # --- diverse_qa ONLY (each fact in 3-10 phrasings, SHORT answers) ---
    path = os.path.join(base_data_dir, 'diverse_qa', 'train.jsonl')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
        # 90% train, 10% eval
        n_eval = max(1, len(lines) // 10)
        random.seed(42)
        random.shuffle(lines)
        train_rows = lines[n_eval:]

        # R12b: Knowledge oversampling (1.4x) to counter arithmetic data flood
        knowledge_rows = []
        arith_rows = []
        other_rows = []
        for row in train_rows:
            text = row['instruction'].lower()
            if any(c.isdigit() for c in row['output']) and any(op in text for op in ['*', '+', '-', '/', 'times', 'plus', 'minus', 'divided', 'multiply', 'sum', 'product']):
                arith_rows.append(row)
            elif any(kw in text for kw in ['capital', 'who', 'where', 'when', 'why', 'explain', 'define', 'describe', 'what is']):
                knowledge_rows.append(row)
            else:
                other_rows.append(row)

        # Oversample knowledge 1.4x, keep arithmetic and other at 1x
        import math as _m
        n_extra = int(len(knowledge_rows) * 0.4)
        random.seed(123)
        oversampled_knowledge = random.choices(knowledge_rows, k=n_extra)
        all_train = arith_rows + knowledge_rows + oversampled_knowledge + other_rows
        random.seed(42)
        random.shuffle(all_train)

        for row in all_train:
            train_texts.append(f"Q: {row['instruction']}\nA: {row['output']}")
        for row in lines[:n_eval]:
            eval_texts.append(f"Q: {row['instruction']}\nA: {row['output']}")
        print(f"  Loaded diverse_qa: {len(lines)} total ({len(all_train)} train [+{n_extra} knowledge oversample], {n_eval} eval)")
        print(f"    Breakdown: {len(arith_rows)} arithmetic, {len(knowledge_rows)} knowledge, {len(other_rows)} other")
    else:
        print(f"  WARNING: No diverse_qa found at {path}")

    print(f"  Total: {len(train_texts)} train, {len(eval_texts)} eval")
    return train_texts, eval_texts


# ============================================================
# Restricted BPE vocab — only keep tokens that appear in data
# ============================================================

def build_restricted_vocab(train_texts, eval_texts, enc, min_count=1):
    """Build a restricted vocab from tokens actually seen in data."""
    counter = Counter()
    for text in train_texts + eval_texts:
        tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
        counter.update(tokens)

    # Keep tokens seen at least min_count times (higher = smaller vocab = less overfit)
    active_tokens = sorted([t for t, c in counter.items() if c >= min_count])

    # Always include EOT
    if enc.eot_token not in active_tokens:
        active_tokens.append(enc.eot_token)
    active_tokens.sort()

    # Reserve local ID 0 as UNK for any unseen token
    # local IDs: 0=UNK, 1..N = active tokens
    bpe_to_local = {}
    local_to_bpe = {0: -1}  # UNK maps to nothing
    for local_id, bpe_id in enumerate(active_tokens, start=1):
        bpe_to_local[bpe_id] = local_id
        local_to_bpe[local_id] = bpe_id

    vocab_size = len(active_tokens) + 1  # +1 for UNK at position 0
    print(f"  Restricted vocab: {vocab_size} tokens "
          f"(from {enc.n_vocab} BPE, {len(counter)} unique in data)")
    return bpe_to_local, local_to_bpe, vocab_size


# ============================================================
# BPE Dataset with restricted vocab
# ============================================================

class BPEDataset(Dataset):
    """Per-example padding with ANSWER-ONLY loss masking (R12b config)."""
    def __init__(self, texts: list, enc, seq_len: int = 256, bpe_to_local: dict = None):
        self.seq_len = seq_len
        eot = enc.eot_token
        eot_local = bpe_to_local.get(eot, 0) if bpe_to_local else eot
        PAD_ID = 0  # UNK/PAD token

        # Pre-encode the answer marker "\nA:" to find where answers begin
        answer_marker_tokens = enc.encode("\nA:", allowed_special={'<|endoftext|>'})
        if bpe_to_local:
            answer_marker_tokens = [bpe_to_local.get(t, 0) for t in answer_marker_tokens]

        self.examples = []
        skipped = 0
        answer_masked = 0
        for text in texts:
            tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
            if bpe_to_local:
                tokens = [bpe_to_local.get(t, 0) for t in tokens]
            tokens.append(eot_local)  # EOT at end of each example

            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]  # truncate if too long
                skipped += 1

            # Pad to seq_len
            n_pad = seq_len - len(tokens)
            padded = tokens + [PAD_ID] * (n_pad + 1)  # +1 for the shift
            inp = torch.tensor(padded[:seq_len], dtype=torch.long)
            lbl = torch.tensor(padded[1:seq_len + 1], dtype=torch.long)

            # Mask padding in labels with -100 (ignore_index)
            real_len = len(tokens)  # includes EOT
            if real_len < seq_len:
                lbl[real_len - 1:] = -100  # mask from after last real token

            # ANSWER-ONLY masking: find "\nA:" marker and mask everything before it
            answer_start = -1
            marker_len = len(answer_marker_tokens)
            for pos in range(len(tokens) - marker_len + 1):
                if tokens[pos:pos + marker_len] == answer_marker_tokens:
                    answer_start = pos + marker_len  # answer begins AFTER "\nA:"
                    break

            if answer_start > 0:
                lbl[:answer_start - 1] = -100
                answer_masked += 1

            self.examples.append((inp, lbl))

        if skipped > 0:
            print(f"  WARNING: {skipped} examples truncated to {seq_len} tokens")
        print(f"  Dataset: {len(self.examples)} examples (per-example padding to {seq_len})")
        print(f"  Answer-only masking applied to {answer_masked}/{len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, eval_loader, device):
    """Compute average eval loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, labels in eval_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        out = model(input_ids, labels=labels)
        total_loss += out['loss'].item()
        n_batches += 1
    model.train()
    if n_batches == 0:
        return float('inf')
    return total_loss / n_batches


@torch.no_grad()
def generate_samples(model, enc, device, prompts, max_tokens=150, temperature=0.7,
                     top_k=40, bpe_to_local=None, local_to_bpe=None):
    """Generate text for prompts using restricted vocab mapping."""
    model.eval()
    results = []
    eot_local = bpe_to_local.get(enc.eot_token, 0) if bpe_to_local else enc.eot_token

    for prompt_text in prompts:
        prompt_ids = enc.encode(prompt_text, allowed_special={'<|endoftext|>'})
        if bpe_to_local:
            prompt_ids = [bpe_to_local.get(t, 0) for t in prompt_ids]
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated = model.generate(prompt_tensor, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)
        gen_local_ids = generated[0].tolist()[len(prompt_ids):]

        # Stop at EOT
        if eot_local in gen_local_ids:
            gen_local_ids = gen_local_ids[:gen_local_ids.index(eot_local)]

        # Map back to BPE IDs for decoding
        if local_to_bpe:
            gen_bpe_ids = [local_to_bpe.get(lid, 0) for lid in gen_local_ids if lid > 0]
        else:
            gen_bpe_ids = gen_local_ids

        gen_text = enc.decode(gen_bpe_ids) if gen_bpe_ids else ""
        results.append((prompt_text, gen_text))
    model.train()
    return results


# ============================================================
# Main training
# ============================================================

def train():
    print("=" * 70)
    print(f"Nexus-R V1 -- BPE Training V14 (Round 15 - R12b Restore)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- BPE tokenizer ----
    enc = tiktoken.get_encoding('gpt2')
    print(f"Base tokenizer: GPT-2 BPE ({enc.n_vocab} tokens)")

    # ---- Load all data ----
    base_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
    print(f"Data dir: {base_data_dir}")

    train_texts, eval_texts = load_all_data(base_data_dir)
    random.seed(42)
    random.shuffle(train_texts)
    print(f"Train texts: {len(train_texts)}, Eval texts: {len(eval_texts)}")

    # ---- Build restricted vocab ----
    bpe_to_local, local_to_bpe, VOCAB_SIZE = build_restricted_vocab(
        train_texts, eval_texts, enc, min_count=1  # R15: restored from R8
    )

    # ---- Datasets ----
    SEQ_LEN = 192  # Longer for CoT outputs (state tracking, logic chains)
    BATCH_SIZE = 4
    ACCUM_STEPS = 2  # Effective batch = BATCH_SIZE * ACCUM_STEPS = 8
    train_dataset = BPEDataset(train_texts, enc, seq_len=SEQ_LEN, bpe_to_local=bpe_to_local)
    eval_dataset = BPEDataset(eval_texts, enc, seq_len=SEQ_LEN, bpe_to_local=bpe_to_local)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # ---- Model ----
    # Round 3: Same arch but with injection fix + diverse data
    # Lower dropout (more repetition per fact), same model scale
    cfg = NexusConfig(
        vocab_size=VOCAB_SIZE,
        d_model=256,       # Round 7: more capacity for reasoning (was 192)
        n_heads=8,          # Round 7: matched to d_model=256 (was 6)
        n_kv_heads=4,       # Round 7: matched (was 3)
        n_anchor_layers=2,
        L_layers=2,
        L_cycles=2,
        H_cycles=5,         # R15: restored from R8 (was 2 in R14)
        ffn_expansion=8/3,
        max_seq_len=SEQ_LEN,
        halt_threshold=0.01,
        dropout=0.2,       # Round 8: stronger regularization (was 0.15)
    )
    model = NexusR(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = cfg.vocab_size * cfg.d_model
    print(f"Model: {total_params:,} params (embed={embed_params:,} = {100*embed_params/total_params:.1f}%)")
    print(f"  d={cfg.d_model}, heads={cfg.n_heads}, L_layers={cfg.L_layers}, "
          f"L_cycles={cfg.L_cycles}, H={cfg.H_cycles}, dropout={cfg.dropout}")

    # ---- EMA state (R12b: exponential moving average, decay=0.999) ----
    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    print(f"  EMA decay: {ema_decay}")

    # ---- Optimizer with proper weight decay groups ----
    N_EPOCHS = 60
    lr = 2e-4
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.2},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))

    total_steps = N_EPOCHS * len(train_loader) // ACCUM_STEPS
    warmup_steps = min(100, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ---- Test prompts ----
    test_prompts = [
        "Q: What is 2 + 3?\nA:",
        "Q: Explain what a variable is in programming.\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: How do you print hello world in Python?\nA:",
        "Q: What is 10 times 5?\nA:",
        "Q: What is a function?\nA:",
        # Round 7: Algorithmic micro-tasks (require reasoning)
        "Q: I have 3 apples. I eat 1. I find 2. How many do I have?\nA:",
        "Q: A is bigger than B. B is bigger than C. Is C bigger than A?\nA:",
        "Q: What is 3 * 4 + 5?\nA:",
        "Q: Reverse the letters in 'cat'.\nA:",
    ]

    # ---- Training loop ----
    print(f"\nTraining for {N_EPOCHS} epochs ({total_steps} optimizer steps, accum={ACCUM_STEPS})")
    print(f"LR: {lr} with cosine schedule, warmup={warmup_steps} steps")
    print("-" * 70)

    best_eval_loss = float('inf')
    train_losses = []
    eval_losses = []
    global_step = 0
    patience = 0
    MAX_PATIENCE = 15  # Early stopping patience

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        halt_sims = []
        aux_losses = []

        # R12b: Annealing schedules (cosine)
        model._current_epoch = epoch
        progress = epoch / max(N_EPOCHS - 1, 1)
        model._current_noise_scale = 0.03 + 0.12 * 0.5 * (1 + math.cos(math.pi * progress))
        model._current_label_smoothing = 0.02 + 0.08 * 0.5 * (1 + math.cos(math.pi * progress))
        tau_val = 0.20 + 0.30 * 0.5 * (1 + math.cos(math.pi * progress))
        model.reasoner._repulsion_tau.fill_(tau_val)
        if epoch % 10 == 0:
            print(f"  [Schedule] epoch={epoch} noise={model._current_noise_scale:.4f} "
                  f"smoothing={model._current_label_smoothing:.4f} tau={tau_val:.4f}")

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        prog_losses = []
        for i, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)

            out = model(input_ids, labels=labels)
            loss = out['loss'] / ACCUM_STEPS
            loss.backward()

            epoch_loss += out['loss'].item()
            n_batches += 1
            if out['recursion_info']['halt_similarities']:
                halt_sims.extend(out['recursion_info']['halt_similarities'])
            if 'aux_loss' in out and hasattr(out['aux_loss'], 'item'):
                aux_losses.append(out['aux_loss'].item())
            if 'prog_loss' in out and hasattr(out['prog_loss'], 'item'):
                prog_losses.append(out['prog_loss'].item())

            # Optimizer step every ACCUM_STEPS
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # R12b: Update EMA after each optimizer step
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)
        elapsed = time.time() - t0
        avg_halt = sum(halt_sims) / max(len(halt_sims), 1)
        avg_aux = sum(aux_losses) / max(len(aux_losses), 1)
        avg_prog = sum(prog_losses) / max(len(prog_losses), 1)

        # R12b: Eval using EMA weights
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        avg_eval = evaluate(model, eval_loader, device)
        eval_losses.append(avg_eval)

        is_best = avg_eval < best_eval_loss
        if is_best:
            best_eval_loss = avg_eval
            patience = 0
            save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': ema_state,  # R12b: Save EMA weights
                'config': cfg,
                'epoch': epoch,
                'train_loss': avg_train,
                'eval_loss': avg_eval,
                'train_losses': train_losses,
                'eval_losses': eval_losses,
                'bpe_to_local': bpe_to_local,
                'local_to_bpe': local_to_bpe,
            }, os.path.join(save_dir, 'nexus_r_v1_best.pt'))
        else:
            patience += 1

        # Restore raw training weights
        model.load_state_dict(original_state)

        gap = avg_eval - avg_train
        ppl_train = math.exp(min(avg_train, 20))
        ppl_eval = math.exp(min(avg_eval, 20))

        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}  "
              f"train={avg_train:.4f} (ppl={ppl_train:.1f})  "
              f"eval={avg_eval:.4f} (ppl={ppl_eval:.1f})  "
              f"gap={gap:+.4f}  halt={avg_halt:.3f}  aux={avg_aux:.4f}  prog={avg_prog:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{elapsed:.1f}s{'  *BEST*' if is_best else ''}")

        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n  --- Generation samples (epoch {epoch+1}, EMA) ---")
            model.load_state_dict(ema_state)
            samples = generate_samples(model, enc, device, test_prompts, max_tokens=80,
                                       bpe_to_local=bpe_to_local, local_to_bpe=local_to_bpe)
            model.load_state_dict(original_state)
            for prompt, gen in samples:
                gen_short = gen[:120].replace('\n', '\\n')
                print(f"    [{prompt[:40]}...] -> {gen_short}")
            print()

        # Early stopping
        if patience >= MAX_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {MAX_PATIENCE} epochs)")
            break

    print("-" * 70)

    # ---- Final summary ----
    print("\n=== TRAINING SUMMARY ===")
    print(f"  Train loss:  {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    print(f"  Eval loss:   {eval_losses[0]:.4f} -> {eval_losses[-1]:.4f}")
    print(f"  Best eval:   {best_eval_loss:.4f}")
    print(f"  Final gap:   {eval_losses[-1] - train_losses[-1]:+.4f}")
    print(f"  Train ppl:   {math.exp(min(train_losses[-1], 20)):.1f}")
    print(f"  Eval ppl:    {math.exp(min(eval_losses[-1], 20)):.1f}")

    overfitting = eval_losses[-1] - train_losses[-1]
    if overfitting > 1.0:
        print(f"  WARNING: Significant overfitting (gap={overfitting:.2f})")
    elif train_losses[-1] > train_losses[0] * 0.9:
        print(f"  WARNING: Model barely learned (train loss dropped {(1-train_losses[-1]/train_losses[0])*100:.1f}%)")
    else:
        print(f"  OK: Training appears healthy")

    # Load EMA weights for final generation + checkpoint
    model.load_state_dict(ema_state)

    # Final generation (greedy — shows what model actually learned)
    print("\n=== FINAL GENERATION (greedy, temp=0.01, EMA) ===")
    samples = generate_samples(model, enc, device, test_prompts, max_tokens=150,
                               temperature=0.01, top_k=1,
                               bpe_to_local=bpe_to_local, local_to_bpe=local_to_bpe)
    for prompt, gen in samples:
        print(f"  PROMPT: {prompt}")
        print(f"  OUTPUT: {gen[:200]}")
        print()

    print("=== FINAL GENERATION (sampled, temp=0.7) ===")
    samples = generate_samples(model, enc, device, test_prompts, max_tokens=150,
                               bpe_to_local=bpe_to_local, local_to_bpe=local_to_bpe)
    for prompt, gen in samples:
        print(f"  PROMPT: {prompt}")
        print(f"  OUTPUT: {gen[:200]}")
        print()

    # Save final checkpoint
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': ema_state,  # R12b: Save EMA weights
        'config': cfg,
        'epoch': len(train_losses),
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'final_train_loss': train_losses[-1],
        'final_eval_loss': eval_losses[-1],
        'bpe_to_local': bpe_to_local,
        'local_to_bpe': local_to_bpe,
    }, os.path.join(save_dir, 'nexus_r_v1_final.pt'))
    print(f"Checkpoints saved to {os.path.normpath(save_dir)}")

    return {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'best_eval_loss': best_eval_loss,
        'final_samples': samples,
    }


if __name__ == '__main__':
    train()
