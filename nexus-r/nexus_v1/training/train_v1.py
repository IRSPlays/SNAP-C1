"""
Nexus-R V1 Training Script
============================
Trains the tiny Nexus-R model on real SNAP-C1 team_thinking data.

Uses character-level tokenization (appropriate for architecture validation
on a from-scratch ~4M param model). The goal is to verify:
  1. Loss decreases consistently over epochs
  2. Recursive halting adapts (cosine similarity converges)
  3. No NaN/Inf instability
  4. The model learns real patterns (not just random)

Usage:
    cd nexus-r
    python -m nexus_v1.training.train_v1
"""

import json
import os
import sys
import time
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nexus_v1.architecture import NexusR, NexusConfig


# ============================================================
# Character-level tokenizer
# ============================================================

class CharTokenizer:
    """Minimal character tokenizer for architecture validation."""
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        # Reserve special tokens
        self._add_special('<pad>', 0)
        self._add_special('<bos>', 1)
        self._add_special('<eos>', 2)
        self._add_special('<unk>', 3)

    def _add_special(self, token: str, idx: int):
        self.char_to_id[token] = idx
        self.id_to_char[idx] = token
        self.vocab_size = max(self.vocab_size, idx + 1)

    def fit(self, texts: list[str]):
        """Build vocabulary from corpus."""
        chars = set()
        for t in texts:
            chars.update(t)
        for ch in sorted(chars):
            if ch not in self.char_to_id:
                idx = self.vocab_size
                self.char_to_id[ch] = idx
                self.id_to_char[idx] = ch
                self.vocab_size += 1
        print(f"  Tokenizer: {self.vocab_size} tokens ({self.vocab_size - 4} unique chars)")

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, 3) for c in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids if i > 2)


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    """Packs instruction+output into fixed-length chunks for causal LM."""
    def __init__(self, texts: list[str], tokenizer: CharTokenizer, seq_len: int = 256):
        self.seq_len = seq_len

        # Tokenize everything into one big sequence
        all_ids = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text))
            all_ids.append(2)  # <eos>

        # Chunk into seq_len+1 pieces (input + next-token label)
        self.chunks = []
        for i in range(0, len(all_ids) - seq_len, seq_len // 2):  # Overlapping
            chunk = all_ids[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(f"  Dataset: {len(self.chunks)} chunks of {seq_len} tokens from {len(texts)} texts")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]  # input, target


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        out = model(input_ids, labels=labels)
        total_loss += out['loss'].item()
        n_batches += 1
    model.train()
    if n_batches == 0:
        return float('inf')
    return total_loss / n_batches


# ============================================================
# Training loop
# ============================================================

def train():
    print("=" * 60)
    print("Nexus-R V1 — Real Data Training")
    print("=" * 60)

    # ---- Load data ----
    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'data', 'team_thinking', 'train.jsonl'
    )
    data_path = os.path.normpath(data_path)
    print(f"\nLoading data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            # Combine instruction + output as full text
            texts.append(f"Q: {row['instruction']}\nA: {row['output']}")

    print(f"  Loaded {len(texts)} examples")

    rng = random.Random(42)
    rng.shuffle(texts)
    n_eval = max(1, len(texts) // 10) if len(texts) > 1 else 0
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:] if n_eval else texts
    if not train_texts:
        train_texts = texts
        eval_texts = texts[:1]
    print(f"  Split: {len(train_texts)} train, {len(eval_texts)} eval")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---- Tokenizer ----
    print("\nBuilding tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.fit(train_texts + eval_texts)

    # ---- Dataset ----
    seq_len = 256
    train_dataset = TextDataset(train_texts, tokenizer, seq_len=seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, seq_len=seq_len)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, drop_last=False)

    # ---- Model ----
    print("\nBuilding model...")
    cfg = NexusConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,
        n_anchor_layers=2,
        L_layers=2,
        L_cycles=2,
        H_cycles=2,
        max_seq_len=seq_len,
        halt_threshold=0.001,
    )
    model = NexusR(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} params")
    print(f"  Config: d={cfg.d_model}, heads={cfg.n_heads}, "
          f"L_layers={cfg.L_layers}, L_cycles={cfg.L_cycles}, H={cfg.H_cycles}")

    # ---- Optimizer ----
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or name.endswith('bias') or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
    )

    n_epochs = 10
    total_steps = max(1, n_epochs * len(loader))
    warmup_steps = min(100, max(1, total_steps // 10))

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ---- Training ----
    print(f"\nTraining for {n_epochs} epochs, {len(loader)} batches/epoch")
    print(f"Validation batches: {len(eval_loader)}")
    print("-" * 60)

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'nexus_r_v1_best.pt')
    latest_path = os.path.join(save_dir, 'nexus_r_v1_latest.pt')

    best_eval_loss = float('inf')
    train_losses = []
    eval_losses = []

    for epoch in range(n_epochs):
        model.train()
        model._current_noise_scale = 0.0
        model._current_label_smoothing = 0.0
        total_loss = 0.0
        n_batches = 0
        total_steps = 0
        halt_sims = []

        t0 = time.time()
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            out = model(input_ids, labels=labels)
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            total_steps += out['recursion_info']['total_recursive_steps']
            if out['recursion_info']['halt_similarities']:
                halt_sims.extend(out['recursion_info']['halt_similarities'])

        avg_loss = total_loss / max(n_batches, 1)
        avg_eval = evaluate(model, eval_loader, device)
        train_losses.append(avg_loss)
        eval_losses.append(avg_eval)
        elapsed = time.time() - t0
        avg_halt = sum(halt_sims) / max(len(halt_sims), 1)

        if avg_eval < best_eval_loss:
            best_eval_loss = avg_eval
            marker = " *"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': cfg,
                'tokenizer_vocab': tokenizer.char_to_id,
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'eval_loss': avg_eval,
                'train_losses': train_losses,
                'eval_losses': eval_losses,
            }, best_path)
        else:
            marker = ""

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': cfg,
            'tokenizer_vocab': tokenizer.char_to_id,
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'eval_loss': avg_eval,
            'train_losses': train_losses,
            'eval_losses': eval_losses,
        }, latest_path)

        print(f"  Epoch {epoch+1:3d}/{n_epochs}  train={avg_loss:.4f}  eval={avg_eval:.4f}  "
              f"halt_sim={avg_halt:.4f}  steps={total_steps}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.1f}s{marker}")

    print("-" * 60)

    # ---- Validation ----
    print("\n=== Validation ===")
    print(f"  Initial train:   {train_losses[0]:.4f}")
    print(f"  Final train:     {train_losses[-1]:.4f}")
    print(f"  Initial eval:    {eval_losses[0]:.4f}")
    print(f"  Final eval:      {eval_losses[-1]:.4f}")
    print(f"  Best eval:       {best_eval_loss:.4f}")
    print(f"  Train improved:  {train_losses[-1] < train_losses[0]}")

    # Perplexity
    ppl = math.exp(min(eval_losses[-1], 20.0))  # Cap to avoid overflow
    print(f"  Final eval perplexity: {ppl:.1f}")

    # Verify loss meaningfully decreased
    if eval_losses[-1] < eval_losses[0] * 0.8:
        print("  PASS: Eval loss decreased by >20%")
    else:
        print("  WARN: Eval loss didn't decrease much — may need more epochs or tuning")

    # ---- Generate sample ----
    print("\n=== Sample Generation ===")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    prompt_text = "Q: What is 2 + 3?\nA:"
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(prompt_tensor, max_new_tokens=100, temperature=0.8, eos_token_id=2)

    gen_text = tokenizer.decode(generated[0].tolist())
    print(f"  Prompt: {prompt_text}")
    print(f"  Generated: {gen_text[:200]}")

    print(f"\n  Best checkpoint:   {os.path.normpath(best_path)}")
    print(f"  Latest checkpoint: {os.path.normpath(latest_path)}")

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == '__main__':
    train()
