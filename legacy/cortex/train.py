"""Cortex V1 — Phase 1 Training

Encoder → PredictiveCoder → LTC-RNN → Decoder
Standard next-token prediction. No hippocampus/neuromodulator yet.

Usage:
    python -m cortex.train
"""

import json
import time
import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from cortex.model import CortexV1
from cortex.tokenizer import train_tokenizer, load_tokenizer, PAD_ID, BOS_ID, EOS_ID


# ── Config ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 8192
    d_model: int = 256
    d_key: int = 512
    n_pc_layers: int = 3
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 30
    grad_clip: float = 1.0
    warmup_steps: int = 200

    # Data
    data_dir: str = 'data/diverse_qa'
    tokenizer_path: str = 'cortex/tokenizer.json'
    checkpoint_dir: str = 'cortex/checkpoints'

    # Phase
    use_memory: bool = False  # Phase 1 = False


# ── Dataset ─────────────────────────────────────────────────────────

class QADataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = f"{item['instruction']}\n{item['output']}"
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids[:self.max_len]

        # Pad to max_len
        pad_len = self.max_len - len(ids)
        ids = ids + [PAD_ID] * pad_len

        return torch.tensor(ids, dtype=torch.long)


# ── Training Loop ───────────────────────────────────────────────────

def get_lr(step: int, config: TrainConfig) -> float:
    """Warmup + cosine decay schedule."""
    if step < config.warmup_steps:
        return config.lr * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(1, config.epochs * 100 - config.warmup_steps)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train():
    config = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Tokenizer ──
    tok_path = Path(config.tokenizer_path)
    if not tok_path.exists():
        print('Training BPE tokenizer...')
        data_files = list(Path(config.data_dir).glob('*.jsonl'))
        if not data_files:
            raise FileNotFoundError(f'No .jsonl files in {config.data_dir}')
        tokenizer = train_tokenizer(
            [str(f) for f in data_files],
            vocab_size=config.vocab_size,
            save_path=config.tokenizer_path,
        )
    else:
        print(f'Loading tokenizer from {tok_path}')
        tokenizer = load_tokenizer(tok_path)

    actual_vocab = tokenizer.get_vocab_size()
    config.vocab_size = actual_vocab
    print(f'Vocab size: {actual_vocab}')

    # ── Data ──
    train_path = Path(config.data_dir) / 'train.jsonl'
    eval_path = Path(config.data_dir) / 'eval_suite.jsonl'

    train_ds = QADataset(str(train_path), tokenizer, config.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)

    eval_ds = None
    eval_loader = None
    if eval_path.exists():
        eval_ds = QADataset(str(eval_path), tokenizer, config.max_seq_len)
        eval_loader = DataLoader(eval_ds, batch_size=config.batch_size, shuffle=False)

    print(f'Train samples: {len(train_ds)}, Eval samples: {len(eval_ds) if eval_ds else 0}')
    print(f'Batches/epoch: {len(train_loader)}')

    # ── Model ──
    model = CortexV1(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_key=config.d_key,
        n_pc_layers=config.n_pc_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        use_memory=config.use_memory,
    ).to(device)

    param_counts = model.count_parameters()
    print(f'\nParameter counts:')
    for name, count in param_counts.items():
        print(f'  {name}: {count:,}')
    total_mb = param_counts['total'] * 4 / 1024 / 1024
    print(f'  VRAM estimate: {total_mb:.1f} MB (fp32)\n')

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # ── Training ──
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device)  # [B, T]

            # Next-token prediction: input = tokens[:-1], target = tokens[1:]
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # Forward
            out = model(input_ids)
            logits = out['logits']  # [B, T-1, vocab_size]

            loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # LR schedule
            lr = get_lr(global_step, config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.step()

            # Stats
            non_pad = (target_ids != PAD_ID).sum().item()
            epoch_loss += loss.item() * non_pad
            epoch_tokens += non_pad
            global_step += 1

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        tok_per_sec = epoch_tokens / elapsed

        print(f'Epoch {epoch+1:3d}/{config.epochs} | '
              f'loss={avg_loss:.4f} | ppl={ppl:.1f} | '
              f'lr={lr:.2e} | {tok_per_sec:.0f} tok/s | {elapsed:.1f}s')

        # ── Eval ──
        if eval_loader is not None:
            model.eval()
            eval_loss = 0.0
            eval_tokens = 0

            with torch.no_grad():
                for batch in eval_loader:
                    batch = batch.to(device)
                    input_ids = batch[:, :-1]
                    target_ids = batch[:, 1:]

                    out = model(input_ids)
                    logits = out['logits']
                    loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))

                    non_pad = (target_ids != PAD_ID).sum().item()
                    eval_loss += loss.item() * non_pad
                    eval_tokens += non_pad

            avg_eval = eval_loss / max(eval_tokens, 1)
            eval_ppl = math.exp(min(avg_eval, 20))
            print(f'          eval_loss={avg_eval:.4f} | eval_ppl={eval_ppl:.1f}')

            if avg_eval < best_eval_loss:
                best_eval_loss = avg_eval
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_loss': avg_eval,
                    'config': config,
                }, ckpt_dir / 'best.pt')
                print(f'          ** saved best checkpoint (eval_loss={avg_eval:.4f})')

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, ckpt_dir / f'epoch_{epoch+1}.pt')

    print(f'\nTraining complete. Best eval loss: {best_eval_loss:.4f}')
    print(f'Checkpoints in: {ckpt_dir}')


if __name__ == '__main__':
    train()
