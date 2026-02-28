"""
SNAP-C1 V5: Instruction-Tuning Script (Phase 2)
=================================================
Fine-tunes a pre-trained V5 model on instruction-following data.
Teaches the model to: fix bugs, follow instructions, generate code,
use chain-of-thought, and handle tool-use patterns.

Data Sources (loaded automatically):
  1. v4_core/data/v4_instruction_dataset.json       — 5,000 bug-fix pairs
  2. data/self_correction/train.jsonl                — 50 self-correction samples
  3. data/team_thinking/train.jsonl                  — 40 multi-agent debate samples
  4. data/tool_use/train.jsonl                       — 50 tool-use samples

Usage:
  # Standard instruction-tuning from pre-trained checkpoint:
  python v5_core/training/v5_instruct.py --pretrained v5_core/checkpoints/v5_pretrain_best.pt

  # Resume interrupted instruction-tuning:
  python v5_core/training/v5_instruct.py --resume v5_core/checkpoints/v5_instruct_latest.pt

  # Custom data (additional datasets):
  python v5_core/training/v5_instruct.py --pretrained v5_core/checkpoints/v5_pretrain_best.pt \\
      --extra_data my_data.jsonl another.json

  # Quick test with small subset:
  python v5_core/training/v5_instruct.py --pretrained v5_core/checkpoints/v5_pretrain_best.pt --quick
"""

import os
import sys
import gc
import time
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel
from v5_core.training.data_loader import InstructionDataset

from torch.utils.data import DataLoader, ConcatDataset


# ──────────────────────────────────────────────────────────────────────────────
# Default data paths (relative to project root)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_DATASETS = [
    PROJECT_ROOT / "v4_core" / "data" / "v4_instruction_dataset.json",
    PROJECT_ROOT / "data" / "self_correction" / "train.jsonl",
    PROJECT_ROOT / "data" / "team_thinking" / "train.jsonl",
    PROJECT_ROOT / "data" / "tool_use" / "train.jsonl",
]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    print("=" * 60)
    print("SNAP-C1 V5 INSTRUCTION-TUNING (Phase 2)")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────
    device = get_device()
    print(f"Device: {device}")
    is_dml = device.type == 'privateuseone'

    # ── Load pre-trained checkpoint ───────────────────────────────────────
    pretrained_path = args.pretrained or args.resume
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"\nERROR: Must provide --pretrained or --resume checkpoint!")
        print(f"  Example: python v5_core/training/v5_instruct.py --pretrained v5_core/checkpoints/v5_pretrain_best.pt")
        return

    print(f"\nLoading checkpoint: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    # Extract model config from checkpoint
    config = ckpt.get('config', None)
    if config is None:
        print("ERROR: Checkpoint has no 'config' key. Is this a V5 checkpoint?")
        return

    # Ensure no THINK loop during instruction-tuning
    config['max_think_steps'] = 0

    # Enable dropout during fine-tuning to prevent memorization
    config['dropout'] = args.dropout

    print(f"  Model config: d_model={config['d_model']}, n_blocks={config['n_blocks']}, "
          f"n_heads={config['n_heads']}, dropout={config['dropout']}")

    # ── Build model ───────────────────────────────────────────────────────
    model = V5ResonanceModel(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)

    total, trainable = count_params(model)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # ── Optional: Freeze embedding for stability ─────────────────────────
    if args.freeze_embed:
        for p in model.encoder.embedding.parameters():
            p.requires_grad = False
        _, trainable = count_params(model)
        print(f"  Froze embedding → {trainable:,} trainable")

    # ── Load datasets ─────────────────────────────────────────────────────
    print(f"\nLoading instruction data...")
    seq_len = min(config.get('max_seq_len', 512), args.seq_len)

    datasets = []
    total_samples = 0

    # Load default datasets
    for dp in DEFAULT_DATASETS:
        if dp.exists():
            try:
                ds = InstructionDataset(str(dp), seq_len=seq_len)
                if len(ds) > 0:
                    datasets.append(ds)
                    total_samples += len(ds)
                    print(f"    ✓ {dp.name}: {len(ds)} samples")
            except Exception as e:
                print(f"    ✗ {dp.name}: {e}")
        else:
            print(f"    - {dp.name}: not found (skipping)")

    # Load extra datasets
    if args.extra_data:
        for dp in args.extra_data:
            if os.path.exists(dp):
                try:
                    ds = InstructionDataset(dp, seq_len=seq_len)
                    if len(ds) > 0:
                        datasets.append(ds)
                        total_samples += len(ds)
                        print(f"    ✓ {Path(dp).name}: {len(ds)} samples")
                except Exception as e:
                    print(f"    ✗ {Path(dp).name}: {e}")

    if not datasets:
        print("\nERROR: No instruction data found!")
        return

    # Combine all datasets
    combined = ConcatDataset(datasets)
    print(f"\n  Total: {total_samples} instruction samples")

    # Quick mode: use small subset
    if args.quick:
        subset_size = min(200, len(combined))
        indices = random.sample(range(len(combined)), subset_size)
        combined = torch.utils.data.Subset(combined, indices)
        print(f"  Quick mode: using {subset_size} samples")

    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    total_batches = len(loader)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Lower LR than pre-training (fine-tuning should be gentler)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # OneCycleLR with cosine annealing
    total_steps = total_batches * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.05,          # 5% warmup (gentler for fine-tuning)
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )

    # ── Resume state ──────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if args.resume:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"\n  Resumed at epoch {start_epoch}, step {global_step}, "
              f"best_loss={best_loss:.4f}")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Instruction-tuning: {args.epochs} epochs, {total_batches} batches/epoch")
    print(f"Optimizer: AdamW lr={args.lr}, wd={args.weight_decay}")
    print(f"Grad clip: {args.grad_clip}")
    if args.freeze_embed:
        print(f"Embedding: FROZEN")
    print(f"{'=' * 60}\n")

    model.train()
    train_start = time.time()
    running_loss = 0.0
    log_interval = max(1, min(total_batches // 20, 100))  # Log frequently

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_response_tokens = 0  # Only response tokens (not masked)

        for batch_idx, (token_ids, type_ids, labels) in enumerate(loader):
            token_ids = token_ids.to(device)
            type_ids = type_ids.to(device)
            labels = labels.to(device)

            # Forward — uses same forward_pretrain with label masking
            # Labels have -100 for prompt tokens, actual IDs for response tokens
            result = model.forward_pretrain(token_ids, type_ids, labels)
            loss = result['loss']

            # Check for NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # Tracking
            loss_val = loss.item()
            epoch_loss += loss_val
            running_loss += loss_val
            epoch_tokens += token_ids.numel()
            epoch_response_tokens += (labels != -100).sum().item()
            global_step += 1

            # Logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                avg_loss = running_loss / log_interval if batch_idx > 0 else loss_val
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                tokens_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                print(f"  [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{total_batches} | "
                      f"loss={loss_val:.4f} avg={avg_loss:.4f} | "
                      f"lr={lr:.2e} | {tokens_per_sec:.0f} tok/s")
                running_loss = 0.0

            # Interim checkpoints
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = ckpt_dir / f"v5_instruct_step{global_step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'best_loss': best_loss,
                    'config': config,
                    'args': vars(args),
                    'phase': 'instruct',
                }, save_path)
                print(f"  💾 Saved checkpoint: {save_path}")

            # Memory cleanup for 8GB VRAM
            if is_dml and batch_idx % 10 == 0:
                gc.collect()

            del loss, result
            gc.collect()

        # ── Epoch summary ─────────────────────────────────────────────────
        epoch_avg_loss = epoch_loss / max(total_batches, 1)
        epoch_time = time.time() - epoch_start

        print(f"\n  ═══ Epoch {epoch+1}/{args.epochs} complete ═══")
        print(f"  Avg loss:        {epoch_avg_loss:.4f}")
        print(f"  Time:            {format_time(epoch_time)}")
        print(f"  Total tokens:    {epoch_tokens:,}")
        print(f"  Response tokens: {epoch_response_tokens:,} "
              f"({epoch_response_tokens * 100 / max(epoch_tokens, 1):.1f}% of total)\n")

        # Save best model
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            save_path = ckpt_dir / "v5_instruct_best.pt"
            # Save with dropout=0 so inference doesn't need to worry about it
            save_config = {k: v for k, v in config.items()}
            save_config['dropout'] = 0.0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_loss': best_loss,
                'config': save_config,
                'phase': 'instruct',
            }, save_path)
            print(f"  ★ New best! Saved to {save_path}")

        # Early stopping check — if loss is very low, model is memorizing
        if epoch_avg_loss < args.early_stop_loss:
            print(f"  ⚠ Early stopping: avg_loss={epoch_avg_loss:.4f} < threshold={args.early_stop_loss}")
            print(f"    Model is memorizing. Stopping to preserve generalization.")
            break

        # Save latest (always)
        save_path = ckpt_dir / "v5_instruct_latest.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'global_step': global_step,
            'best_loss': best_loss,
            'config': config,
            'args': vars(args),
            'phase': 'instruct',
        }, save_path)

    # ── Done ──────────────────────────────────────────────────────────────
    total_time = time.time() - train_start
    print(f"\n{'=' * 60}")
    print(f"INSTRUCTION-TUNING COMPLETE")
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Best loss:   {best_loss:.4f}")
    print(f"  Total steps: {global_step}")
    print(f"  Checkpoint:  {ckpt_dir / 'v5_instruct_best.pt'}")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Instruction-Tuning")

    # Data
    parser.add_argument('--extra_data', nargs='*', default=None,
                        help='Additional instruction dataset files (JSON/JSONL)')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Training sequence length')

    # Model / checkpoint
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pre-trained checkpoint (Phase 1 output)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume instruction-tuning from this checkpoint')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze embedding layer for stability')

    # Training
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 for 8GB VRAM)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Peak learning rate (lower than pre-training)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate during fine-tuning (prevents memorization)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--early_stop_loss', type=float, default=0.5,
                        help='Stop when avg epoch loss falls below this (prevents memorization)')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', default='v5_core/checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N steps (0 to disable)')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 200 samples, 2 epochs')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 2
        args.save_every = 100

    train(args)


if __name__ == "__main__":
    main()
