"""
SNAP-C1 V5: Pre-training Script
=================================
Next-token prediction on Python source code.

Phase 1 (local, RX 7600 8GB):
  - d_model=256, 4 blocks — small model to validate pipeline
  - Trains on your own codebase as seed data
  - ~15M params, fits easily in 8GB

Phase 2 (RunPod A100):
  - d_model=1536, 12 blocks — full 1.38B param model
  - Trains on large code corpus (The Stack, etc.)

Usage:
  # Quick local test (trains on RX.AI project files):
  python v5_core/training/v5_pretrain.py --quick

  # Local training on a specific directory:
  python v5_core/training/v5_pretrain.py --data_dir ./some/code/repo --epochs 10

  # Resume from checkpoint:
  python v5_core/training/v5_pretrain.py --resume v5_core/checkpoints/v5_pretrain_latest.pt

  # Full-scale RunPod training:
  python v5_core/training/v5_pretrain.py --scale runpod --data_dir /data/code_corpus --epochs 3
"""

import os
import sys
import gc
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel
from v5_core.training.data_loader import CodeFileDataset, create_pretrain_loader


# ──────────────────────────────────────────────────────────────────────────────
# Model configs
# ──────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    'tiny': dict(
        d_model=256, n_blocks=2, n_heads=4,
        window_size=32, max_seq_len=512,
        vocab_size=100279, K_hash=4, d_hash=32,
        dropout=0.0,
    ),
    'local': dict(
        d_model=512, n_blocks=4, n_heads=8,
        window_size=64, max_seq_len=1024,
        vocab_size=100279, K_hash=8, d_hash=64,
        dropout=0.0,
    ),
    'medium': dict(
        d_model=1024, n_blocks=8, n_heads=8,
        window_size=128, max_seq_len=2048,
        vocab_size=100279, K_hash=8, d_hash=128,
        dropout=0.1,
    ),
    'runpod': dict(
        d_model=1536, n_blocks=12, n_heads=12,
        window_size=128, max_seq_len=8192,
        vocab_size=100279, K_hash=8, d_hash=192,
        dropout=0.1,
    ),
}


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
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    print("=" * 60)
    print("SNAP-C1 V5 PRE-TRAINING")
    print("=" * 60)

    # Device
    device = get_device()
    print(f"Device: {device}")
    is_dml = device.type == 'privateuseone'

    # Model config
    config = CONFIGS[args.scale]
    seq_len = min(config['max_seq_len'], args.seq_len)
    config['max_think_steps'] = 0  # No THINK loop for pretrain

    print(f"\nScale: {args.scale}")
    print(f"Seq length: {seq_len}")

    # Build model
    model = V5ResonanceModel(**config).to(device)
    total, trainable = count_params(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    print(f"Model size: {total * 4 / 1024 / 1024:.1f} MB")

    # Data
    print(f"\nLoading data...")
    data_dirs = args.data_dir if args.data_dir else [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # v5_core/
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'training'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'inference'),
    ]
    # Filter to existing dirs
    data_dirs = [d for d in data_dirs if os.path.isdir(d)]

    loader = create_pretrain_loader(
        data_dirs, seq_len=seq_len, batch_size=args.batch_size,
        max_files=args.max_files,
    )

    if loader is None or len(loader.dataset) == 0:
        print("\nERROR: No training data found!")
        print("  The --data_dir must point to folders containing .py files.")
        print("")
        print("  Quick start (train on your own project):")
        print("    python v5_core/training/v5_pretrain.py --quick")
        print("")
        print("  Or download repos first:")
        print("    git clone https://github.com/pallets/flask training_data/flask")
        print("    git clone https://github.com/psf/requests training_data/requests")
        print("    python v5_core/training/v5_pretrain.py --data_dir training_data")
        return

    total_samples = len(loader.dataset)
    total_batches = len(loader)
    print(f"Batches per epoch: {total_batches}")

    # Optimizer
    # AdamW works on DirectML but aten::lerp falls back to CPU (warning, not crash)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # OneCycleLR scheduler
    total_steps = total_batches * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,         # 10% warmup
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            global_step = ckpt.get('global_step', 0)
            best_loss = ckpt.get('best_loss', float('inf'))
            print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_loss={best_loss:.4f}")
        else:
            print(f"WARNING: Checkpoint {args.resume} not found, starting fresh")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Training: {args.epochs} epochs, {total_batches} batches/epoch")
    print(f"Optimizer: AdamW lr={args.lr}, wd={args.weight_decay}")
    print(f"Grad clip: {args.grad_clip}")
    print(f"{'=' * 60}\n")

    model.train()
    train_start = time.time()
    running_loss = 0.0
    log_interval = max(1, min(total_batches // 20, 200))  # Log frequently (cap at every 200 batches)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, (token_ids, type_ids, labels) in enumerate(loader):
            token_ids = token_ids.to(device)
            type_ids = type_ids.to(device)
            labels = labels.to(device)

            # Forward
            result = model.forward_pretrain(token_ids, type_ids, labels)
            loss = result['loss']

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
                save_path = ckpt_dir / f"v5_pretrain_step{global_step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'best_loss': best_loss,
                    'config': config,
                    'args': vars(args),
                }, save_path)
                print(f"  💾 Saved checkpoint: {save_path}")

            # Memory cleanup for 8GB VRAM
            if is_dml and batch_idx % 10 == 0:
                gc.collect()

        # Epoch summary
        epoch_avg_loss = epoch_loss / total_batches
        epoch_time = time.time() - epoch_start

        print(f"\n  ═══ Epoch {epoch+1}/{args.epochs} complete ═══")
        print(f"  Avg loss: {epoch_avg_loss:.4f}")
        print(f"  Time: {format_time(epoch_time)}")
        print(f"  Tokens processed: {epoch_tokens:,}\n")

        # Save best model
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            save_path = ckpt_dir / "v5_pretrain_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_loss': best_loss,
                'config': config,
            }, save_path)
            print(f"  ★ New best! Saved to {save_path}")

        # Save latest (always)
        save_path = ckpt_dir / "v5_pretrain_latest.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'global_step': global_step,
            'best_loss': best_loss,
            'config': config,
            'args': vars(args),
        }, save_path)

    # Done
    total_time = time.time() - train_start
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Best loss:   {best_loss:.4f}")
    print(f"  Total steps: {global_step}")
    print(f"  Checkpoint:  {ckpt_dir / 'v5_pretrain_best.pt'}")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Pre-training")

    # Data
    parser.add_argument('--data_dir', nargs='+', default=None,
                        help='Directories of source files to train on')
    parser.add_argument('--max_files', type=int, default=10000,
                        help='Max files to load')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Training sequence length')

    # Model
    parser.add_argument('--scale', choices=['tiny', 'local', 'medium', 'runpod'],
                        default='tiny', help='Model scale')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 for 8GB VRAM)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', default='v5_core/checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    # Quick mode: tiny model, trains on project files, 3 epochs
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: tiny model, project files, 3 epochs')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.scale = 'tiny'
        args.epochs = 3
        args.seq_len = 256
        # Use the entire RX.AI project as training data
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.data_dir = [project_root]

    train(args)


if __name__ == "__main__":
    main()
