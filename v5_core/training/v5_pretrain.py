"""
SNAP-C1 V5: Pre-training Script
=================================
Next-token prediction on Python source code.

Scales:
  tiny   — d=256,  2 blocks  (~15M params)   local quick test
  local  — d=512,  4 blocks  (~65M params)   local training
  medium — d=1024, 8 blocks  (~332M params)  medium GPU
  4b     — d=2560, 28 blocks (~4.4B params)  RunPod RTX 6000 Ada (48GB)
  runpod — d=1536, 12 blocks (~1.38B params) RunPod A100

Usage:
  # Quick local test (trains on RX.AI project files):
  python v5_core/training/v5_pretrain.py --quick

  # Local training on a specific directory:
  python v5_core/training/v5_pretrain.py --data_dir ./some/code/repo --epochs 10

  # Resume from checkpoint:
  python v5_core/training/v5_pretrain.py --resume v5_core/checkpoints/v5_pretrain_latest.pt

  # 4B RunPod training (RTX 6000 Ada, 48GB):
  python v5_core/training/v5_pretrain.py --scale 4b --data_dir /workspace/training_data --epochs 3 --batch_size 16 --seq_len 1024 --lr 2e-4 --save_every 200
"""

import os
import sys
import gc
import glob
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 8-bit optimizer (saves ~50% optimizer memory)
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

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
    '4b': dict(
        d_model=2560, n_blocks=28, n_heads=16,
        window_size=128, max_seq_len=2048,
        vocab_size=100279, K_hash=8, d_hash=256,
        dropout=0.1,
    ),
    # DigitalOcean H200 (141 GB VRAM) — ~7.8B params
    # bf16: ~15.6 GB model, bs=32 seq=4096 fits ~100 GB with grad ckpt
    '8b': dict(
        d_model=3072, n_blocks=32, n_heads=24,
        window_size=256, max_seq_len=4096,
        vocab_size=100279, K_hash=8, d_hash=384,
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
# Chunked cross-entropy: avoids OOM on large vocab (100K) at big batch sizes
# F.cross_entropy casts to fp32 internally → 16*1024*100279*4 = 6.12 GiB
# Chunking processes pieces at a time so the fp32 buffer is smaller
# ──────────────────────────────────────────────────────────────────────────────
def chunked_cross_entropy(logits, labels, chunk_size=1024, ignore_index=-100):
    """Compute cross-entropy in chunks to avoid OOM on large vocab."""
    B, S, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)

    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    total_count = 0

    for i in range(0, logits_flat.size(0), chunk_size):
        chunk_logits = logits_flat[i:i + chunk_size]
        chunk_labels = labels_flat[i:i + chunk_size]

        valid = (chunk_labels != ignore_index).sum().item()
        if valid == 0:
            continue

        chunk_loss = F.cross_entropy(
            chunk_logits.float(), chunk_labels,
            ignore_index=ignore_index, reduction='sum'
        )
        total_loss = total_loss + chunk_loss
        total_count += valid

    if total_count == 0:
        return logits.sum() * 0.0
    return total_loss / total_count


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint cleanup: delete old step checkpoints to save disk space
# ──────────────────────────────────────────────────────────────────────────────
def cleanup_step_checkpoints(ckpt_dir, keep_latest=True):
    """Delete old step checkpoints, keep only the latest one + best."""
    step_files = sorted(glob.glob(str(Path(ckpt_dir) / "v5_pretrain_step*.pt")))
    if keep_latest and len(step_files) > 1:
        for f in step_files[:-1]:
            os.remove(f)
            print(f"  Cleaned up old checkpoint: {os.path.basename(f)}")
    elif not keep_latest:
        for f in step_files:
            os.remove(f)
            print(f"  Cleaned up checkpoint: {os.path.basename(f)}")


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    print("=" * 60)
    print("SNAP-C1 V5 PRE-TRAINING")
    print("=" * 60)

    # ── Device setup ───────────────────────────────────────────────────────
    device = get_device()
    print(f"Device: {device}")
    is_cuda = device.type == 'cuda'
    is_dml = device.type == 'privateuseone'

    # Use bf16 on CUDA for 50% memory savings
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    print(f"Precision: {'bf16' if use_bf16 else 'fp32'}")

    # ── Model config ──────────────────────────────────────────────────────
    config = CONFIGS[args.scale]
    seq_len = min(config['max_seq_len'], args.seq_len)
    config['max_think_steps'] = 0  # No THINK loop for pretrain

    print(f"\nScale: {args.scale}")
    print(f"Seq length: {seq_len}")

    # Build model
    if use_bf16:
        model = V5ResonanceModel(**config).to(dtype=torch.bfloat16, device=device)
    else:
        model = V5ResonanceModel(**config).to(device)

    total, trainable = count_params(model)
    bytes_per_param = 2 if use_bf16 else 4
    model_mb = total * bytes_per_param / 1024 / 1024
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    print(f"Model size: {model_mb:.1f} MB ({'bf16' if use_bf16 else 'fp32'})")

    # Enable gradient checkpointing for large models (saves ~40% activation memory)
    if args.scale in ('4b', '8b', 'runpod', 'medium'):
        if hasattr(model.resonance, 'enable_gradient_checkpointing'):
            model.resonance.enable_gradient_checkpointing()
            print("Gradient checkpointing: ENABLED")
        else:
            # Manual fallback
            from torch.utils.checkpoint import checkpoint as ckpt_fn
            for block in model.resonance.blocks:
                block._original_forward = block.forward
                block.forward = lambda x, causal=True, _b=block: ckpt_fn(
                    _b._original_forward, x, causal, use_reentrant=False
                )
            print("Gradient checkpointing: ENABLED (manual fallback)")

    if is_cuda:
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_used = torch.cuda.memory_allocated() / 1024**3
        print(f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB after model load")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\nLoading data...")
    data_dirs = args.data_dir if args.data_dir else [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'training'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'inference'),
    ]
    data_dirs = [d for d in data_dirs if os.path.isdir(d)]

    loader = create_pretrain_loader(
        data_dirs, seq_len=seq_len, batch_size=args.batch_size,
        max_files=args.max_files,
    )

    if loader is None or len(loader.dataset) == 0:
        print("\nERROR: No training data found!")
        print("  --data_dir must point to folders containing .py files.")
        print("  Quick start: python v5_core/training/v5_pretrain.py --quick")
        return

    total_samples = len(loader.dataset)
    total_batches = len(loader)
    print(f"Samples: {total_samples}, Batches per epoch: {total_batches}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    if is_cuda and HAS_BNB and args.scale in ('4b', '8b', 'runpod'):
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        print(f"Optimizer: AdamW 8-bit (bitsandbytes)")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        print(f"Optimizer: AdamW (standard)")

    # OneCycleLR scheduler
    total_steps = total_batches * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────
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

    # Get stride-1 boundary for chunked loss computation
    stride1_boundary = model.encoder.elastic.boundaries[0]

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Training: {args.epochs} epochs, {total_batches} batches/epoch")
    print(f"Grad clip: {args.grad_clip}")
    print(f"Checkpoint cleanup: ENABLED (keeps only latest + best)")
    print(f"{'=' * 60}\n")

    model.train()
    train_start = time.time()
    running_loss = 0.0
    log_interval = max(1, min(total_batches // 20, 200))

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, (token_ids, type_ids, labels) in enumerate(loader):
            token_ids = token_ids.to(device)
            type_ids = type_ids.to(device)
            labels = labels.to(device)

            # ── Forward pass ────────────────────────────────────────────
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Get logits only (no loss) to avoid OOM in model's F.cross_entropy
                    result = model.forward_pretrain(token_ids, type_ids, labels=None)
                    logits = result['logits']

                    # Compute loss on stride-1 (uncompressed) slots only, using chunks
                    S = logits.size(1)
                    s1 = min(stride1_boundary, S)
                    logits_s1 = logits[:, :s1, :]
                    labels_s1 = labels[:, :s1]

                    # Chunked cross-entropy avoids 6+ GB fp32 allocation
                    loss = chunked_cross_entropy(logits_s1, labels_s1, chunk_size=1024)
            else:
                result = model.forward_pretrain(token_ids, type_ids, labels)
                loss = result['loss']

            # ── Backward pass ───────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # ── Tracking ────────────────────────────────────────────────
            loss_val = loss.item()
            epoch_loss += loss_val
            running_loss += loss_val
            epoch_tokens += token_ids.numel()
            global_step += 1

            # Free logits immediately to save VRAM
            del logits, result, loss
            if use_bf16:
                del logits_s1, labels_s1

            # ── Logging ─────────────────────────────────────────────────
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                avg_loss = running_loss / log_interval if batch_idx > 0 else loss_val
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                tokens_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                vram_str = ""
                if is_cuda:
                    vram_alloc = torch.cuda.memory_allocated() / 1024**3
                    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
                    vram_str = f" | vram={vram_alloc:.1f}/{vram_peak:.1f}GB"
                print(f"  [{epoch+1}/{args.epochs}] batch {batch_idx+1}/{total_batches} | "
                      f"loss={loss_val:.4f} avg={avg_loss:.4f} | "
                      f"lr={lr:.2e} | {tokens_per_sec:.0f} tok/s{vram_str}")
                running_loss = 0.0

            # ── Interim checkpoints ─────────────────────────────────────
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
                print(f"  Saved checkpoint: {save_path}")
                # Clean up old step checkpoints to save disk
                cleanup_step_checkpoints(ckpt_dir)

            # Memory cleanup
            if is_dml and batch_idx % 10 == 0:
                gc.collect()
            if is_cuda and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        # ── Epoch summary ──────────────────────────────────────────────
        epoch_avg_loss = epoch_loss / total_batches
        epoch_time = time.time() - epoch_start

        print(f"\n  === Epoch {epoch+1}/{args.epochs} complete ===")
        print(f"  Avg loss: {epoch_avg_loss:.4f}")
        print(f"  Time: {format_time(epoch_time)}")
        print(f"  Tokens processed: {epoch_tokens:,}\n")

        # Save best model
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            save_path = ckpt_dir / f"v5_pretrain_best_{args.scale}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_loss': best_loss,
                'config': config,
            }, save_path)
            print(f"  * New best! Saved to {save_path}")

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

    # ── Done ──────────────────────────────────────────────────────────────
    total_time = time.time() - train_start
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Best loss:   {best_loss:.4f}")
    print(f"  Total steps: {global_step}")
    print(f"  Checkpoint:  {ckpt_dir / ('v5_pretrain_best_' + args.scale + '.pt')}")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Pre-training")

    # Data
    parser.add_argument('--data_dir', nargs='+', default=None,
                        help='Directories of source files to train on')
    parser.add_argument('--max_files', type=int, default=50000,
                        help='Max files to load')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Training sequence length')

    # Model
    parser.add_argument('--scale', choices=['tiny', 'local', 'medium', '4b', '8b', 'runpod'],
                        default='tiny', help='Model scale')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
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

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: tiny model, project files, 3 epochs')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.scale = 'tiny'
        args.epochs = 3
        args.seq_len = 256
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.data_dir = [project_root]

    train(args)


if __name__ == "__main__":
    main()
