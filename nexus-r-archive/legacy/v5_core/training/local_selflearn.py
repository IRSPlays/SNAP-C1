"""
SNAP-C1 V5: Local Self-Learning Launcher (RX 7600 / 8GB VRAM)
===============================================================
One command to run the full self-learning pipeline on a consumer GPU.

Strategy for 8GB VRAM:
  - Use 'medium' config (332M params, ~1.3GB in fp32) or 'local' (65M)
  - Reference model lives on CPU (--ref_on_cpu)
  - Batch size = 1, seq_len = 256
  - Self-play generates DPO pairs → auto-DPO trains LoRA adapters
  - Everything fits in ~4-5GB, leaving headroom for activations

Workflow:
  1. Pre-train medium/local model on H200 (takes ~30 min)
  2. Instruct-tune on H200 (takes ~20 min)
  3. Copy checkpoint to local machine
  4. Run this script → self-learning runs 24/7 for free

Usage:
  # Full pipeline (self-play → DPO training loop):
  python local_selflearn.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt

  # Self-play only (generates DPO pairs, no training):
  python local_selflearn.py --checkpoint ... --play_only --pairs 200

  # DPO training only (from existing pairs):
  python local_selflearn.py --checkpoint ... --train_only

  # Interactive agent mode (use the model + learn from interactions):
  python local_selflearn.py --checkpoint ... --agent

  # Distill from 8B knowledge cache (NOVEL — student vs teacher):
  python local_selflearn.py --checkpoint ... --distill

  # Harvest on H200 (run this on the big GPU before distilling):
  python local_selflearn.py --checkpoint v5_core/checkpoints/v5_8b_instruct.pt --harvest
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def estimate_vram(config: dict) -> dict:
    """Estimate VRAM usage for a model config."""
    d = config.get('d_model', 512)
    n = config.get('n_blocks', 4)
    # Rough param count: 4 * d^2 per attention + 8/3 * d^2 per FFN, per block
    params_per_block = 4 * d * d + 3 * d * d  # ~7 * d^2
    total_params = n * params_per_block + 100279 * d  # + embedding
    fp32_gb = total_params * 4 / (1024 ** 3)
    return {
        'params': total_params,
        'params_m': total_params / 1e6,
        'fp32_gb': fp32_gb,
        'policy_only_gb': fp32_gb,
        'policy_and_ref_gb': fp32_gb * 2,
        'with_ref_on_cpu_gb': fp32_gb * 1.2,  # ~20% overhead for activations
    }


CONFIGS = {
    'local': {'d_model': 512, 'n_blocks': 4},
    'medium': {'d_model': 1024, 'n_blocks': 8},
    'runpod': {'d_model': 1536, 'n_blocks': 12},
    '4b': {'d_model': 2560, 'n_blocks': 28},
    '8b': {'d_model': 3072, 'n_blocks': 32},
}


def print_vram_table():
    """Show what fits on 8GB."""
    print("\n  VRAM Estimates (your GPU: RX 7600 — 8GB):")
    print("  " + "─" * 58)
    print(f"  {'Config':<10} {'Params':>10} {'Policy':>10} {'+ Ref GPU':>10} {'+ Ref CPU':>10}")
    print("  " + "─" * 58)
    for name, cfg in CONFIGS.items():
        est = estimate_vram(cfg)
        fits = "✓" if est['with_ref_on_cpu_gb'] < 7.5 else "✗"
        print(f"  {name:<10} {est['params_m']:>8.0f}M {est['policy_only_gb']:>8.1f}GB "
              f"{est['policy_and_ref_gb']:>8.1f}GB {est['with_ref_on_cpu_gb']:>8.1f}GB  {fits}")
    print("  " + "─" * 58)
    print("  ✓ = fits on 8GB with ref on CPU | ✗ = too large\n")


def run_pipeline(args):
    """Run the full self-learning pipeline."""

    print("=" * 60)
    print("  SNAP-C1 V5 — LOCAL SELF-LEARNING")
    print("  GPU: AMD RX 7600 (8GB) — Free forever")
    print("=" * 60)

    if args.vram_check:
        print_vram_table()
        return

    # Mode mutual exclusion
    modes = [args.play_only, args.train_only, args.agent,
             getattr(args, 'distill', False), getattr(args, 'harvest', False)]
    if sum(bool(m) for m in modes) > 1:
        print("ERROR: Only one mode flag allowed "
              "(--play_only, --train_only, --agent, --distill, --harvest)")
        return

    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("  Pre-train a model on H200 first, then copy it here.")
        print("  Recommended: 'medium' config (332M params, fits 8GB)")
        return

    dpo_buffer = args.dpo_buffer
    lora_path = args.lora or "v5_core/checkpoints/v5_lora_latest.pt"

    # ── Mode: Harvest (H200 only) ────────────────────────────────────────
    if args.harvest:
        print("\n[Mode] Knowledge Harvest (run on H200 with 8B)")
        print("  This generates the knowledge cache the local GPU trains against.")
        from v5_core.training.knowledge_harvester import main as harvest_main
        sys.argv = [
            'knowledge_harvester.py',
            '--checkpoint', checkpoint,
            '--output', args.knowledge_cache,
            '--tasks', str(args.harvest_tasks),
        ]
        harvest_main()
        return

    # ── Mode: Distill (LOCAL — student vs teacher) ───────────────────────
    if args.distill:
        if not os.path.exists(args.knowledge_cache):
            print(f"ERROR: Knowledge cache not found: {args.knowledge_cache}")
            print("  Run with --harvest on H200 first to generate it.")
            print("  Then copy knowledge_cache.jsonl to your local machine.")
            return
        print(f"\n[Mode] Progressive Distillation (Student vs Teacher)")
        from v5_core.training.progressive_distill import main as distill_main
        sys.argv = [
            'progressive_distill.py',
            '--student_checkpoint', checkpoint,
            '--knowledge_cache', args.knowledge_cache,
            '--dpo_output', dpo_buffer,
            '--scoreboard', args.scoreboard,
            '--rounds', str(args.distill_rounds),
            '--tasks_per_round', str(args.distill_tasks_per_round),
        ]
        if os.path.exists(lora_path):
            sys.argv += ['--lora', lora_path]
        if args.continuous:
            sys.argv.append('--continuous')
        distill_main()
        return

    # ── Mode: Agent ──────────────────────────────────────────────────────
    if args.agent:
        print("\n[Mode] Interactive Agent (learn from every interaction)")
        from v5_core.training.agent_loop import main as agent_main
        sys.argv = [
            'agent_loop.py',
            '--checkpoint', checkpoint,
            '--dpo_output', dpo_buffer,
            '--max_attempts', str(args.max_attempts),
            '--timeout', '10',
        ]
        if os.path.exists(lora_path):
            sys.argv += ['--lora', lora_path]
        agent_main()
        return

    # ── Mode: Train only ─────────────────────────────────────────────────
    if args.train_only:
        print(f"\n[Mode] DPO Training from {dpo_buffer}")
        from v5_core.training.auto_dpo_v5 import main as dpo_main
        sys.argv = [
            'auto_dpo_v5.py',
            '--base_checkpoint', checkpoint,
            '--dpo_buffer', dpo_buffer,
            '--output', lora_path,
            '--ref_on_cpu',              # Critical: saves 50% VRAM
            '--grad_checkpoint',         # Save more VRAM
            '--batch_size', '1',         # Tiny batch for 8GB
            '--seq_len', '256',          # Short sequences
            '--steps', str(args.dpo_steps),
            '--lr', '2e-5',
            '--save_every', '25',
            '--regression_check',
        ]
        if os.path.exists(lora_path):
            sys.argv += ['--resume_lora', lora_path]
        dpo_main()
        return

    # ── Mode: Play only ──────────────────────────────────────────────────
    if args.play_only:
        print(f"\n[Mode] Self-Play → {dpo_buffer}")
        from v5_core.training.self_play_coder import main as play_main
        sys.argv = [
            'self_play_coder.py',
            '--checkpoint', checkpoint,
            '--pairs', str(args.pairs),
            '--output', dpo_buffer,
            '--timeout', '10',
            '--start_difficulty', str(args.start_difficulty),
        ]
        if os.path.exists(lora_path):
            sys.argv += ['--lora', lora_path]
        play_main()
        return

    # ── Mode: Full Pipeline (default) ────────────────────────────────────
    print(f"\n[Mode] Full Pipeline (self-play → DPO → repeat)")
    print(f"  Checkpoint:  {checkpoint}")
    print(f"  DPO buffer:  {dpo_buffer}")
    print(f"  LoRA:        {lora_path}")
    print(f"  Pairs/round: {args.pairs}")
    print(f"  DPO steps:   {args.dpo_steps}")

    round_num = 0
    while True:
        round_num += 1
        print(f"\n{'=' * 60}")
        print(f"  ROUND {round_num}")
        print(f"{'=' * 60}")

        # Phase 1: Self-play (generate DPO pairs)
        print(f"\n[Phase 1] Self-play: generating {args.pairs} DPO pairs...")
        try:
            from v5_core.training.self_play_coder import main as play_main
            sys.argv = [
                'self_play_coder.py',
                '--checkpoint', checkpoint,
                '--pairs', str(args.pairs),
                '--output', dpo_buffer,
                '--timeout', '10',
                '--start_difficulty', str(min(round_num, 4)),
            ]
            if os.path.exists(lora_path):
                sys.argv += ['--lora', lora_path]
            play_main()
        except Exception as e:
            print(f"  [Self-play error] {e}")
            time.sleep(10)
            continue

        # Phase 2: DPO training on the collected pairs
        print(f"\n[Phase 2] DPO training ({args.dpo_steps} steps)...")
        try:
            from v5_core.training.auto_dpo_v5 import main as dpo_main
            sys.argv = [
                'auto_dpo_v5.py',
                '--base_checkpoint', checkpoint,
                '--dpo_buffer', dpo_buffer,
                '--ref_on_cpu',
                '--grad_checkpoint',
                '--batch_size', '1',
                '--seq_len', '256',
                '--steps', str(args.dpo_steps),
                '--lr', '2e-5',
                '--output', lora_path,
                '--save_every', '25',
                '--regression_check',
            ]
            if os.path.exists(lora_path):
                sys.argv += ['--resume_lora', lora_path]
            dpo_main()
        except Exception as e:
            print(f"  [DPO error] {e}")
            time.sleep(10)
            continue

        print(f"\n[Round {round_num}] Complete. LoRA saved to {lora_path}")
        print(f"  Buffer: {dpo_buffer}")

        if not args.continuous:
            print("\nDone. Run with --continuous for infinite self-learning.")
            break

        print(f"\n  Sleeping 30s before next round...")
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(
        description="SNAP-C1 V5 Local Self-Learning (8GB VRAM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check what fits on your GPU:
  python local_selflearn.py --vram_check

  # Full self-learning pipeline:
  python local_selflearn.py --checkpoint v5_core/checkpoints/v5_medium_instruct.pt

  # Continuous (runs forever):
  python local_selflearn.py --checkpoint ... --continuous

  # Interactive coding agent:
  python local_selflearn.py --checkpoint ... --agent

  # Distill from 8B (novel: student vs teacher tracked):
  python local_selflearn.py --checkpoint ... --distill --continuous

  # Harvest on H200 (generates knowledge cache):
  python local_selflearn.py --checkpoint v5_8b.pt --harvest --harvest_hours 5
        """,
    )

    parser.add_argument('--checkpoint', default='v5_core/checkpoints/v5_instruct_best.pt',
                        help='Path to base checkpoint')
    parser.add_argument('--lora', default=None,
                        help='LoRA checkpoint to resume from')
    parser.add_argument('--dpo_buffer', default='v5_core/data/dpo_buffer_v5.jsonl',
                        help='DPO pair output file')

    # Modes
    parser.add_argument('--play_only', action='store_true',
                        help='Only generate DPO pairs (no training)')
    parser.add_argument('--train_only', action='store_true',
                        help='Only train DPO on existing pairs')
    parser.add_argument('--agent', action='store_true',
                        help='Interactive coding agent mode')
    parser.add_argument('--distill', action='store_true',
                        help='Progressive distillation from 8B knowledge cache')
    parser.add_argument('--harvest', action='store_true',
                        help='Knowledge harvest (H200 only — generates cache)')
    parser.add_argument('--continuous', action='store_true',
                        help='Run self-play → DPO in an infinite loop')
    parser.add_argument('--vram_check', action='store_true',
                        help='Show VRAM estimates for each model size')

    # Self-play
    parser.add_argument('--pairs', type=int, default=32,
                        help='DPO pairs to generate per round')
    parser.add_argument('--start_difficulty', type=int, default=1,
                        help='Starting difficulty (1-4)')
    parser.add_argument('--max_attempts', type=int, default=3,
                        help='Max revision attempts in agent mode')

    # DPO training
    parser.add_argument('--dpo_steps', type=int, default=25,
                        help='DPO training steps per round')

    # Distillation
    parser.add_argument('--knowledge_cache',
                        default='v5_core/data/knowledge_cache.jsonl',
                        help='Path to 8B knowledge cache')
    parser.add_argument('--scoreboard',
                        default='v5_core/data/scoreboard.json',
                        help='Scoreboard save path')
    parser.add_argument('--distill_rounds', type=int, default=10,
                        help='Distillation rounds')
    parser.add_argument('--distill_tasks_per_round', type=int, default=50,
                        help='Tasks per distillation round')

    # Harvest
    parser.add_argument('--harvest_tasks', type=int, default=5000,
                        help='Max tasks to harvest on H200')

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
