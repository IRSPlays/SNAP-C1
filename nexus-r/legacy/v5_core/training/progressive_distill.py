"""
SNAP-C1 V5: Progressive Self-Distillation (Student vs Teacher)
================================================================
Novel training paradigm: the medium model (student) learns from the
8B model (teacher) through execution-grounded comparison.

Unlike standard distillation (match teacher's logits), this system:
  1. Loads the 8B's Knowledge Cache (verified solutions per task)
  2. Has the medium model attempt the SAME tasks
  3. Compares: if student fails where teacher succeeded → DPO pair
  4. Tracks per-task scoreboard: student vs teacher
  5. Focuses training on tasks where student < teacher (weakness map)
  6. Celebrates when student BEATS teacher (strength tracker)

Over weeks of continuous local training, the medium model approaches
then EXCEEDS the 8B on specific task categories — despite being 20x smaller.
The small model literally learns to be better than its teacher at the
things it practices most.

This has never been built before: execution-grounded progressive
distillation where the student can surpass the teacher, tracked
automatically.

Usage (on local GPU, runs 24/7):
  python progressive_distill.py \\
      --student_checkpoint v5_core/checkpoints/v5_medium_instruct.pt \\
      --knowledge_cache v5_core/data/knowledge_cache.jsonl \\
      --continuous

  # Single round:
  python progressive_distill.py \\
      --student_checkpoint v5_core/checkpoints/v5_medium_instruct.pt \\
      --knowledge_cache v5_core/data/knowledge_cache.jsonl \\
      --rounds 1 --tasks_per_round 50
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel
from v5_core.training.terminal_loop import TerminalRunner, DPOBufferWriter, ExecutionResult
from v5_core.training.self_play_coder import _strip_code_blocks


# ─── Scoreboard ──────────────────────────────────────────────────────────────

@dataclass
class TaskScore:
    """Tracks student vs teacher performance on a specific task."""
    task_id: str
    prompt: str
    difficulty: int
    teacher_passed: bool
    teacher_reward: float
    teacher_solution: str
    # Student history
    student_attempts: int = 0
    student_best_reward: float = -99.0
    student_best_code: str = ""
    student_ever_passed: bool = False
    # Comparison
    student_surpassed: bool = False  # Student reward > teacher reward
    last_attempt_time: float = 0.0
    test_code: str = ""  # Test code for execution verification

    def gap(self) -> float:
        """How far behind the teacher the student is."""
        return self.teacher_reward - self.student_best_reward

    def priority(self) -> float:
        """Higher priority = more valuable to train on.
        Focus on tasks where teacher succeeded but student hasn't yet.
        """
        if not self.teacher_passed:
            return 0.0  # Teacher failed too — not useful
        if self.student_surpassed:
            return 0.1  # Already surpassed — low priority
        if self.student_ever_passed:
            return 0.5  # Student can do it — medium priority
        # Student has never passed — HIGH priority
        recency_bonus = max(0, 1.0 - (time.time() - self.last_attempt_time) / 3600)
        return 2.0 + self.gap() + recency_bonus


class Scoreboard:
    """Tracks student vs teacher across all tasks."""

    def __init__(self, save_path: str = "v5_core/data/scoreboard.json"):
        self.scores: dict[str, TaskScore] = {}
        self.save_path = save_path
        self._load()

    def _load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            for entry in data.get('scores', []):
                ts = TaskScore(**entry)
                self.scores[ts.task_id] = ts
            print(f"[Scoreboard] Loaded {len(self.scores)} tasks")

    def save(self):
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'scores': [asdict(s) for s in self.scores.values()],
            'summary': self.summary(),
            'timestamp': time.time(),
        }
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def register_teacher(self, task_id: str, prompt: str, difficulty: int,
                         passed: bool, reward: float, solution: str):
        """Register a task from the knowledge cache."""
        if task_id not in self.scores:
            self.scores[task_id] = TaskScore(
                task_id=task_id, prompt=prompt, difficulty=difficulty,
                teacher_passed=passed, teacher_reward=reward,
                teacher_solution=solution,
            )

    def record_student(self, task_id: str, passed: bool, reward: float,
                       code: str):
        """Record a student attempt."""
        if task_id not in self.scores:
            return

        ts = self.scores[task_id]
        ts.student_attempts += 1
        ts.last_attempt_time = time.time()

        if reward > ts.student_best_reward:
            ts.student_best_reward = reward
            ts.student_best_code = code

        if passed:
            ts.student_ever_passed = True

        if reward > ts.teacher_reward:
            if not ts.student_surpassed:
                print(f"    *** SURPASSED TEACHER on {task_id}! "
                      f"Student={reward:.1f} > Teacher={ts.teacher_reward:.1f}")
            ts.student_surpassed = True

    def get_priority_tasks(self, n: int = 20) -> list[TaskScore]:
        """Get the N highest-priority tasks (where student needs most work)."""
        ranked = sorted(self.scores.values(), key=lambda s: s.priority(), reverse=True)
        return ranked[:n]

    def summary(self) -> dict:
        total = len(self.scores)
        if total == 0:
            return {'total_tasks': 0, 'teacher_passed': 0, 'student_passed': 0,
                    'surpassed_teacher': 0, 'never_tried': 0, 'avg_gap': 0.0,
                    'by_difficulty': {}}

        teacher_passed = sum(1 for s in self.scores.values() if s.teacher_passed)
        student_passed = sum(1 for s in self.scores.values() if s.student_ever_passed)
        surpassed = sum(1 for s in self.scores.values() if s.student_surpassed)
        never_tried = sum(1 for s in self.scores.values() if s.student_attempts == 0)
        avg_gap = sum(s.gap() for s in self.scores.values() if s.teacher_passed) / max(teacher_passed, 1)

        by_diff = defaultdict(lambda: {'teacher': 0, 'student': 0, 'surpassed': 0, 'total': 0})
        for s in self.scores.values():
            d = s.difficulty
            by_diff[d]['total'] += 1
            if s.teacher_passed:
                by_diff[d]['teacher'] += 1
            if s.student_ever_passed:
                by_diff[d]['student'] += 1
            if s.student_surpassed:
                by_diff[d]['surpassed'] += 1

        return {
            'total_tasks': total,
            'teacher_passed': teacher_passed,
            'student_passed': student_passed,
            'surpassed_teacher': surpassed,
            'never_tried': never_tried,
            'avg_gap': round(avg_gap, 2),
            'by_difficulty': dict(by_diff),
        }

    def print_report(self):
        s = self.summary()
        print(f"\n{'─' * 50}")
        print(f"  SCOREBOARD: Student vs Teacher (8B)")
        print(f"{'─' * 50}")
        print(f"  Total tasks:       {s['total_tasks']}")
        print(f"  Teacher passed:    {s['teacher_passed']}")
        print(f"  Student passed:    {s['student_passed']}")
        print(f"  Student SURPASSED: {s['surpassed_teacher']}")
        print(f"  Never tried:       {s['never_tried']}")
        print(f"  Avg gap:           {s['avg_gap']:.2f}")
        for d in sorted(s.get('by_difficulty', {}).keys()):
            info = s['by_difficulty'][d]
            print(f"    Diff {d}: teacher={info['teacher']}/{info['total']}, "
                  f"student={info['student']}/{info['total']}, "
                  f"surpassed={info['surpassed']}")
        print(f"{'─' * 50}")


# ─── Knowledge Cache Loader ─────────────────────────────────────────────────

def load_knowledge_cache(path: str) -> list[dict]:
    """Load the 8B's harvested knowledge."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"[KnowledgeCache] Loaded {len(entries)} entries from {path}")
    return entries


# ─── Progressive Distillation Engine ────────────────────────────────────────

def distillation_round(
    model: V5ResonanceModel,
    config: dict,
    device: torch.device,
    scoreboard: Scoreboard,
    runner: TerminalRunner,
    writer: DPOBufferWriter,
    tasks_per_round: int = 50,
) -> dict:
    """
    One round of progressive distillation:
    1. Pick high-priority tasks (where student < teacher)
    2. Student attempts each task
    3. Compare with teacher's cached solution
    4. Create DPO pairs from the comparison
    """
    from v5_core.inference.v5_generate import generate

    priority_tasks = scoreboard.get_priority_tasks(tasks_per_round)
    if not priority_tasks:
        print("  [Distill] No priority tasks — scoreboard empty?")
        return {'pairs_created': 0, 'teacher_wins': 0, 'student_wins': 0,
                'ties': 0, 'tasks_attempted': 0}

    pairs_created = 0
    student_wins = 0
    teacher_wins = 0
    ties = 0

    for i, task_score in enumerate(priority_tasks):
        # Student attempts the task
        try:
            student_code = generate(
                model=model, prompt=task_score.prompt,
                max_new_tokens=384,
                temperature=0.4 + random.random() * 0.3,
                top_p=0.92, top_k=50,
                device=device, config=config,
                phase='instruct',
            )
            student_code = _strip_code_blocks(student_code)
        except Exception as e:
            student_code = f"# Generation error: {e}"

        # Execute student's code
        if task_score.test_code:
            exec_result = runner.run(
                student_code + "\n" + task_score.test_code,
                expected_output="PASS",
            )
        else:
            exec_result = runner.run(student_code)

        student_reward = exec_result.reward
        student_passed = exec_result.exit_code == 0 and student_reward > 0

        # Record in scoreboard
        scoreboard.record_student(
            task_score.task_id,
            passed=student_passed,
            reward=student_reward,
            code=student_code,
        )

        # Create DPO pair based on comparison
        if task_score.teacher_passed and not student_passed:
            # Teacher succeeded, student failed → DPO pair
            writer.add_pair(
                prompt=task_score.prompt,
                chosen_code=task_score.teacher_solution,
                rejected_code=student_code,
                chosen_result=ExecutionResult(
                    code=task_score.teacher_solution,
                    exit_code=0, stdout='', stderr='',
                    duration_ms=0, reward=task_score.teacher_reward,
                ),
                rejected_result=exec_result,
                min_margin=0.5,
            )
            pairs_created += 1
            teacher_wins += 1
        elif student_passed and student_reward > task_score.teacher_reward:
            # Student surpassed teacher — no DPO needed, but track it!
            student_wins += 1
        elif student_passed and task_score.teacher_passed:
            # Both passed — student is learning but hasn't surpassed
            # Create a pair if there's a margin
            if task_score.teacher_reward - student_reward > 1.0:
                writer.add_pair(
                    prompt=task_score.prompt,
                    chosen_code=task_score.teacher_solution,
                    rejected_code=student_code,
                    chosen_result=ExecutionResult(
                        code=task_score.teacher_solution,
                        exit_code=0, stdout='', stderr='',
                        duration_ms=0, reward=task_score.teacher_reward,
                    ),
                    rejected_result=exec_result,
                    min_margin=0.5,
                )
                pairs_created += 1
                teacher_wins += 1
            else:
                ties += 1
        else:
            ties += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(priority_tasks)}] "
                  f"pairs={pairs_created} teacher_wins={teacher_wins} "
                  f"student_wins={student_wins} ties={ties}")

    writer.save()
    scoreboard.save()

    return {
        'pairs_created': pairs_created,
        'teacher_wins': teacher_wins,
        'student_wins': student_wins,
        'ties': ties,
        'tasks_attempted': len(priority_tasks),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_progressive_distill(args):
    """Main progressive distillation loop."""
    print("=" * 60)
    print("  SNAP-C1 V5 PROGRESSIVE SELF-DISTILLATION")
    print("  Student (medium) learning from Teacher (8B)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load student model
    from v5_core.inference.v5_generate import load_model
    model, config, phase = load_model(args.student_checkpoint, device)

    # Load LoRA if available
    if args.lora and os.path.exists(args.lora):
        from v5_core.training.auto_dpo_v5 import inject_lora, load_lora
        lora_modules = inject_lora(model, rank=16, alpha=32.0)
        load_lora(lora_modules, args.lora)
        model._lora_injected = True
        model._lora_modules = lora_modules

    # Load knowledge cache
    knowledge = load_knowledge_cache(args.knowledge_cache)

    # Initialize scoreboard
    scoreboard = Scoreboard(save_path=args.scoreboard)

    # Register teacher solutions
    for entry in knowledge:
        scoreboard.register_teacher(
            task_id=entry['task_id'],
            prompt=entry['prompt'],
            difficulty=entry.get('difficulty', 2),
            passed=entry['passed'],
            reward=entry['reward'],
            solution=entry['solution'],
        )

    # Add test_code to TaskScore objects (not stored in scoreboard, add from cache)
    cache_lookup = {e['task_id']: e for e in knowledge}
    for task_id, score in scoreboard.scores.items():
        if task_id in cache_lookup:
            score.test_code = cache_lookup[task_id].get('test_code', '')

    # Setup
    runner = TerminalRunner(timeout_sec=args.timeout)
    writer = DPOBufferWriter(args.dpo_output)

    print(f"\n  Knowledge cache:  {len(knowledge)} entries")
    print(f"  Scoreboard:       {len(scoreboard.scores)} tasks")
    print(f"  DPO output:       {args.dpo_output}")
    scoreboard.print_report()

    # Distillation loop
    round_num = 0
    while True:
        round_num += 1
        print(f"\n{'=' * 60}")
        print(f"  ROUND {round_num}")
        print(f"{'=' * 60}")

        metrics = distillation_round(
            model=model, config=config, device=device,
            scoreboard=scoreboard,
            runner=runner, writer=writer,
            tasks_per_round=args.tasks_per_round,
        )

        print(f"\n  Round {round_num} results:")
        print(f"    DPO pairs created: {metrics['pairs_created']}")
        print(f"    Teacher wins:      {metrics['teacher_wins']}")
        print(f"    Student wins:      {metrics['student_wins']}")
        print(f"    Ties:              {metrics['ties']}")

        scoreboard.print_report()

        # Run DPO training after enough pairs
        buffer_size = len(writer.pairs)
        if buffer_size >= args.train_threshold and not args.no_train:
            print(f"\n  [Auto-DPO] {buffer_size} pairs in buffer, training...")
            try:
                from v5_core.training.auto_dpo_v5 import (
                    inject_lora, load_lora, save_lora,
                    DPODataset, compute_logprobs, dpo_loss, train_dpo,
                )
                import types

                dpo_args = types.SimpleNamespace(
                    base_checkpoint=args.student_checkpoint,
                    dpo_buffer=args.dpo_output,
                    output=args.lora or 'v5_core/checkpoints/v5_lora_latest.pt',
                    resume_lora=args.lora if args.lora and os.path.exists(args.lora) else None,
                    lora_rank=16,
                    lora_alpha=32.0,
                    steps=min(buffer_size // 2, 25),
                    batch_size=1,
                    lr=2e-5,
                    beta=0.1,
                    seq_len=256,
                    kl_threshold=0.3,
                    kl_check_every=10,
                    regression_check=False,
                    save_every=50,
                    ref_on_cpu=True,
                    grad_checkpoint=True,
                )
                train_dpo(dpo_args)
                # Reload LoRA into active model so next round uses updated weights
                lora_out = dpo_args.output
                if os.path.exists(lora_out):
                    try:
                        if not hasattr(model, '_lora_injected'):
                            model._lora_modules = inject_lora(model, rank=16, alpha=32.0)
                            model._lora_injected = True
                        load_lora(model._lora_modules, lora_out)  # reload updated weights
                    except Exception as le:
                        print(f"  [Auto-DPO] LoRA reload warning: {le}")
            except Exception as e:
                print(f"  [Auto-DPO] Training error: {e}")

        if not args.continuous and round_num >= args.rounds:
            break

        if args.continuous:
            print(f"\n  Sleeping {args.sleep_between}s before next round...")
            time.sleep(args.sleep_between)

    # Final report
    print(f"\n{'=' * 60}")
    print(f"  PROGRESSIVE DISTILLATION COMPLETE")
    print(f"{'=' * 60}")
    scoreboard.print_report()
    print(f"  DPO pairs: {writer.stats()}")
    print(f"  Scoreboard saved to: {args.scoreboard}")


def main():
    parser = argparse.ArgumentParser(
        description="SNAP-C1 V5 Progressive Self-Distillation",
    )

    parser.add_argument('--student_checkpoint', required=True,
                        help='Path to student (medium) model checkpoint')
    parser.add_argument('--knowledge_cache',
                        default='v5_core/data/knowledge_cache.jsonl',
                        help='Path to 8B knowledge cache (from harvester)')
    parser.add_argument('--lora', default=None,
                        help='LoRA checkpoint to load/save')
    parser.add_argument('--dpo_output',
                        default='v5_core/data/dpo_buffer_v5.jsonl',
                        help='DPO buffer output')
    parser.add_argument('--scoreboard',
                        default='v5_core/data/scoreboard.json',
                        help='Scoreboard save path')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Sandbox timeout (seconds)')

    # Round control
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of distillation rounds')
    parser.add_argument('--tasks_per_round', type=int, default=50,
                        help='Tasks to attempt per round')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously (infinite rounds)')
    parser.add_argument('--sleep_between', type=int, default=30,
                        help='Seconds between rounds in continuous mode')

    # Training
    parser.add_argument('--train_threshold', type=int, default=32,
                        help='Min DPO pairs before auto-training')
    parser.add_argument('--no_train', action='store_true',
                        help='Skip auto-DPO training (pairs only)')

    args = parser.parse_args()
    run_progressive_distill(args)


if __name__ == "__main__":
    main()
