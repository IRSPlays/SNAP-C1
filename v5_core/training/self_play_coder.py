"""
SNAP-C1 V5: Self-Play Coder (Execution-Grounded Self-Play)
============================================================
The model generates TWO competing solutions to the same coding task,
runs both in a sandbox, the winner = chosen, loser = rejected,
and the pair feeds into auto_dpo_v5.py.

This is the autonomous training data generator. It runs 24/7 on the
H200, producing DPO pairs without any human involvement.

Novel features:
  1. Automatic curriculum: tracks per-difficulty success rate.
     When success > 80%, escalate. When < 30%, drop down.
  2. Failure memory: tasks the model failed get replayed with priority.
  3. Task mining: extracts coding tasks from the training corpus itself
     (function signatures → "implement this function").
  4. Diversity enforcement: temperature varies per attempt to avoid
     generating identical solutions.

Usage:
  # Generate 100 self-play DPO pairs:
  python self_play_coder.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --pairs 100 --output v5_core/data/dpo_buffer_v5.jsonl

  # Continuous self-play (daemon mode):
  python self_play_coder.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --continuous --pairs_per_round 32
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel
from v5_core.training.terminal_loop import TerminalRunner, DPOBufferWriter, ExecutionResult


# ──────────────────────────────────────────────────────────────────────────────
# Task Bank: Synthetic coding tasks at increasing difficulty
# ──────────────────────────────────────────────────────────────────────────────

TASKS_BY_DIFFICULTY = {
    1: [  # Basic — string/list/math
        {
            'prompt': 'Write a Python function `reverse_string(s)` that returns the reverse of a string.',
            'test': 'assert reverse_string("hello") == "olleh"\nassert reverse_string("") == ""\nassert reverse_string("a") == "a"\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `sum_list(lst)` that returns the sum of all numbers in a list.',
            'test': 'assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `is_even(n)` that returns True if n is even, False otherwise.',
            'test': 'assert is_even(2) == True\nassert is_even(3) == False\nassert is_even(0) == True\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `max_of_three(a, b, c)` that returns the largest of three numbers.',
            'test': 'assert max_of_three(1, 2, 3) == 3\nassert max_of_three(5, 1, 3) == 5\nassert max_of_three(-1, -2, -3) == -1\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `count_vowels(s)` that returns the number of vowels in a string.',
            'test': 'assert count_vowels("hello") == 2\nassert count_vowels("xyz") == 0\nassert count_vowels("aeiou") == 5\nprint("PASS")',
        },
    ],
    2: [  # Intermediate — algorithms, data structures
        {
            'prompt': 'Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.',
            'test': 'assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\nassert fibonacci(20) == 6765\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `flatten(lst)` that flattens a nested list of arbitrary depth.',
            'test': 'assert flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nassert flatten([[1], [[2]], [[[3]]]]) == [1, 2, 3]\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted array, or -1 if not found.',
            'test': 'assert binary_search([1, 3, 5, 7, 9], 5) == 2\nassert binary_search([1, 3, 5, 7, 9], 4) == -1\nassert binary_search([], 1) == -1\nassert binary_search([1], 1) == 0\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `remove_duplicates(lst)` that returns a new list with duplicates removed, preserving order.',
            'test': 'assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1, 1, 1]) == [1]\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.',
            'test': 'assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]\nassert merge_sorted([1], []) == [1]\nprint("PASS")',
        },
    ],
    3: [  # Advanced — classes, error handling, complex logic
        {
            'prompt': 'Write a Python class `Stack` with methods push(val), pop(), peek(), and is_empty(). pop() and peek() should raise IndexError if empty.',
            'test': 's = Stack()\nassert s.is_empty() == True\ns.push(1); s.push(2)\nassert s.peek() == 2\nassert s.pop() == 2\nassert s.pop() == 1\nassert s.is_empty() == True\ntry:\n    s.pop()\n    assert False\nexcept IndexError:\n    pass\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `group_anagrams(words)` that groups a list of words into lists of anagrams. Return a list of lists, sorted by first element.',
            'test': 'result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])\nresult = [sorted(g) for g in result]\nresult.sort()\nassert result == [[\"ate\", \"eat\", \"tea\"], [\"bat\"], [\"nat\", \"tan\"]]\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `lru_cache_manual(capacity)` that returns get(key) and put(key, value) functions implementing an LRU cache.',
            'test': 'get, put = lru_cache_manual(2)\nput(1, 1); put(2, 2)\nassert get(1) == 1\nput(3, 3)  # evicts key 2\nassert get(2) == -1\nassert get(3) == 3\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `validate_parentheses(s)` that returns True if the string has valid bracket matching for (), [], {}.',
            'test': 'assert validate_parentheses("()[]{}") == True\nassert validate_parentheses("([)]") == False\nassert validate_parentheses("{[]}") == True\nassert validate_parentheses("(") == False\nassert validate_parentheses("") == True\nprint("PASS")',
        },
    ],
    4: [  # Hard — multi-step, recursion, edge cases
        {
            'prompt': 'Write a Python function `eval_rpn(tokens)` that evaluates a Reverse Polish Notation expression. tokens is a list of strings (numbers or "+", "-", "*", "/"). Division truncates toward zero.',
            'test': 'assert eval_rpn(["2", "1", "+", "3", "*"]) == 9\nassert eval_rpn(["4", "13", "5", "/", "+"]) == 6\nassert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `serialize(root)` and `deserialize(s)` for a binary tree. Use a simple Node class with val, left, right.',
            'test': 'class Node:\n    def __init__(self, val=0, left=None, right=None):\n        self.val, self.left, self.right = val, left, right\nroot = Node(1, Node(2), Node(3, Node(4), Node(5)))\ns = serialize(root)\nr = deserialize(s)\nassert r.val == 1\nassert r.left.val == 2\nassert r.right.right.val == 5\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `longest_common_subsequence(s1, s2)` that returns the length of the LCS.',
            'test': 'assert longest_common_subsequence("abcde", "ace") == 3\nassert longest_common_subsequence("abc", "def") == 0\nassert longest_common_subsequence("", "abc") == 0\nassert longest_common_subsequence("abcba", "abcbcba") == 5\nprint("PASS")',
        },
    ],
}


@dataclass
class CurriculumState:
    """Tracks success rate per difficulty level for automatic escalation."""
    difficulty: int = 1
    attempts: dict = None  # difficulty -> [success_count, total_count]
    failure_queue: list = None  # Tasks to retry

    def __post_init__(self):
        if self.attempts is None:
            self.attempts = {d: [0, 0] for d in TASKS_BY_DIFFICULTY}
        if self.failure_queue is None:
            self.failure_queue = []

    def record(self, difficulty: int, success: bool, task: dict = None):
        if difficulty not in self.attempts:
            self.attempts[difficulty] = [0, 0]
        self.attempts[difficulty][1] += 1
        if success:
            self.attempts[difficulty][0] += 1
        elif task is not None:
            # Add to failure queue for retry (max 50 stored)
            if len(self.failure_queue) < 50:
                self.failure_queue.append({**task, 'difficulty': difficulty})

    def success_rate(self, difficulty: int) -> float:
        s, t = self.attempts.get(difficulty, [0, 0])
        return s / max(t, 1)

    def should_escalate(self) -> bool:
        return (self.success_rate(self.difficulty) > 0.8
                and self.attempts[self.difficulty][1] >= 5
                and self.difficulty < max(TASKS_BY_DIFFICULTY.keys()))

    def should_deescalate(self) -> bool:
        return (self.success_rate(self.difficulty) < 0.3
                and self.attempts[self.difficulty][1] >= 5
                and self.difficulty > 1)

    def update_difficulty(self):
        if self.should_escalate():
            self.difficulty = min(self.difficulty + 1, max(TASKS_BY_DIFFICULTY.keys()))
            print(f"  [Curriculum] Escalating to difficulty {self.difficulty} "
                  f"(success rate was {self.success_rate(self.difficulty - 1):.0%})")
        elif self.should_deescalate():
            self.difficulty = max(self.difficulty - 1, 1)
            print(f"  [Curriculum] De-escalating to difficulty {self.difficulty}")

    def get_next_task(self) -> tuple[dict, int]:
        """Get the next task to attempt. Occasionally replays failures."""
        # 30% chance to replay a failure
        if self.failure_queue and random.random() < 0.3:
            task = self.failure_queue.pop(0)
            return task, task.get('difficulty', self.difficulty)

        # Otherwise, pick from current difficulty
        tasks = TASKS_BY_DIFFICULTY.get(self.difficulty, TASKS_BY_DIFFICULTY[1])
        return random.choice(tasks), self.difficulty

    def to_dict(self) -> dict:
        return {
            'difficulty': self.difficulty,
            'attempts': {str(k): v for k, v in self.attempts.items()},
            'failure_queue_size': len(self.failure_queue),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Task Mining: extract tasks from real code
# ──────────────────────────────────────────────────────────────────────────────

def mine_tasks_from_code(code_dir: str, max_tasks: int = 50) -> list[dict]:
    """
    Extract coding tasks from real Python files by finding function
    signatures and turning them into "implement this function" prompts.

    This provides infinite diverse tasks beyond the static task bank.
    """
    import ast as _ast

    tasks = []
    py_files = list(Path(code_dir).rglob("*.py"))
    random.shuffle(py_files)

    for pyf in py_files[:200]:  # Cap files scanned
        try:
            source = pyf.read_text(encoding='utf-8', errors='ignore')
            tree = _ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and len(tasks) < max_tasks:
                # Skip private/test/short functions
                if node.name.startswith('_') or node.name.startswith('test'):
                    continue
                if len(node.body) < 3:
                    continue

                # Extract signature + docstring
                args = [a.arg for a in node.args.args if a.arg != 'self']
                if not args:
                    continue

                docstring = _ast.get_docstring(node) or ""
                if not docstring:
                    continue

                sig = f"def {node.name}({', '.join(args)}):"
                prompt = f"Write a Python function:\n```python\n{sig}\n    \"\"\"{docstring}\"\"\"\n```\nImplement the function body."

                # We can't auto-generate tests for mined tasks, so we
                # use syntax-only validation (code must be valid Python + no crash)
                tasks.append({
                    'prompt': prompt,
                    'test': None,  # Will use syntax + no-crash check
                    'mined_from': str(pyf.name),
                    'func_name': node.name,
                })

    print(f"[TaskMiner] Mined {len(tasks)} tasks from {code_dir}")
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Self-Play Engine
# ──────────────────────────────────────────────────────────────────────────────

def self_play_round(
    model: V5ResonanceModel,
    config: dict,
    device: torch.device,
    task: dict,
    runner: TerminalRunner,
    attempt_temperatures: tuple = (0.4, 0.9),
) -> Optional[dict]:
    """
    Generate two competing solutions at different temperatures.
    Run both. The one with higher reward = chosen, lower = rejected.

    Returns a DPO pair dict, or None if no valid pair.
    """
    from v5_core.inference.v5_generate import generate

    prompt = task['prompt']
    test_code = task.get('test')
    results = []

    for temp in attempt_temperatures:
        try:
            code = generate(
                model=model, prompt=prompt,
                max_new_tokens=384,
                temperature=temp,
                top_p=0.95, top_k=50,
                device=device, config=config,
                phase='instruct',
            )

            # Strip markdown code blocks if present
            code = _strip_code_blocks(code)

            if test_code:
                # Run with test
                exec_result = runner.run(
                    code + "\n" + test_code,
                    expected_output="PASS",
                )
            else:
                # Syntax + no-crash check only (mined tasks)
                exec_result = runner.run(code)

            results.append((code, exec_result, temp))

        except Exception as e:
            results.append((f"# Error: {e}", ExecutionResult(
                code="", exit_code=-1, stdout="", stderr=str(e),
                duration_ms=0, timed_out=False, reward=-5.0
            ), temp))

    if len(results) < 2:
        return None

    # Sort by reward (highest first)
    results.sort(key=lambda r: r[1].reward, reverse=True)
    chosen_code, chosen_result, chosen_temp = results[0]
    rejected_code, rejected_result, rejected_temp = results[1]

    # Only create pair if margin is significant
    margin = chosen_result.reward - rejected_result.reward
    if margin < 0.5:
        return None

    return {
        'prompt': prompt,
        'chosen': chosen_code,
        'rejected': rejected_code,
        'chosen_reward': chosen_result.reward,
        'rejected_reward': rejected_result.reward,
        'metadata': {
            'chosen_temp': chosen_temp,
            'rejected_temp': rejected_temp,
            'chosen_exit': chosen_result.exit_code,
            'rejected_exit': rejected_result.exit_code,
            'margin': margin,
            'timestamp': time.time(),
        },
    }


def _strip_code_blocks(text: str) -> str:
    """Remove markdown ```python ... ``` wrapping if present."""
    lines = text.strip().split('\n')
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_self_play(args):
    """Main self-play loop."""
    print("=" * 60)
    print("SNAP-C1 V5 SELF-PLAY CODER")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load model
    from v5_core.inference.v5_generate import load_model
    model, config, phase = load_model(args.checkpoint, device)

    # Optionally load LoRA
    if args.lora:
        from v5_core.training.auto_dpo_v5 import inject_lora, load_lora
        lora_modules = inject_lora(model, rank=16, alpha=32.0)
        load_lora(lora_modules, args.lora)

    # Runner + writer
    runner = TerminalRunner(timeout_sec=args.timeout)
    writer = DPOBufferWriter(args.output)

    # Curriculum
    curriculum = CurriculumState(difficulty=args.start_difficulty)

    # Mine additional tasks from training data (if available)
    mined_tasks = []
    if args.mine_from and os.path.isdir(args.mine_from):
        mined_tasks = mine_tasks_from_code(args.mine_from, max_tasks=100)

    # ── Self-play loop ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Self-play: generating {args.pairs} DPO pairs")
    print(f"Starting difficulty: {curriculum.difficulty}")
    print(f"Output: {args.output}")
    print(f"{'=' * 60}\n")

    generated = 0
    attempted = 0
    start_time = time.time()

    while generated < args.pairs:
        # Pick a task
        use_mined = mined_tasks and random.random() < 0.3
        if use_mined:
            task = random.choice(mined_tasks)
            difficulty = 2  # Mined tasks are medium difficulty
        else:
            task, difficulty = curriculum.get_next_task()

        attempted += 1

        # Self-play round
        pair = self_play_round(
            model=model, config=config, device=device,
            task=task, runner=runner,
            attempt_temperatures=(0.3 + random.random() * 0.2,
                                   0.7 + random.random() * 0.3),
        )

        if pair is not None:
            # Write to buffer
            writer.add_pair(
                prompt=pair['prompt'],
                chosen_code=pair['chosen'],
                rejected_code=pair['rejected'],
                chosen_result=ExecutionResult(
                    code=pair['chosen'], exit_code=0, stdout='', stderr='',
                    duration_ms=0, reward=pair['chosen_reward'],
                ),
                rejected_result=ExecutionResult(
                    code=pair['rejected'], exit_code=1, stdout='', stderr='',
                    duration_ms=0, reward=pair['rejected_reward'],
                ),
                min_margin=0.5,
            )
            generated += 1
            curriculum.record(difficulty, success=True)
            status = "OK"
        else:
            curriculum.record(difficulty, success=False, task=task)
            status = "SKIP (no margin)"

        # Update curriculum
        curriculum.update_difficulty()

        # Logging
        elapsed = time.time() - start_time
        rate = generated / elapsed * 3600 if elapsed > 0 else 0
        if attempted % 5 == 0 or generated == 1:
            print(f"  [{generated}/{args.pairs}] attempt={attempted} | "
                  f"diff={difficulty} | {status} | "
                  f"rate={rate:.0f} pairs/hr | "
                  f"success={curriculum.success_rate(difficulty):.0%}")

        # Periodic save
        if generated % 10 == 0 and generated > 0:
            writer.save()

    # Final save
    writer.save()

    # Save curriculum state
    curriculum_path = Path(args.output).parent / "curriculum_state.json"
    with open(curriculum_path, 'w') as f:
        json.dump(curriculum.to_dict(), f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SELF-PLAY COMPLETE")
    print(f"  Generated:  {generated} pairs from {attempted} attempts")
    print(f"  Time:       {elapsed / 60:.1f} minutes")
    print(f"  Rate:       {generated / elapsed * 3600:.0f} pairs/hour")
    print(f"  Final diff: {curriculum.difficulty}")
    print(f"  Output:     {args.output}")
    print(f"  Curriculum: {curriculum_path}")
    print(f"  Buffer stats: {json.dumps(writer.stats(), indent=2)}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Self-Play Coder")

    parser.add_argument('--checkpoint', required=True,
                        help='Path to V5 checkpoint (.pt)')
    parser.add_argument('--lora', default=None,
                        help='Optional LoRA checkpoint to load on top')
    parser.add_argument('--pairs', type=int, default=100,
                        help='Number of DPO pairs to generate')
    parser.add_argument('--output', default='v5_core/data/dpo_buffer_v5.jsonl',
                        help='Output JSONL file for DPO pairs')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Sandbox timeout per execution (seconds)')
    parser.add_argument('--start_difficulty', type=int, default=1,
                        help='Starting difficulty level (1-4)')
    parser.add_argument('--mine_from', default=None,
                        help='Directory of Python files to mine tasks from')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously (restart after each round)')
    parser.add_argument('--pairs_per_round', type=int, default=32,
                        help='Pairs per round in continuous mode')

    args = parser.parse_args()

    if args.continuous:
        round_num = 0
        while True:
            round_num += 1
            print(f"\n[Continuous] Round {round_num}")
            args.pairs = args.pairs_per_round
            try:
                run_self_play(args)
            except KeyboardInterrupt:
                print("\n[Continuous] Stopped.")
                break
            except Exception as e:
                print(f"[Continuous] Error in round {round_num}: {e}")
                time.sleep(30)
    else:
        run_self_play(args)


if __name__ == "__main__":
    main()
