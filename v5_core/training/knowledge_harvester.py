"""
SNAP-C1 V5: Knowledge Harvester (8B → Local Knowledge Transfer)
================================================================
Runs on H200 during the LAST 5-8 hours of compute time.

The 8B model speed-runs the entire coding curriculum, generating
verified solutions at every difficulty level. Each solution is
sandbox-tested and scored. The output is a "knowledge cache" —
a massive, execution-verified dataset that captures what the 8B
model KNOWS, crystallized into tested code.

This is NOT just distillation. Each entry has:
  - The prompt
  - The 8B's solution (verified working)
  - Multiple attempts at different temperatures
  - Test results (pass/fail, reward, timing)
  - The 8B's token-level log probabilities (teacher signal)

The local machine then uses this to train the medium model:
  - Medium tries the same task → if it fails but 8B succeeded,
    instant DPO pair (8B = chosen, medium = rejected)
  - Medium approaches 8B quality but 20x smaller
  - When medium beats 8B on a task → surpassed the teacher

Usage (on H200, last few hours):
  python knowledge_harvester.py --checkpoint v5_core/checkpoints/v5_8b_instruct.pt \\
      --output v5_core/data/knowledge_cache.jsonl \\
      --tasks 5000 --include_logprobs

  # Quick harvest (for testing):
  python knowledge_harvester.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --output v5_core/data/knowledge_cache.jsonl --tasks 100
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel
from v5_core.training.terminal_loop import TerminalRunner, ExecutionResult
from v5_core.training.self_play_coder import (
    TASKS_BY_DIFFICULTY, mine_tasks_from_code, _strip_code_blocks,
)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENC = None


# ─── Extended Task Bank ──────────────────────────────────────────────────────
# Additional tasks beyond self_play_coder's bank, focused on real-world patterns

HARVEST_EXTRA_TASKS = {
    2: [
        {
            'prompt': 'Write a Python function `word_count(text)` that returns a dict mapping each word (lowercased) to its count.',
            'test': 'result = word_count("the cat sat on the mat")\nassert result["the"] == 2\nassert result["cat"] == 1\nassert result["mat"] == 1\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `matrix_multiply(A, B)` that multiplies two 2D lists (matrices).',
            'test': 'assert matrix_multiply([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]\nassert matrix_multiply([[1]], [[2]]) == [[2]]\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `caesar_cipher(text, shift)` that encrypts text with a Caesar cipher. Only shift letters, preserve case.',
            'test': 'assert caesar_cipher("Hello", 3) == "Khoor"\nassert caesar_cipher("xyz", 3) == "abc"\nassert caesar_cipher("Test 123!", 1) == "Uftu 123!"\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `chunk_list(lst, n)` that splits a list into chunks of size n.',
            'test': 'assert chunk_list([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\nassert chunk_list([1,2,3], 3) == [[1,2,3]]\nassert chunk_list([], 5) == []\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `deep_copy(obj)` that deep-copies a nested dict/list structure without using copy module.',
            'test': 'original = {"a": [1, {"b": 2}]}\nresult = deep_copy(original)\nresult["a"][1]["b"] = 99\nassert original["a"][1]["b"] == 2\nprint("PASS")',
        },
    ],
    3: [
        {
            'prompt': 'Write a Python function `json_flatten(nested, prefix="")` that flattens nested dicts with dot-separated keys.',
            'test': 'assert json_flatten({"a": {"b": 1, "c": {"d": 2}}}) == {"a.b": 1, "a.c.d": 2}\nassert json_flatten({"x": 1}) == {"x": 1}\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `topological_sort(graph)` where graph is a dict {node: [dependencies]}. Return a valid ordering or raise ValueError for cycles.',
            'test': 'result = topological_sort({"a": ["b", "c"], "b": ["c"], "c": []})\nassert result.index("c") < result.index("b") < result.index("a")\ntry:\n    topological_sort({"a": ["b"], "b": ["a"]})\n    assert False\nexcept ValueError:\n    pass\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `rate_limiter(max_calls, period_sec)` that returns a decorator which limits function calls. Raise RuntimeError if limit exceeded.',
            'test': 'import time\n@rate_limiter(2, 1.0)\ndef test_fn(): return "ok"\nassert test_fn() == "ok"\nassert test_fn() == "ok"\ntry:\n    test_fn()\n    assert False\nexcept RuntimeError:\n    pass\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python class `MinHeap` with methods insert(val), extract_min(), peek(), and size().',
            'test': 'h = MinHeap()\nh.insert(5); h.insert(3); h.insert(7); h.insert(1)\nassert h.peek() == 1\nassert h.extract_min() == 1\nassert h.extract_min() == 3\nassert h.size() == 2\nprint("PASS")',
        },
    ],
    4: [
        {
            'prompt': 'Write a Python function `regex_match(pattern, text)` that supports "." (any char) and "*" (zero or more of previous). Return True/False for full match.',
            'test': 'assert regex_match("a.b", "acb") == True\nassert regex_match("a*", "aaa") == True\nassert regex_match("a*", "") == True\nassert regex_match("a*b", "aaab") == True\nassert regex_match("a*b", "aaac") == False\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `find_median_sorted_arrays(nums1, nums2)` that finds the median of two sorted arrays in O(log(min(m,n))) time.',
            'test': 'assert find_median_sorted_arrays([1,3], [2]) == 2.0\nassert find_median_sorted_arrays([1,2], [3,4]) == 2.5\nassert find_median_sorted_arrays([], [1]) == 1.0\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `interval_merge(intervals)` that merges overlapping intervals. Input: list of [start, end], Output: merged list.',
            'test': 'assert interval_merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert interval_merge([[1,4],[4,5]]) == [[1,5]]\nassert interval_merge([]) == []\nprint("PASS")',
        },
        {
            'prompt': 'Write a Python function `word_break(s, word_dict)` that returns True if string s can be segmented into space-separated dictionary words.',
            'test': 'assert word_break("leetcode", {"leet", "code"}) == True\nassert word_break("applepenapple", {"apple", "pen"}) == True\nassert word_break("catsandog", {"cats", "dog", "sand", "and", "cat"}) == False\nprint("PASS")',
        },
    ],
}


@dataclass
class HarvestEntry:
    """One entry in the knowledge cache."""
    task_id: str
    prompt: str
    test_code: str
    difficulty: int
    # 8B model's solution
    solution: str
    passed: bool
    reward: float
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    # Multiple attempts for diversity
    all_attempts: list = field(default_factory=list)
    # Teacher log probabilities (optional, for KD loss)
    teacher_logprobs: Optional[list] = None
    # Metadata
    temperature: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Skip logprobs in JSON (too large, saved separately)
        d.pop('teacher_logprobs', None)
        return d


def _task_id(prompt: str) -> str:
    """Generate a stable ID for a task."""
    return hashlib.md5(prompt.encode()).hexdigest()[:12]


def build_task_list(include_mined: bool = False, mine_dir: str = None,
                    max_mined: int = 100) -> list[dict]:
    """Build the full task list from all sources."""
    tasks = []

    # Static tasks from self_play_coder
    for diff, task_list in TASKS_BY_DIFFICULTY.items():
        for task in task_list:
            tasks.append({**task, 'difficulty': diff})

    # Extra harvest tasks
    for diff, task_list in HARVEST_EXTRA_TASKS.items():
        for task in task_list:
            tasks.append({**task, 'difficulty': diff})

    # Mined tasks
    if include_mined and mine_dir and os.path.isdir(mine_dir):
        mined = mine_tasks_from_code(mine_dir, max_tasks=max_mined)
        for task in mined:
            tasks.append({**task, 'difficulty': 2})

    # Deduplicate by prompt hash
    seen = set()
    unique = []
    for task in tasks:
        tid = _task_id(task['prompt'])
        if tid not in seen:
            seen.add(tid)
            unique.append(task)

    return unique


# ─── Harvester Engine ────────────────────────────────────────────────────────

def harvest_single_task(
    model: V5ResonanceModel,
    config: dict,
    device: torch.device,
    task: dict,
    runner: TerminalRunner,
    temperatures: tuple = (0.2, 0.5, 0.8),
    include_logprobs: bool = False,
) -> HarvestEntry:
    """
    Run the 8B model on a single task at multiple temperatures.
    Pick the best solution, record all attempts.
    """
    from v5_core.inference.v5_generate import generate

    prompt = task['prompt']
    test_code = task.get('test', '')
    difficulty = task.get('difficulty', 2)

    attempts = []
    best_attempt = None

    for temp in temperatures:
        try:
            code = generate(
                model=model, prompt=prompt,
                max_new_tokens=512,
                temperature=temp,
                top_p=0.95, top_k=50,
                device=device, config=config,
                phase='instruct',
            )
            code = _strip_code_blocks(code)

            # Execute
            if test_code:
                exec_result = runner.run(
                    code + "\n" + test_code,
                    expected_output="PASS",
                )
            else:
                exec_result = runner.run(code)

            attempt = {
                'temperature': temp,
                'code': code,
                'passed': exec_result.exit_code == 0 and exec_result.reward > 0,
                'reward': exec_result.reward,
                'exit_code': exec_result.exit_code,
            }
            attempts.append(attempt)

            if best_attempt is None or exec_result.reward > best_attempt['reward']:
                best_attempt = {
                    **attempt,
                    'stdout': exec_result.stdout,
                    'stderr': exec_result.stderr,
                    'duration_ms': exec_result.duration_ms,
                }

        except Exception as e:
            attempts.append({
                'temperature': temp,
                'code': f'# Error: {e}',
                'passed': False,
                'reward': -5.0,
                'exit_code': -1,
            })

    if best_attempt is None:
        best_attempt = {
            'code': '', 'passed': False, 'reward': -5.0,
            'exit_code': -1, 'stdout': '', 'stderr': 'All attempts failed',
            'duration_ms': 0, 'temperature': 0,
        }

    # Optionally compute teacher log probabilities
    teacher_logprobs = None
    if include_logprobs and best_attempt['passed'] and _ENC is not None:
        try:
            teacher_logprobs = _compute_teacher_logprobs(
                model, config, device, prompt, best_attempt['code']
            )
        except Exception:
            pass

    return HarvestEntry(
        task_id=_task_id(prompt),
        prompt=prompt,
        test_code=test_code or '',
        difficulty=difficulty,
        solution=best_attempt['code'],
        passed=best_attempt['passed'],
        reward=best_attempt['reward'],
        exit_code=best_attempt['exit_code'],
        stdout=best_attempt.get('stdout', ''),
        stderr=best_attempt.get('stderr', ''),
        duration_ms=best_attempt.get('duration_ms', 0),
        all_attempts=attempts,
        teacher_logprobs=teacher_logprobs,
        temperature=best_attempt.get('temperature', 0),
        timestamp=time.time(),
    )


def _compute_teacher_logprobs(
    model: V5ResonanceModel,
    config: dict,
    device: torch.device,
    prompt: str,
    solution: str,
    max_len: int = 512,
) -> list[float]:
    """
    Compute per-token log probabilities of the solution under the teacher model.
    This is the "teacher signal" — tells the student model how the teacher
    distributes probability over tokens.
    """
    tokens = _ENC.encode(prompt + "\n---\n" + solution, disallowed_special=())[:max_len]
    token_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    type_ids = torch.zeros_like(token_ids)

    with torch.no_grad():
        result = model.forward_pretrain(token_ids, type_ids, labels=None)
        logits = result['logits']  # [1, S, V]
        log_probs = F.log_softmax(logits.float(), dim=-1)  # [1, S, V]

        # Gather target token log probs
        S = log_probs.shape[1]
        target = token_ids[:, :S]
        per_token_lp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [1, S]
        return per_token_lp[0].cpu().tolist()


# ─── Main Harvest Loop ──────────────────────────────────────────────────────

def run_harvest(args):
    """Main knowledge harvesting loop."""
    print("=" * 60)
    print("  SNAP-C1 V5 KNOWLEDGE HARVESTER")
    print("  Crystallizing 8B model knowledge into verified code")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load model
    from v5_core.inference.v5_generate import load_model
    model, config, phase = load_model(args.checkpoint, device)

    runner = TerminalRunner(timeout_sec=args.timeout)

    # Build task list
    tasks = build_task_list(
        include_mined=args.mine_from is not None,
        mine_dir=args.mine_from,
        max_mined=200,
    )

    # Duplicate tasks to reach target count (with slight variations)
    if len(tasks) < args.tasks:
        # Repeat tasks to reach target
        multiplier = (args.tasks // len(tasks)) + 1
        tasks = (tasks * multiplier)[:args.tasks]
    else:
        random.shuffle(tasks)
        tasks = tasks[:args.tasks]

    print(f"\n  Tasks to harvest: {len(tasks)}")
    print(f"  Temperatures:     (0.2, 0.5, 0.8)")
    print(f"  Include logprobs: {args.include_logprobs}")
    print(f"  Output:           {args.output}")

    # Ensure output dir exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Harvest
    results = []
    passed_count = 0
    t0 = time.time()

    with open(output_path, 'a', encoding='utf-8') as f:
        for i, task in enumerate(tasks):
            entry = harvest_single_task(
                model, config, device, task, runner,
                temperatures=(0.2, 0.5, 0.8),
                include_logprobs=args.include_logprobs,
            )

            # Write immediately (don't lose data on crash)
            f.write(json.dumps(entry.to_dict()) + '\n')
            f.flush()

            results.append(entry)
            if entry.passed:
                passed_count += 1

            # Save logprobs separately (too large for JSONL)
            if entry.teacher_logprobs and args.include_logprobs:
                lp_dir = output_path.parent / "teacher_logprobs"
                lp_dir.mkdir(exist_ok=True)
                torch.save(
                    {'task_id': entry.task_id, 'logprobs': entry.teacher_logprobs},
                    lp_dir / f"{entry.task_id}.pt"
                )

            # Progress
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600
            eta = (len(tasks) - i - 1) / max(rate / 3600, 0.001)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(tasks)}] passed={passed_count} "
                      f"({passed_count/(i+1)*100:.0f}%) | "
                      f"rate={rate:.0f}/hr | ETA={eta/60:.1f}min | "
                      f"diff={task.get('difficulty', '?')}")

    # Summary
    elapsed = time.time() - t0
    by_difficulty = {}
    for entry in results:
        d = entry.difficulty
        by_difficulty.setdefault(d, {'total': 0, 'passed': 0})
        by_difficulty[d]['total'] += 1
        if entry.passed:
            by_difficulty[d]['passed'] += 1

    print(f"\n{'=' * 60}")
    print(f"  HARVEST COMPLETE")
    print(f"  Total tasks:    {len(results)}")
    print(f"  Passed:         {passed_count} ({passed_count/max(len(results),1)*100:.0f}%)")
    print(f"  Time:           {elapsed/60:.1f} minutes")
    print(f"  Rate:           {len(results)/elapsed*3600:.0f} tasks/hour")
    print(f"  Output:         {args.output}")
    print(f"\n  By difficulty:")
    for d in sorted(by_difficulty.keys()):
        info = by_difficulty[d]
        print(f"    Diff {d}: {info['passed']}/{info['total']} "
              f"({info['passed']/max(info['total'],1)*100:.0f}%)")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="SNAP-C1 V5 Knowledge Harvester (8B → Local)",
    )

    parser.add_argument('--checkpoint', required=True,
                        help='Path to 8B instruct checkpoint')
    parser.add_argument('--output',
                        default='v5_core/data/knowledge_cache.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--tasks', type=int, default=5000,
                        help='Number of tasks to harvest')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Sandbox timeout per execution')
    parser.add_argument('--include_logprobs', action='store_true',
                        help='Save teacher log probabilities (for KD)')
    parser.add_argument('--mine_from', default=None,
                        help='Directory to mine additional tasks from')

    args = parser.parse_args()
    run_harvest(args)


if __name__ == "__main__":
    main()
