"""
v5_core/training/terminal_loop.py
===================================
Pillar B + C: Terminal Learning Loop + Unit-Test DPO Buffer

Executes generated code in a subprocess sandbox, converts results
into (prompt, chosen, rejected) DPO pairs, and stores them in a
JSON buffer for the AutoDPO trainer.

Usage (collect mode):
    python terminal_loop.py --collect 1000 --output dpo_buffer.json

Usage (single evaluation):
    from v5_core.training.terminal_loop import TerminalRunner
    runner = TerminalRunner()
    result = runner.run("print(1+1)")
    print(result.reward)   # 1.0
"""

import subprocess
import json
import time
import os
import re
import sys
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Result ─────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    code: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    reward: float = 0.0
    timed_out: bool = False

    def is_success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Reward Function ─────────────────────────────────────────────────────────

def compute_reward(
    result: ExecutionResult,
    expected_output: Optional[str] = None,
    test_mode: bool = False,
) -> float:
    """
    Programmatic reward — no human labels needed.

    Scoring:
      + 1.0  ran without crash (exit_code == 0)
      + 2.0  matched expected_output (if provided)
      + 3.0  pytest reported passing tests
      - 0.5  produced any stderr output
      - 1.0  runtime error (Exception in stderr)
      - 2.0  SyntaxError
      - 3.0  timed out
    """
    r = 0.0

    if result.timed_out:
        return -3.0

    if result.exit_code == 0:
        r += 1.0

    if expected_output and expected_output.strip() in result.stdout:
        r += 2.0

    stdout_lower = result.stdout.lower()
    stderr_lower = result.stderr.lower()

    # pytest signals
    passed_match = re.search(r"(\d+) passed", stdout_lower)
    failed_match = re.search(r"(\d+) failed", stdout_lower)
    if passed_match:
        r += 3.0 * int(passed_match.group(1))
    if failed_match:
        r -= 1.5 * int(failed_match.group(1))

    # error penalties
    if result.stderr:
        r -= 0.5
    if "error" in stderr_lower or "exception" in stderr_lower:
        r -= 1.0
    if "syntaxerror" in stderr_lower:
        r -= 2.0
    if "traceback" in stderr_lower:
        r -= 0.5

    return r


# ─── Sandbox Runner ─────────────────────────────────────────────────────────

class TerminalRunner:
    """
    Executes Python code in a subprocess sandbox.

    Safety:
    - shell=False to prevent shell injection
    - timeout enforced (default 10s)
    - no network access (caller responsibility to sandbox further if needed)
    """

    def __init__(
        self,
        timeout_sec: float = 10.0,
        python_exe: Optional[str] = None,
    ):
        self.timeout_sec = timeout_sec
        self.python_exe = python_exe or sys.executable

    def run(
        self,
        code: str,
        expected_output: Optional[str] = None,
        extra_files: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Run a code snippet and return an ExecutionResult with reward.

        Args:
            code:            Python source to execute
            expected_output: Optional string to check for in stdout
            extra_files:     {filename: content} to write before running
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write extra files if provided
            if extra_files:
                for fname, content in extra_files.items():
                    Path(tmpdir, fname).write_text(content, encoding="utf-8")

            # Write main script
            script_path = Path(tmpdir, "_rx_ai_sandbox.py")
            script_path.write_text(code, encoding="utf-8")

            t0 = time.perf_counter()
            timed_out = False
            try:
                proc = subprocess.run(
                    [self.python_exe, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    cwd=tmpdir,
                    shell=False,
                )
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
            except subprocess.TimeoutExpired:
                exit_code = -1
                stdout = ""
                stderr = f"TimeoutExpired after {self.timeout_sec}s"
                timed_out = True

            duration_ms = (time.perf_counter() - t0) * 1000

        result = ExecutionResult(
            code=code,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
        result.reward = compute_reward(result, expected_output=expected_output)
        return result

    def run_pytest(self, test_code: str, source_code: str = "") -> ExecutionResult:
        """
        Run pytest on test_code against source_code.

        test_code:   content of test_something.py
        source_code: the code being tested (written as module.py)
        """
        extra = {}
        if source_code:
            extra["module.py"] = source_code
        extra["test_rx.py"] = test_code

        # Inject pytest runner at bottom
        runner = "\nimport pytest, sys\nsys.exit(pytest.main(['-x', '--tb=short', 'test_rx.py']))\n"
        return self.run(runner, extra_files=extra)


# ─── DPO Pair Generator ─────────────────────────────────────────────────────

@dataclass
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    metadata: dict = field(default_factory=dict)

    def margin(self) -> float:
        return self.chosen_reward - self.rejected_reward

    def to_dict(self) -> dict:
        return asdict(self)


class DPOBufferWriter:
    """Accumulates DPO pairs and saves them to a JSON lines file."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.pairs: list[DPOPair] = []
        # Append mode: load existing if present
        if self.output_path.exists():
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.pairs.append(DPOPair(**json.loads(line)))
            print(f"[DPOBufferWriter] Loaded {len(self.pairs)} existing pairs from {output_path}")

    def add_pair(
        self,
        prompt: str,
        chosen_code: str,
        rejected_code: str,
        chosen_result: ExecutionResult,
        rejected_result: ExecutionResult,
        min_margin: float = 1.0,
    ) -> bool:
        """
        Add a DPO pair only if margin is large enough.
        Returns True if the pair was added.
        """
        margin = chosen_result.reward - rejected_result.reward
        if margin < min_margin:
            return False
        pair = DPOPair(
            prompt=prompt,
            chosen=chosen_code,
            rejected=rejected_code,
            chosen_reward=chosen_result.reward,
            rejected_reward=rejected_result.reward,
            metadata={
                "chosen_exit": chosen_result.exit_code,
                "rejected_exit": rejected_result.exit_code,
                "timestamp": time.time(),
            },
        )
        self.pairs.append(pair)
        return True

    def save(self):
        """Write all pairs as JSON lines."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")
        print(f"[DPOBufferWriter] Saved {len(self.pairs)} pairs to {self.output_path}")

    def stats(self) -> dict:
        if not self.pairs:
            return {"count": 0}
        margins = [p.margin() for p in self.pairs]
        return {
            "count": len(self.pairs),
            "avg_margin": sum(margins) / len(margins),
            "min_margin": min(margins),
            "max_margin": max(margins),
        }


# ─── Demo Collection Loop ────────────────────────────────────────────────────

DEMO_TASKS = [
    {
        "prompt": "Write a Python function that returns the sum of a list.",
        "chosen": "def sum_list(lst):\n    return sum(lst)\nprint(sum_list([1,2,3]))",
        "rejected": "def sum_list(lst): pass\nprint(sum_list([1,2,3]))",
        "expected": "6",
    },
    {
        "prompt": "Write a Python function to reverse a string.",
        "chosen": "def rev(s):\n    return s[::-1]\nprint(rev('hello'))",
        "rejected": "def rev(s):\n    return s\nprint(rev('hello'))",
        "expected": "olleh",
    },
    {
        "prompt": "Write a function to find the maximum in a list.",
        "chosen": "def find_max(lst):\n    return max(lst)\nprint(find_max([3,1,4,1,5]))",
        "rejected": "def find_max(lst):\n    return lst[0]\nprint(find_max([3,1,4,1,5]))",
        "expected": "5",
    },
    {
        "prompt": "Write a function that checks if a number is prime.",
        "chosen": (
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    for i in range(2, int(n**0.5)+1):\n"
            "        if n % i == 0: return False\n"
            "    return True\n"
            "print(is_prime(7), is_prime(4))"
        ),
        "rejected": (
            "def is_prime(n):\n"
            "    return n > 1\n"
            "print(is_prime(7), is_prime(4))"
        ),
        "expected": "True False",
    },
]


def collect_demo_pairs(n: int, output_path: str):
    """Collect DPO pairs from demo tasks using the terminal runner."""
    runner = TerminalRunner()
    writer = DPOBufferWriter(output_path)

    tasks = (DEMO_TASKS * ((n // len(DEMO_TASKS)) + 1))[:n]

    added = 0
    for i, task in enumerate(tasks):
        chosen_result = runner.run(task["chosen"], expected_output=task.get("expected"))
        rejected_result = runner.run(task["rejected"], expected_output=task.get("expected"))

        ok = writer.add_pair(
            prompt=task["prompt"],
            chosen_code=task["chosen"],
            rejected_code=task["rejected"],
            chosen_result=chosen_result,
            rejected_result=rejected_result,
        )
        if ok:
            added += 1

        if (i + 1) % 10 == 0:
            print(
                f"  [{i+1}/{n}] added={added} | "
                f"chosen_reward={chosen_result.reward:.1f} "
                f"rejected_reward={rejected_result.reward:.1f}"
            )

    writer.save()
    print("\n[terminal_loop] Collection complete.")
    print(json.dumps(writer.stats(), indent=2))


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RX.AI V5 Terminal Learning Loop")
    parser.add_argument("--collect", type=int, default=0,
                        help="Collect N DPO pairs from demo tasks")
    parser.add_argument("--output", type=str,
                        default="v5_core/data/dpo_buffer_v5.jsonl",
                        help="Output path for DPO buffer (JSON lines)")
    parser.add_argument("--test", type=str, default="",
                        help="Run a single code snippet and print reward")
    args = parser.parse_args()

    if args.test:
        runner = TerminalRunner()
        result = runner.run(args.test)
        print(f"exit_code : {result.exit_code}")
        print(f"stdout    : {result.stdout.strip()}")
        print(f"stderr    : {result.stderr.strip()}")
        print(f"reward    : {result.reward:.2f}")
        print(f"duration  : {result.duration_ms:.1f}ms")

    elif args.collect > 0:
        print(f"[terminal_loop] Collecting {args.collect} DPO pairs → {args.output}")
        collect_demo_pairs(args.collect, args.output)

    else:
        parser.print_help()
