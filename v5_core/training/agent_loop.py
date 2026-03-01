"""
SNAP-C1 V5: Agent Loop (Interactive Coding Agent)
===================================================
The user-facing runtime that turns V5 into a coding assistant.

This is the system that ties everything together:
  - The user gives a task ("write a sorting function")
  - The agent plans, generates code, executes it, verifies
  - Every interaction produces a DPO pair → self-improvement
  - The model learns from EVERY conversation

Architecture (single function call chain, no frameworks):
  1. Parse & Plan: understand what the user wants
  2. Generate Code: produce a solution (with py_compile validation)
  3. Execute & Verify: run in sandbox, check output
  4. Iterate: if tests fail, revise up to N times
  5. Respond: return verified code to the user
  6. Learn: (chosen=working code, rejected=broken attempts) → DPO buffer

No LangChain. No AutoGen. No frameworks. Pure PyTorch + subprocess.

Usage:
  # Interactive mode:
  python agent_loop.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt

  # Single task:
  python agent_loop.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --task "Write a function that checks if a number is prime"

  # Batch mode (process tasks from file):
  python agent_loop.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --batch tasks.txt --output results/
"""

import os
import sys
import json
import time
import argparse
import py_compile
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel
from v5_core.training.terminal_loop import TerminalRunner, DPOBufferWriter, ExecutionResult


# ─── Result Types ────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """The agent's response to a user task."""
    task: str
    code: str
    passed: bool
    stdout: str
    stderr: str
    attempts: int
    total_time_sec: float
    revision_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'=' * 50}")
        print(f"[{status}] Task: {self.task[:80]}...")
        print(f"Attempts: {self.attempts} | Time: {self.total_time_sec:.1f}s")
        print(f"{'─' * 50}")
        print(self.code)
        if self.stdout.strip():
            print(f"{'─' * 50}")
            print(f"Output: {self.stdout[:500]}")
        if not self.passed and self.stderr.strip():
            print(f"Error: {self.stderr[:300]}")
        print(f"{'=' * 50}")


# ─── Code Validation ────────────────────────────────────────────────────────

def validate_syntax(code: str) -> tuple[bool, str]:
    """Check if code is valid Python without executing it."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                      encoding='utf-8') as f:
        f.write(code)
        tmp_path = f.name

    try:
        py_compile.compile(tmp_path, doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def strip_code_blocks(text: str) -> str:
    """Remove markdown ```python ... ``` wrapping."""
    lines = text.strip().split('\n')
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines)


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a model response."""
    # Try to find code blocks first
    import re
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        # Return the longest code block
        return max(blocks, key=len).strip()

    # Otherwise strip markdown and return
    return strip_code_blocks(response)


# ─── Agent Core ──────────────────────────────────────────────────────────────

class CodingAgent:
    """
    The V5 interactive coding agent.

    Controls:
    - max_attempts: how many revision attempts before giving up
    - verify_with_tests: whether to execute code (False = generation only)
    - learn_from_interactions: whether to write DPO pairs
    """

    def __init__(
        self,
        model: V5ResonanceModel,
        config: dict,
        device: torch.device,
        dpo_output: str = "v5_core/data/dpo_buffer_v5.jsonl",
        max_attempts: int = 3,
        verify_with_tests: bool = True,
        learn_from_interactions: bool = True,
        timeout_sec: float = 10.0,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.max_attempts = max_attempts
        self.verify = verify_with_tests
        self.learn = learn_from_interactions
        self.runner = TerminalRunner(timeout_sec=timeout_sec)
        self.writer = DPOBufferWriter(dpo_output) if learn_from_interactions else None
        self.history = []  # conversation memory

        # Stats
        self.total_tasks = 0
        self.total_passed = 0
        self.total_dpo_pairs = 0

    def solve(self, task: str, test_code: Optional[str] = None) -> AgentResult:
        """
        Solve a coding task end-to-end.

        1. Generate initial solution
        2. Validate syntax
        3. Execute in sandbox
        4. If failed, generate revision with error context
        5. Repeat up to max_attempts
        6. Record DPO pair (best attempt vs worst attempt)
        """
        from v5_core.inference.v5_generate import generate

        t0 = time.time()
        attempts = []
        best_result = None
        best_code = ""
        worst_result = None
        worst_code = ""

        for attempt in range(self.max_attempts):
            # Build prompt
            if attempt == 0:
                prompt = self._build_initial_prompt(task)
            else:
                # Revision prompt: include the error from last attempt
                last_code, last_result = attempts[-1]
                prompt = self._build_revision_prompt(
                    task, last_code, last_result
                )

            # Generate code
            try:
                response = generate(
                    model=self.model, prompt=prompt,
                    max_new_tokens=512,
                    temperature=0.4 + attempt * 0.15,  # Higher temp on retries
                    top_p=0.92, top_k=50,
                    device=self.device, config=self.config,
                    phase='instruct',
                )
                code = extract_code_from_response(response)
            except Exception as e:
                code = f"# Generation error: {e}"

            # Syntax check
            valid, syntax_err = validate_syntax(code)
            if not valid:
                exec_result = ExecutionResult(
                    code=code, exit_code=1, stdout='',
                    stderr=f"SyntaxError: {syntax_err}",
                    duration_ms=0, reward=-3.0,
                )
                attempts.append((code, exec_result))
                continue

            # Execute
            if self.verify:
                if test_code:
                    full_code = code + "\n\n" + test_code
                    exec_result = self.runner.run(full_code, expected_output="PASS")
                else:
                    exec_result = self.runner.run(code)
            else:
                exec_result = ExecutionResult(
                    code=code, exit_code=0, stdout='(not verified)',
                    stderr='', duration_ms=0, reward=1.0,
                )

            attempts.append((code, exec_result))

            # Track best/worst
            if best_result is None or exec_result.reward > best_result.reward:
                best_result = exec_result
                best_code = code
            if worst_result is None or exec_result.reward < worst_result.reward:
                worst_result = exec_result
                worst_code = code

            # Early exit on success
            if exec_result.exit_code == 0 and exec_result.reward > 0:
                break

        # Record DPO pair
        if self.learn and self.writer and best_result and worst_result:
            if best_result.reward != worst_result.reward:
                self.writer.add_pair(
                    prompt=task,
                    chosen_code=best_code,
                    rejected_code=worst_code,
                    chosen_result=best_result,
                    rejected_result=worst_result,
                    min_margin=0.5,
                )
                self.total_dpo_pairs += 1

        # Build result
        elapsed = time.time() - t0
        final_code = best_code if best_result else (attempts[-1][0] if attempts else "")
        final_result = best_result or (attempts[-1][1] if attempts else None)
        passed = (final_result is not None and final_result.exit_code == 0
                  and final_result.reward > 0)

        self.total_tasks += 1
        if passed:
            self.total_passed += 1

        result = AgentResult(
            task=task,
            code=final_code,
            passed=passed,
            stdout=final_result.stdout if final_result else '',
            stderr=final_result.stderr if final_result else '',
            attempts=len(attempts),
            total_time_sec=elapsed,
            revision_history=[
                {
                    'attempt': i + 1,
                    'reward': r.reward,
                    'exit_code': r.exit_code,
                    'code_preview': c[:100],
                }
                for i, (c, r) in enumerate(attempts)
            ],
        )

        # Add to conversation history
        self.history.append({
            'task': task,
            'passed': passed,
            'attempts': len(attempts),
        })

        return result

    def _build_initial_prompt(self, task: str) -> str:
        """Build the initial generation prompt."""
        # Include brief conversation history for context
        history_ctx = ""
        if self.history:
            recent = self.history[-3:]  # Last 3 interactions
            history_lines = []
            for h in recent:
                status = "passed" if h['passed'] else "failed"
                history_lines.append(f"- {h['task'][:50]}... ({status})")
            history_ctx = f"\nRecent conversation:\n" + "\n".join(history_lines) + "\n"

        return f"""You are an expert Python programmer. Write clean, working Python code.
{history_ctx}
Task: {task}

Requirements:
- Write complete, runnable Python code
- Handle edge cases
- Use clear variable names
- Include the function definition(s) requested

```python
"""

    def _build_revision_prompt(
        self, task: str, failed_code: str, failed_result: ExecutionResult
    ) -> str:
        """Build a revision prompt that includes the error."""
        error = failed_result.stderr[:400] if failed_result.stderr else "Unknown error"
        return f"""You are an expert Python programmer. Fix the code below.

Task: {task}

Previous attempt (FAILED):
```python
{failed_code}
```

Error:
{error}

Write the corrected version. Fix all bugs.

```python
"""

    def save_state(self):
        """Save DPO pairs and stats."""
        if self.writer:
            self.writer.save()

    def print_stats(self):
        rate = self.total_passed / max(self.total_tasks, 1) * 100
        print(f"\n[Agent Stats] Tasks: {self.total_tasks} | "
              f"Passed: {self.total_passed} ({rate:.0f}%) | "
              f"DPO pairs: {self.total_dpo_pairs}")


# ─── Interactive REPL ────────────────────────────────────────────────────────

def interactive_mode(agent: CodingAgent):
    """Interactive coding assistant REPL."""
    print("\n" + "=" * 60)
    print("  SNAP-C1 V5 — Coding Agent")
    print("  Type a coding task, or 'quit' to exit.")
    print("  Type 'stats' to see performance statistics.")
    print("=" * 60 + "\n")

    while True:
        try:
            task = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not task:
            continue
        if task.lower() in ('quit', 'exit', 'q'):
            break
        if task.lower() == 'stats':
            agent.print_stats()
            continue
        if task.lower() == 'save':
            agent.save_state()
            print("State saved.")
            continue

        # Solve the task
        result = agent.solve(task)
        result.display()

    # Save on exit
    agent.save_state()
    agent.print_stats()
    print("\nGoodbye!")


# ─── Batch Mode ──────────────────────────────────────────────────────────────

def batch_mode(agent: CodingAgent, tasks_file: str, output_dir: str):
    """Process tasks from a file, one per line."""
    tasks_path = Path(tasks_file)
    if not tasks_path.exists():
        print(f"Tasks file not found: {tasks_file}")
        return

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tasks = [
        line.strip() for line in tasks_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

    print(f"Processing {len(tasks)} tasks from {tasks_file}")
    results = []

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task[:60]}...")
        result = agent.solve(task)
        result.display()
        results.append(result.to_dict())

        # Save individual result
        (output / f"task_{i+1:04d}.json").write_text(
            json.dumps(result.to_dict(), indent=2)
        )

    # Save summary
    summary = {
        'total': len(results),
        'passed': sum(1 for r in results if r['passed']),
        'failed': sum(1 for r in results if not r['passed']),
        'avg_attempts': sum(r['attempts'] for r in results) / max(len(results), 1),
        'avg_time': sum(r['total_time_sec'] for r in results) / max(len(results), 1),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    agent.save_state()
    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE: {summary['passed']}/{summary['total']} passed")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 60}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Coding Agent")

    parser.add_argument('--checkpoint', required=True,
                        help='Path to V5 checkpoint')
    parser.add_argument('--lora', default=None,
                        help='Optional LoRA checkpoint')
    parser.add_argument('--task', default=None,
                        help='Single task to solve (non-interactive)')
    parser.add_argument('--batch', default=None,
                        help='File with tasks (one per line)')
    parser.add_argument('--output', default='v5_core/data/agent_results',
                        help='Output directory for batch results')
    parser.add_argument('--dpo_output',
                        default='v5_core/data/dpo_buffer_v5.jsonl',
                        help='DPO buffer output path')
    parser.add_argument('--max_attempts', type=int, default=3,
                        help='Max revision attempts per task')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Sandbox execution timeout (seconds)')
    parser.add_argument('--no_learn', action='store_true',
                        help='Disable DPO learning from interactions')
    parser.add_argument('--no_verify', action='store_true',
                        help='Skip code execution (generation only)')

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    from v5_core.inference.v5_generate import load_model
    model, config, phase = load_model(args.checkpoint, device)

    # Load LoRA if provided
    if args.lora:
        from v5_core.training.auto_dpo_v5 import inject_lora, load_lora
        lora_modules = inject_lora(model, rank=16, alpha=32.0)
        load_lora(lora_modules, args.lora)

    # Create agent
    agent = CodingAgent(
        model=model, config=config, device=device,
        dpo_output=args.dpo_output,
        max_attempts=args.max_attempts,
        verify_with_tests=not args.no_verify,
        learn_from_interactions=not args.no_learn,
        timeout_sec=args.timeout,
    )

    # Run mode
    if args.task:
        result = agent.solve(args.task)
        result.display()
        agent.save_state()
    elif args.batch:
        batch_mode(agent, args.batch, args.output)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
