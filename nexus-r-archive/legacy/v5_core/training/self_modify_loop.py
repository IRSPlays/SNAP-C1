"""
SNAP-C1 V5: Self-Modification Loop
====================================
The model examines its OWN source code, generates improvement patches,
applies them, runs tests, and commits or reverts. Every successful
patch teaches the model (via DPO) to write better patches.

This is the most novel piece: a model that can literally edit its
own training code and architecture, test the change, and learn from
the outcome.

Safety:
  1. Every file gets backed up (.bak) before modification
  2. Tests MUST pass before a patch is committed
  3. Automatic revert on test failure
  4. Rate-limited: max N patches per session
  5. Protected files list (won't touch critical infrastructure)

Usage:
  python self_modify_loop.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --target v5_core/training/terminal_loop.py \\
      --function compute_reward \\
      --goal "add bonus reward for code that runs under 100ms"

  # Autonomous mode (model picks targets):
  python self_modify_loop.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --auto --max_patches 5
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel
from v5_core.architecture.code_introspector import CodeIntrospector, FileContext, Patch
from v5_core.training.terminal_loop import TerminalRunner, DPOBufferWriter, ExecutionResult


# ─── Protected Files ────────────────────────────────────────────────────────
# These files are NEVER modified by the self-modify loop.
# Critical infrastructure that must remain stable.

PROTECTED_FILES = {
    "v5_core/architecture/v5_assembly.py",      # Core model — too risky
    "v5_core/architecture/elastic_context.py",   # Core attention
    "v5_core/training/v5_pretrain.py",           # Pretrain script
    "v5_core/training/self_modify_loop.py",      # Can't modify yourself
    "v5_core/training/auto_dpo_v5.py",           # DPO engine
    "v5_core/utils/dml_ops.py",                  # Device backend
    "setup_do_h200.sh",                          # Deployment script
}


@dataclass
class PatchResult:
    """Records what happened when a patch was applied."""
    target_file: str
    target_function: str
    goal: str
    patch_source: str
    test_passed: bool
    test_output: str
    committed: bool
    reverted: bool
    chosen_response: str = ""    # The response that worked (for DPO)
    rejected_response: str = ""  # The response that failed (for DPO)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Improvement Goal Generator ─────────────────────────────────────────────

IMPROVEMENT_GOALS = [
    # Performance
    "optimize this function to reduce unnecessary memory allocations",
    "add early termination when result is already determined",
    "replace list concatenation with extend for better performance",

    # Robustness
    "add input validation and handle edge cases gracefully",
    "add a try/except block to handle potential exceptions",
    "handle empty inputs and None values safely",

    # Clarity
    "add type hints to function parameters and return type",
    "improve the docstring with parameter descriptions and examples",
    "rename variables to be more descriptive",

    # Features
    "add logging to track function execution for debugging",
    "add an optional parameter with a sensible default",
]


def pick_improvement_goal(file_ctx: FileContext, func_name: str) -> str:
    """Pick a relevant improvement goal for a function."""
    func_info = file_ctx.functions.get(func_name, {})
    args = func_info.get('args', [])
    docstring = func_info.get('docstring', '')

    # Heuristic: pick based on what's missing
    # Check if function has type annotations (look at source, not arg names)
    has_annotations = func_info.get('has_annotations', False)
    if not has_annotations:
        return "add type hints to function parameters and return type"
    if not docstring:
        return "improve the docstring with parameter descriptions and examples"
    # Random otherwise
    import random
    return random.choice(IMPROVEMENT_GOALS)


# ─── Test Runner ─────────────────────────────────────────────────────────────

def run_tests_for_file(
    target_path: str,
    runner: TerminalRunner,
    root: str,
) -> tuple[bool, str]:
    """
    Run tests related to a given file.

    Strategy:
    1. If test_<filename>.py exists, run it
    2. If <filename> has if __name__ == "__main__", run the file itself
    3. Run py_compile as minimum validation
    """
    target = Path(target_path)

    # 1. Look for test file
    test_file = target.parent / f"test_{target.name}"
    if test_file.exists():
        result = runner.run(test_file.read_text(encoding='utf-8'))
        return result.exit_code == 0, f"Test file: {result.stdout}\n{result.stderr}"

    # 2. Syntax check
    validate_code = f"""
import py_compile
import sys
try:
    py_compile.compile(r'{target}', doraise=True)
    print("SYNTAX_OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX_ERROR: {{e}}")
    sys.exit(1)
"""
    result = runner.run(validate_code)
    if result.exit_code != 0:
        return False, f"Syntax error: {result.stderr}"

    # 3. Import check
    module_path = str(target.relative_to(root)).replace(os.sep, '.').replace('.py', '')
    import_code = f"""
import sys
sys.path.insert(0, r'{root}')
try:
    import importlib
    mod = importlib.import_module('{module_path}')
    print("IMPORT_OK")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
    sys.exit(1)
"""
    result = runner.run(import_code)
    return result.exit_code == 0, f"Import test: {result.stdout}\n{result.stderr}"


# ─── Self-Modify Engine ─────────────────────────────────────────────────────

def generate_patch(
    model: V5ResonanceModel,
    config: dict,
    device: torch.device,
    file_ctx: FileContext,
    func_name: str,
    goal: str,
) -> Optional[str]:
    """
    Use the model to generate a new version of a function.

    The prompt includes the current source, function context, and
    the improvement goal.
    """
    from v5_core.inference.v5_generate import generate

    # Extract current function source
    import ast
    current_source = None
    for node in ast.walk(file_ctx.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                current_source = ast.get_source_segment(file_ctx.source, node)
                if current_source is None:
                    # Fallback: unparse the node
                    current_source = ast.unparse(node)
                break

    if current_source is None:
        print(f"  [SelfModify] Could not find function '{func_name}'")
        return None

    # Build context
    summary = file_ctx.to_summary()
    callers = [k for k, vs in file_ctx.call_graph.items() if func_name in vs]

    prompt = f"""You are an expert Python programmer. Your task is to improve a function.

FILE: {file_ctx.path}
CALLED BY: {', '.join(callers) if callers else 'unknown'}
OTHER FUNCTIONS IN FILE: {', '.join(summary['functions'][:10])}

CURRENT IMPLEMENTATION:
```python
{current_source}
```

GOAL: {goal}

Write ONLY the improved function. Keep the same name and signature.
Do not include any explanations, just the Python code for the function.

```python
"""

    try:
        response = generate(
            model=model, prompt=prompt,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.9, top_k=40,
            device=device, config=config,
            phase='instruct',
        )
        # Strip markdown
        lines = response.strip().split('\n')
        if lines and lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines)
    except Exception as e:
        print(f"  [SelfModify] Generation error: {e}")
        return None


def self_modify_round(
    model: V5ResonanceModel,
    config: dict,
    device: torch.device,
    introspector: CodeIntrospector,
    target_rel_path: str,
    func_name: str,
    goal: str,
    runner: TerminalRunner,
    root: str,
) -> PatchResult:
    """
    One round of self-modification:
    1. Read current code
    2. Generate improved version
    3. Apply patch (with backup)
    4. Run tests
    5. Commit or revert
    """
    timestamp = time.time()
    full_path = str(Path(root) / target_rel_path)

    # Get file context
    file_ctx = introspector.get_file_context(target_rel_path)
    if file_ctx is None:
        return PatchResult(
            target_file=target_rel_path, target_function=func_name,
            goal=goal, patch_source="", test_passed=False,
            test_output="Could not load file", committed=False,
            reverted=False, timestamp=timestamp,
        )

    # Get original source for DPO
    import ast as _ast
    original_source = None
    for node in _ast.walk(file_ctx.tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name == func_name:
                original_source = _ast.get_source_segment(file_ctx.source, node)
                if original_source is None:
                    original_source = _ast.unparse(node)
                break

    # Generate patch
    print(f"  [SelfModify] Generating patch for {func_name} in {target_rel_path}")
    print(f"  [SelfModify] Goal: {goal}")
    new_source = generate_patch(model, config, device, file_ctx, func_name, goal)

    if new_source is None or len(new_source.strip()) < 10:
        return PatchResult(
            target_file=target_rel_path, target_function=func_name,
            goal=goal, patch_source="", test_passed=False,
            test_output="Generation failed", committed=False,
            reverted=False, timestamp=timestamp,
        )

    # Create patch object
    patch = Patch(
        target_path=full_path,
        target_function=func_name,
        new_source=new_source,
        description=goal,
    )

    # Apply patch
    success, msg = introspector.apply_patch(patch)
    if not success:
        return PatchResult(
            target_file=target_rel_path, target_function=func_name,
            goal=goal, patch_source=new_source, test_passed=False,
            test_output=f"Patch application failed: {msg}",
            committed=False, reverted=False, timestamp=timestamp,
        )

    print(f"  [SelfModify] Patch applied, running tests...")

    # Run tests
    test_passed, test_output = run_tests_for_file(full_path, runner, root)

    if test_passed:
        print(f"  [SelfModify] Tests PASSED — committing patch")
        # Remove backup
        backup = Path(full_path + ".bak")
        if backup.exists():
            backup.unlink()
        return PatchResult(
            target_file=target_rel_path, target_function=func_name,
            goal=goal, patch_source=new_source, test_passed=True,
            test_output=test_output, committed=True, reverted=False,
            chosen_response=new_source,
            rejected_response=original_source or "",
            timestamp=timestamp,
        )
    else:
        print(f"  [SelfModify] Tests FAILED — reverting")
        introspector.revert_patch(patch)
        return PatchResult(
            target_file=target_rel_path, target_function=func_name,
            goal=goal, patch_source=new_source, test_passed=False,
            test_output=test_output, committed=False, reverted=True,
            # In failed case, original = chosen, patch = rejected
            chosen_response=original_source or "",
            rejected_response=new_source,
            timestamp=timestamp,
        )


def find_modifiable_targets(
    introspector: CodeIntrospector,
    subdir: str = "v5_core",
    max_targets: int = 20,
) -> list[tuple[str, str]]:
    """
    Find (file, function) pairs that are safe to modify.
    Excludes protected files and private/dunder methods.
    """
    import random
    targets = []
    contexts = introspector.get_all_contexts(subdir)

    for ctx in contexts:
        rel_path = str(Path(ctx.path).relative_to(introspector.root))
        # Normalize path separators
        rel_path_fwd = rel_path.replace('\\', '/')

        if rel_path_fwd in PROTECTED_FILES:
            continue

        for func_name in ctx.functions:
            # Skip private, dunder, and very short functions
            if func_name.startswith('_'):
                continue
            func_info = ctx.functions[func_name]
            if len(func_info.get('args', [])) == 0:
                continue
            targets.append((rel_path, func_name))

    random.shuffle(targets)
    return targets[:max_targets]


# ─── Main ────────────────────────────────────────────────────────────────────

def run_self_modify(args):
    """Main self-modification loop."""
    print("=" * 60)
    print("SNAP-C1 V5 SELF-MODIFICATION LOOP")
    print("=" * 60)

    device = get_device()
    root = args.root or str(Path(__file__).parent.parent.parent)
    print(f"Device: {device}")
    print(f"Root:   {root}")

    # Load model
    from v5_core.inference.v5_generate import load_model
    model, config, phase = load_model(args.checkpoint, device)

    # Optionally load LoRA
    if args.lora:
        from v5_core.training.auto_dpo_v5 import inject_lora, load_lora
        lora_modules = inject_lora(model, rank=16, alpha=32.0)
        load_lora(lora_modules, args.lora)

    # Setup
    introspector = CodeIntrospector(root)
    runner = TerminalRunner(timeout_sec=args.timeout)
    writer = DPOBufferWriter(args.dpo_output)
    results = []

    if args.auto:
        # Autonomous mode: pick targets automatically
        targets = find_modifiable_targets(introspector)
        print(f"\n[Auto] Found {len(targets)} modifiable targets")
        targets = targets[:args.max_patches]
    elif args.target and args.function:
        # Check protection even for manual targets
        norm_target = args.target.replace('\\', '/')
        if norm_target in PROTECTED_FILES:
            print(f"ERROR: {args.target} is a protected file. Cannot modify.")
            return
        targets = [(args.target, args.function)]
    else:
        print("ERROR: Specify --target/--function or --auto")
        return

    # Self-modify loop
    print(f"\n{'=' * 60}")
    print(f"Running {len(targets)} modification round(s)")
    print(f"{'=' * 60}\n")

    for i, (target_path, func_name) in enumerate(targets):
        file_ctx = introspector.get_file_context(target_path)
        if file_ctx is None:
            print(f"  [Skip] Could not load {target_path}")
            continue
        goal = args.goal or pick_improvement_goal(file_ctx, func_name)

        print(f"\n--- Round {i+1}/{len(targets)} ---")
        result = self_modify_round(
            model=model, config=config, device=device,
            introspector=introspector,
            target_rel_path=target_path,
            func_name=func_name,
            goal=goal,
            runner=runner,
            root=root,
        )
        results.append(result)

        # Create DPO pair from the result
        # When test passes: chosen = new code (committed), rejected = original
        # When test fails: chosen = original (safe), rejected = broken patch
        if result.chosen_response and result.rejected_response:
            prompt = (f"Improve the function `{func_name}` in "
                      f"`{target_path}`. Goal: {goal}")
            writer.add_pair(
                prompt=prompt,
                chosen_code=result.chosen_response,
                rejected_code=result.rejected_response,
                chosen_result=ExecutionResult(
                    code=result.chosen_response, exit_code=0,
                    stdout='', stderr='', duration_ms=0,
                    reward=3.0 if result.test_passed else 1.0,
                ),
                rejected_result=ExecutionResult(
                    code=result.rejected_response, exit_code=1,
                    stdout='', stderr='', duration_ms=0,
                    reward=-1.0 if result.test_passed else -3.0,
                ),
                min_margin=0.5,
            )

        # Log
        status = "COMMITTED" if result.committed else "REVERTED" if result.reverted else "FAILED"
        print(f"  Result: {status}")

    # Save DPO pairs
    writer.save()

    # Save session log
    log_path = Path(args.dpo_output).parent / "self_modify_log.json"
    with open(log_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Summary
    committed = sum(1 for r in results if r.committed)
    reverted = sum(1 for r in results if r.reverted)
    failed = len(results) - committed - reverted

    print(f"\n{'=' * 60}")
    print(f"SELF-MODIFICATION COMPLETE")
    print(f"  Committed: {committed}")
    print(f"  Reverted:  {reverted}")
    print(f"  Failed:    {failed}")
    print(f"  DPO pairs: {writer.stats()}")
    print(f"  Log:       {log_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Self-Modify Loop")

    parser.add_argument('--checkpoint', required=True,
                        help='Path to V5 checkpoint')
    parser.add_argument('--lora', default=None,
                        help='Optional LoRA checkpoint')
    parser.add_argument('--root', default=None,
                        help='Project root directory')

    # Target specification
    parser.add_argument('--target', default=None,
                        help='Relative path to file to modify')
    parser.add_argument('--function', default=None,
                        help='Name of function to improve')
    parser.add_argument('--goal', default=None,
                        help='Improvement goal (natural language)')

    # Auto mode
    parser.add_argument('--auto', action='store_true',
                        help='Automatically find and improve targets')
    parser.add_argument('--max_patches', type=int, default=5,
                        help='Max patches in auto mode')

    # Output
    parser.add_argument('--dpo_output', default='v5_core/data/dpo_buffer_v5.jsonl',
                        help='DPO buffer output file')
    parser.add_argument('--timeout', type=float, default=15.0,
                        help='Test timeout in seconds')

    args = parser.parse_args()
    run_self_modify(args)


if __name__ == "__main__":
    main()
