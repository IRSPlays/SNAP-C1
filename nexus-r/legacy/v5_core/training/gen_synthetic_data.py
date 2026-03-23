"""
Synthetic Instruction Data Generator
======================================
Generates diverse instruction-tuning data from raw Python source files.
No LLM needed — uses code analysis heuristics to create prompt/response pairs.

Categories:
  1. Bug-fix pairs (inject bug → original is the fix)
  2. Code completion (give partial function → complete it)
  3. Docstring generation (give code → explain it)
  4. Code explanation (what does this code do?)
  5. Function signature (describe behavior → generate function)
  6. Refactoring (make code more pythonic)

Output: JSON file compatible with InstructionDataset
"""

import os
import re
import ast
import json
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Code extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_functions(source: str) -> List[Dict]:
    """Extract top-level and class method functions from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get source lines
            try:
                lines = source.splitlines()
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 1
                func_source = '\n'.join(lines[start:end])

                # Skip tiny functions and test functions
                if len(func_source.strip()) < 30:
                    continue
                if node.name.startswith('test_'):
                    continue

                # Get docstring
                docstring = ast.get_docstring(node) or ""

                # Get args
                args = []
                for arg in node.args.args:
                    args.append(arg.arg)

                functions.append({
                    'name': node.name,
                    'source': func_source,
                    'docstring': docstring,
                    'args': args,
                    'lines': end - start,
                })
            except Exception:
                continue

    return functions


def extract_classes(source: str) -> List[Dict]:
    """Extract classes with their methods."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            try:
                lines = source.splitlines()
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 1
                class_source = '\n'.join(lines[start:end])

                if len(class_source.strip()) < 50:
                    continue

                docstring = ast.get_docstring(node) or ""
                methods = [n.name for n in node.body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

                classes.append({
                    'name': node.name,
                    'source': class_source,
                    'docstring': docstring,
                    'methods': methods,
                    'lines': end - start,
                })
            except Exception:
                continue

    return classes


# ──────────────────────────────────────────────────────────────────────────────
# Bug injection strategies
# ──────────────────────────────────────────────────────────────────────────────
BUG_TEMPLATES = [
    # Off-by-one
    ("< ", "<= "),
    ("<= ", "< "),
    ("> ", ">= "),
    (">= ", "> "),
    ("range(1,", "range(0,"),
    ("range(0,", "range(1,"),
    # Wrong operator
    (" + ", " - "),
    (" - ", " + "),
    (" * ", " / "),
    (" == ", " != "),
    (" != ", " == "),
    (" and ", " or "),
    (" or ", " and "),
    (" is ", " is not "),
    (" not in ", " in "),
    # Missing return
    ("return ", "# return "),
    # Wrong variable
    ("True", "False"),
    ("False", "True"),
    # None check removal
    ("is None", "is not None"),
    ("is not None", "is None"),
    # Bracket errors
    ("[0]", "[1]"),
    ("[-1]", "[0]"),
    # String errors
    ('""', '"None"'),
    ("[]", "[None]"),
]

BUG_DESCRIPTIONS = {
    ("< ", "<= "): "off-by-one error in comparison (< should be <=)",
    ("<= ", "< "): "off-by-one error in comparison (<= should be <)",
    ("> ", ">= "): "off-by-one error in comparison (> should be >=)",
    (">= ", "> "): "off-by-one error in comparison (>= should be >)",
    ("range(1,", "range(0,"): "wrong range start index",
    ("range(0,", "range(1,"): "wrong range start index",
    (" + ", " - "): "wrong arithmetic operator (+ should be -)",
    (" - ", " + "): "wrong arithmetic operator (- should be +)",
    (" * ", " / "): "wrong arithmetic operator (* should be /)",
    (" == ", " != "): "wrong comparison operator (== should be !=)",
    (" != ", " == "): "wrong comparison operator (!= should be ==)",
    (" and ", " or "): "wrong logical operator (and should be or)",
    (" or ", " and "): "wrong logical operator (or should be and)",
    (" is ", " is not "): "wrong identity check",
    (" not in ", " in "): "wrong membership test",
    ("return ", "# return "): "missing return statement",
    ("True", "False"): "wrong boolean value",
    ("False", "True"): "wrong boolean value",
    ("is None", "is not None"): "inverted None check",
    ("is not None", "is None"): "inverted None check",
    ("[0]", "[1]"): "wrong index",
    ("[-1]", "[0]"): "wrong index for last element",
    ('""', '"None"'): "wrong default string value",
    ("[]", "[None]"): "wrong default list value",
}


def inject_bug(code: str) -> Optional[Tuple[str, str, str]]:
    """Inject a realistic bug into code. Returns (buggy_code, bug_description, original_correct)."""
    random.shuffle(BUG_TEMPLATES)
    for correct, buggy in BUG_TEMPLATES:
        if correct in code:
            # Only replace first occurrence to keep bug localized
            buggy_code = code.replace(correct, buggy, 1)
            if buggy_code != code:
                desc = BUG_DESCRIPTIONS.get((correct, buggy), f"replaced '{correct.strip()}' with '{buggy.strip()}'")
                return buggy_code, desc, code
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Data generation strategies
# ──────────────────────────────────────────────────────────────────────────────
def gen_bugfix_pairs(functions: List[Dict]) -> List[Dict]:
    """Generate bug-fix instruction pairs."""
    pairs = []
    for func in functions:
        if func['lines'] < 3 or func['lines'] > 40:
            continue
        result = inject_bug(func['source'])
        if result:
            buggy, desc, original = result
            prompt = f"Fix the bug in this Python function ({desc}):\n\n```python\n{buggy}\n```"
            response = f"<think>\nThe bug is: {desc}.\nI need to fix this to make the function work correctly.\n</think>\n{original}"
            pairs.append({"prompt": prompt, "target_code": response})
    return pairs


def gen_completion_pairs(functions: List[Dict]) -> List[Dict]:
    """Generate code completion pairs (give first half, complete rest)."""
    pairs = []
    for func in functions:
        lines = func['source'].splitlines()
        if len(lines) < 4:
            continue

        # Give first 30-60% of the function
        split = random.randint(max(2, len(lines) // 3), max(3, len(lines) * 2 // 3))
        partial = '\n'.join(lines[:split])
        full = func['source']

        prompt = f"Complete this Python function:\n\n```python\n{partial}\n    # ... complete the rest\n```"
        response = f"<think>\nI need to complete the function `{func['name']}`.\n</think>\n{full}"
        pairs.append({"prompt": prompt, "target_code": response})

    return pairs


def gen_explanation_pairs(functions: List[Dict]) -> List[Dict]:
    """Generate code explanation pairs."""
    pairs = []
    for func in functions:
        if func['lines'] < 3 or func['lines'] > 30:
            continue

        prompt = f"What does this Python function do?\n\n```python\n{func['source']}\n```"

        # Generate explanation from function name and docstring
        name_words = re.sub(r'_+', ' ', func['name']).strip()
        if func['docstring']:
            explanation = func['docstring'].split('\n')[0].strip()
        else:
            explanation = f"This function '{name_words}' takes {len(func['args'])} argument(s)"
            if func['args']:
                explanation += f" ({', '.join(func['args'][:3])})"
            explanation += " and performs the operation described by its name."

        args_desc = ", ".join(f"`{a}`" for a in func['args'][:4]) if func['args'] else "no arguments"
        response = (
            f"<think>\nAnalyzing function `{func['name']}` with parameters: {args_desc}.\n"
            f"</think>\n"
            f"The function `{func['name']}` {explanation}\n\n"
            f"Parameters: {args_desc}\n"
            f"Lines of code: {func['lines']}"
        )
        pairs.append({"prompt": prompt, "target_code": response})

    return pairs


def gen_docstring_pairs(functions: List[Dict]) -> List[Dict]:
    """Generate 'add docstring' instruction pairs."""
    pairs = []
    for func in functions:
        if func['docstring'] or func['lines'] < 3 or func['lines'] > 25:
            continue  # Only for functions WITHOUT docstrings

        prompt = f"Add a docstring to this Python function:\n\n```python\n{func['source']}\n```"

        # Create a simple docstring from the function name
        name_words = re.sub(r'_+', ' ', func['name']).strip()
        args_doc = ""
        for arg in func['args'][:4]:
            if arg != 'self':
                args_doc += f"\n        {arg}: TODO"

        # Insert docstring after def line
        lines = func['source'].splitlines()
        indent = len(lines[0]) - len(lines[0].lstrip()) + 4
        doc_indent = ' ' * indent
        docstring = f'{doc_indent}"""{name_words.capitalize()}.{args_doc}\n{doc_indent}"""'

        new_lines = [lines[0], docstring] + lines[1:]
        new_source = '\n'.join(new_lines)

        response = f"<think>\nThe function needs a docstring describing what '{name_words}' does.\n</think>\n{new_source}"
        pairs.append({"prompt": prompt, "target_code": response})

    return pairs


def gen_signature_pairs(functions: List[Dict]) -> List[Dict]:
    """Generate 'write a function that...' pairs."""
    pairs = []
    for func in functions:
        if func['lines'] < 3 or func['lines'] > 20 or not func['docstring']:
            continue

        name_words = re.sub(r'_+', ' ', func['name']).strip()
        desc = func['docstring'].split('\n')[0].strip()

        prompt = f"Write a Python function called `{func['name']}` that {desc.lower()}"
        response = f"<think>\nI need to write a function `{func['name']}` that {desc.lower()}\n</think>\n{func['source']}"
        pairs.append({"prompt": prompt, "target_code": response})

    return pairs


def gen_refactor_pairs(functions: List[Dict]) -> List[Dict]:
    """Generate refactoring pairs (verbose → pythonic)."""
    pairs = []
    for func in functions:
        source = func['source']

        # Find functions with verbose patterns we can identify
        refactored = source

        # Pattern: for loop append → list comprehension (just describe it)
        if 'for ' in source and '.append(' in source and len(source.splitlines()) < 15:
            prompt = f"Make this Python function more Pythonic:\n\n```python\n{source}\n```"
            response = (
                f"<think>\nThe function uses a for-loop with append, which could potentially "
                f"be converted to a list comprehension for more Pythonic style.\n</think>\n{source}"
            )
            pairs.append({"prompt": prompt, "target_code": response})

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_from_files(data_dirs: List[str], max_files: int = 5000,
                        target_samples: int = 50000) -> List[Dict]:
    """Generate instruction samples from Python source files."""

    # Collect files
    all_files = []
    for data_dir in data_dirs:
        for root, dirs, files in os.walk(data_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')
                      and d not in ('__pycache__', 'venv', 'node_modules', '.git',
                                    'chroma_db', 'test', 'tests')]
            for f in files:
                if f.endswith('.py') and not f.startswith('test_'):
                    all_files.append(os.path.join(root, f))

    random.shuffle(all_files)
    all_files = all_files[:max_files]
    print(f"  Found {len(all_files)} source files (excluding test files)")

    # Extract functions from all files
    all_functions = []
    all_classes = []
    files_parsed = 0

    for fp in all_files:
        try:
            source = Path(fp).read_text(encoding='utf-8', errors='ignore')
            if len(source) < 100:
                continue

            funcs = extract_functions(source)
            classes = extract_classes(source)

            all_functions.extend(funcs)
            all_classes.extend(classes)
            files_parsed += 1
        except Exception:
            continue

    print(f"  Parsed {files_parsed} files → {len(all_functions)} functions, {len(all_classes)} classes")

    # Generate data with each strategy
    all_pairs = []

    generators = [
        ("Bug-fix", gen_bugfix_pairs),
        ("Completion", gen_completion_pairs),
        ("Explanation", gen_explanation_pairs),
        ("Docstring", gen_docstring_pairs),
        ("Signature", gen_signature_pairs),
        ("Refactoring", gen_refactor_pairs),
    ]

    for name, gen_func in generators:
        pairs = gen_func(all_functions)
        print(f"    {name}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    # Shuffle
    random.shuffle(all_pairs)

    # Truncate to target
    if len(all_pairs) > target_samples:
        all_pairs = all_pairs[:target_samples]

    print(f"\n  Total generated: {len(all_pairs)} instruction pairs")
    return all_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic instruction data")
    parser.add_argument('--data_dirs', nargs='+', default=['training_data'],
                        help='Directories with Python source files')
    parser.add_argument('--output', default='v5_core/data/v5_synthetic_instructions.json',
                        help='Output JSON file')
    parser.add_argument('--max_files', type=int, default=5000,
                        help='Max source files to process')
    parser.add_argument('--target', type=int, default=50000,
                        help='Target number of samples')
    parser.add_argument('--merge', nargs='*', default=None,
                        help='Additional JSON/JSONL files to merge in')
    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC INSTRUCTION DATA GENERATOR")
    print("=" * 60)

    # Generate from code
    pairs = generate_from_files(args.data_dirs, args.max_files, args.target)

    # Merge existing datasets
    if args.merge:
        for fp in args.merge:
            if not os.path.exists(fp):
                print(f"  Skipping {fp} (not found)")
                continue
            try:
                if fp.endswith('.jsonl'):
                    with open(fp, encoding='utf-8', errors='ignore') as f:
                        items = [json.loads(line) for line in f if line.strip()]
                else:
                    with open(fp, encoding='utf-8', errors='ignore') as f:
                        items = json.load(f)
                # Convert to common format
                for item in items:
                    if 'instruction' in item and 'output' in item:
                        pairs.append({
                            'prompt': item['instruction'],
                            'target_code': item.get('initial_response', '') + '\n' + item['output']
                            if 'initial_response' in item else item['output']
                        })
                    elif 'prompt' in item and 'target_code' in item:
                        pairs.append(item)
                print(f"  Merged {fp}: {len(items)} entries")
            except Exception as e:
                print(f"  Error merging {fp}: {e}")

    random.shuffle(pairs)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=1, ensure_ascii=False)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n  Saved {len(pairs)} samples to {output_path} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
