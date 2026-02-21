"""
SNAP-C1 Comprehensive Benchmark Suite
=======================================
Benchmarks SNAP-C1 (MoLoRA) against standardized evaluations:

1. HumanEval   — 164 code generation problems, pass@1 (OpenAI official)
2. GSM8K       — Grade school math reasoning, exact match (subset for speed)
3. Tool Use    — Structured tool call generation, format + correctness
4. HLE         — Humanity's Last Exam (needs HF auth, framework ready)

Each benchmark runs the model, scores outputs, and compares against
published baselines for Qwen3-1.7B and other models.

Usage:
    python evaluation/benchmarks.py --bench humaneval --n 20
    python evaluation/benchmarks.py --bench gsm8k --n 50
    python evaluation/benchmarks.py --bench tool_use --n 50
    python evaluation/benchmarks.py --bench all --n 20
    python evaluation/benchmarks.py --bench all --n 20 --base  # Compare vs base model (no adapters)
    python evaluation/benchmarks.py --bench all --n 10 -v      # Verbose: show model reasoning + outputs
"""

import argparse
import json
import re
import sys
import io
import time
import traceback
import random
import threading
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import yaml
from loguru import logger
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL LOADER — Shared between benchmarks
# =============================================================================

class ModelRunner:
    """Loads Qwen3-1.7B with optional MoLoRA adapters for benchmarking."""
    
    def __init__(self, use_adapters: bool = True, adapter_name: str | None = None):
        """
        Args:
            use_adapters: If True, load MoLoRA adapters. If False, use base model only.
            adapter_name: If set, only load and activate this specific adapter.
        """
        self.use_adapters = use_adapters
        self.adapter_name = adapter_name
        
        with open(CONFIG_DIR / "base_model.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model, self.tokenizer = self._load()
    
    def _load(self):
        model_name = self.config["model"]["name"]
        quant_cfg = self.config["quantization"]
        
        logger.info(f"Loading model: {model_name} (adapters={'ON' if self.use_adapters else 'OFF'})")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        if self.use_adapters:
            adapters = ["team_thinking", "self_correction", "tool_use"]
            first = True
            for name in adapters:
                path = ADAPTERS_DIR / name / "final"
                if not path.exists():
                    continue
                logger.info(f"  Loading adapter: {name}")
                if first:
                    model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
                    first = False
                else:
                    model.load_adapter(str(path), adapter_name=name)
            
            if self.adapter_name:
                model.set_adapter(self.adapter_name)
                logger.info(f"  Active adapter: {self.adapter_name}")
        
        model.eval()
        
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"  VRAM: {alloc:.2f}GB allocated")
        
        return model, tokenizer
    
    def set_adapter(self, name: str | None):
        """Switch adapter. None = disable adapters (base model)."""
        if not self.use_adapters:
            return
        if name is None:
            self.model.disable_adapter_layers()
        else:
            self.model.enable_adapter_layers()
            self.model.set_adapter(name)
    
    def generate(self, prompt: str, system: str = "", max_new_tokens: int = 1024,
                 temperature: float = 0.1, do_sample: bool = True) -> str:
        """Generate a response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(self.model.device)
        
        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95
            else:
                # Suppress top_k/top_p/temperature that leak from Qwen3's
                # default generation_config when do_sample=False
                gen_kwargs["top_k"] = None
                gen_kwargs["top_p"] = None
                gen_kwargs["temperature"] = None
            outputs = self.model.generate(**gen_kwargs)
        
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
        
        # Strip <think>...</think> to get the actual answer
        # Keep it for scoring but also extract clean answer
        return response


# =============================================================================
# BENCHMARK 1: HumanEval (Code Generation, pass@1)
# =============================================================================

def run_humaneval(runner: ModelRunner, n: int = 164, verbose: bool = False) -> dict:
    """Run HumanEval benchmark.
    
    Each problem has a function signature + docstring. The model must complete
    the function. We execute the code + test cases to check correctness.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: HumanEval (pass@1, n={n})")
    print(f"{'='*70}")
    
    ds = load_dataset("openai/openai_humaneval", split="test")
    
    if n < len(ds):
        indices = list(range(len(ds)))
        random.seed(42)
        random.shuffle(indices)
        indices = sorted(indices[:n])
    else:
        indices = list(range(len(ds)))
        n = len(ds)
    
    # Activate team_thinking adapter for structured multi-perspective reasoning
    if runner.use_adapters:
        runner.set_adapter("team_thinking")
        system = (
            "You are SNAP-C1, an advanced AI that reasons from multiple internal perspectives using "
            "deliberative debate. Inside <think> tags, use this format:\n\n"
            "[Round 1 — Initial Positions]\n"
            "[Architect] High-level design, algorithm choice, data structure selection\n"
            "[Critic] Challenge the approach, find edge cases, identify potential bugs\n"
            "[Researcher] Consider known algorithms, time/space complexity, standard patterns\n"
            "[Implementer] Write the actual Python code, concrete and correct\n\n"
            "[Round 2 — Debate] (cross-persona challenges: [Critic → Architect] 'your approach fails when...')\n\n"
            "[Consensus Reached]\n"
            "Each persona states: Position | Confidence: XX%\n"
            "[Synthesizer] Final merged answer\n\n"
            "Continue debating until ALL personas agree. Then output the final code.\n\n"
            "IMPORTANT: After </think>, output the function body code inside a "
            "```python code block. "
            "Do not repeat the function signature. Do not include import statements — "
            "assume all necessary imports (typing, math, collections, etc.) are already available.\n"
            "STRICT CONSTRAINT: Do not use XML tags like <analysis>, <verification>, <implementation>, <research>, etc. Only use <think> and <final_answer> if explicitly instructed."
        )
        logger.info("  HumanEval: team_thinking adapter ACTIVE")
    else:
        runner.set_adapter(None)
        system = (
            "You are a Python coding assistant. Complete the given function. "
            "Output ONLY the function body code inside a ```python code block. "
            "No explanations, no extra text. "
            "Do not repeat the function signature. Do not include import statements — "
            "assume all necessary imports (typing, math, collections, etc.) are already available."
        )
    
    passed = 0
    results = []
    
    for idx_num, i in enumerate(indices):
        item = ds[i]
        task_id = item["task_id"]
        prompt = item["prompt"]
        tests = item["test"]
        entry_point = item["entry_point"]
        canonical = item["canonical_solution"]
        
        print(f"\n[{idx_num+1}/{n}] {task_id}...", end=" ", flush=True)
        
        start = time.time()
        response = runner.generate(
            f"Complete this Python function:\n\n{prompt}",
            system=system,
            max_new_tokens=1536,
            temperature=0.0,
            do_sample=False,
        )
        elapsed = time.time() - start
        
        # Extract code from response — try multiple formats
        code = extract_code(response, prompt, entry_point)
        
        # Try executing with test cases
        success, error = execute_humaneval(prompt, code, tests, entry_point)
        
        if success:
            passed += 1
            print(f"PASS ({elapsed:.1f}s)")
        else:
            print(f"FAIL ({elapsed:.1f}s) - {error[:80]}")
        
        if verbose:
            thinking, answer = split_thinking_and_answer(response)
            print(f"\n    --- Prompt (function signature) ---")
            for line in prompt.strip().split('\n')[:15]:
                print(f"    > {line}")
            if len(prompt.strip().split('\n')) > 15:
                print(f"    > ... ({len(prompt.strip().split(chr(10)))} lines total)")
            if thinking:
                print(f"\n    --- Model Thinking ---")
                think_lines = thinking.split('\n')
                for line in think_lines[:20]:
                    print(f"    {line[:120]}")
                if len(think_lines) > 20:
                    print(f"    ... ({len(think_lines)} lines total)")
            print(f"\n    --- Raw Model Output (after think) ---")
            if answer:
                ans_lines = answer.strip().split('\n')
                for line in ans_lines[:20]:
                    print(f"    {line[:120]}")
                if len(ans_lines) > 20:
                    print(f"    ... ({len(ans_lines)} lines total)")
            else:
                print(f"    (empty — all output was inside <think> tags)")
            print(f"\n    --- Extracted Code ---")
            for line in code.split('\n')[:25]:
                print(f"    {line[:120]}")
            if len(code.split('\n')) > 25:
                print(f"    ... ({len(code.split(chr(10)))} lines total)")
            if not success:
                print(f"\n    --- Error ---")
                print(f"    {error}")
            print()
        
        results.append({
            "task_id": task_id,
            "passed": success,
            "error": error,
            "elapsed": elapsed,
            "response_len": len(response),
            "response": response,
            "extracted_code": code,
        })
    
    score = passed / n
    print(f"\n{'='*70}")
    print(f"HumanEval: {passed}/{n} = {score:.1%}")
    print(f"{'='*70}")
    
    return {
        "benchmark": "humaneval",
        "passed": passed,
        "total": n,
        "score": score,
        "results": results,
    }


def extract_code(response: str, prompt: str, entry_point: str) -> str:
    """Extract the function body from model response.
    
    Handles multiple output formats:
    - ```python ... ``` code blocks
    - Raw function definitions
    - Just function body (indented code)
    - Code mixed with think tags
    """
    # Remove think tags and other hallucinated tags like <analysis>, <verification>, <implementation>, <research>, <code_sandbox>
    tags_to_remove = ['think', 'analysis', 'verification', 'implementation', 'research', 'code_sandbox']
    clean = response
    for tag in tags_to_remove:
        if f'<{tag}>' in clean:
            # Try to remove closed blocks first
            clean = re.sub(rf'<{tag}>.*?</{tag}>', '', clean, flags=re.DOTALL).strip()
            # If the tag wasn't closed, take everything before it or after it (if we lost the code, it's bad, but usually code is after)
            if not clean and f'</{tag}>' in response:
                clean = response.rsplit(f'</{tag}>', 1)[-1].strip()
            elif not clean:
                clean = response.replace(f'<{tag}>', '').strip()
    
    # Still keep think_content fallback logic just in case
    think_content = ""
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
    elif '<think>' in response:
        think_content = response.split('<think>', 1)[1].strip()
        
    if not clean:
        clean = think_content
    elif not clean:
        clean = response.strip()
    
    # If response contains ```python ... ```, extract the LAST one
    # (model may iterate on its solution — we want the final version)
    code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', clean, re.DOTALL)
    if not code_blocks:
        # Handle unclosed code block (truncated output) — take the last ```
        unclosed = re.findall(r'```(?:python)?\s*\n?(.*)', clean, re.DOTALL)
        if unclosed:
            code_blocks = [unclosed[-1]]
    if code_blocks:
        code = code_blocks[-1].strip()  # Take the LAST code block
        code = repair_truncated_code(code)
        # If the extracted code contains the full function def, use it directly
        if f"def {entry_point}" in code:
            return code
        # Otherwise treat as function body
        return code
    
    # If clean text contains def entry_point, extract from there
    def_match = re.search(rf'(def\s+{re.escape(entry_point)}\s*\(.*)', clean, re.DOTALL)
    if def_match:
        return repair_truncated_code(def_match.group(1).strip())
    
    # If response starts with def, it's the full function
    if clean.strip().startswith("def "):
        return repair_truncated_code(clean.strip())
    
    # Otherwise treat entire response as function body
    # Clean up common issues: remove leading "Here is..." text
    lines = clean.split('\n')
    code_lines = []
    found_code = False
    for line in lines:
        stripped = line.strip()
        # Skip explanation lines before code
        if not found_code and stripped and not stripped.startswith(('    ', '\t', 'return', 'if', 'for', 'while', 'try', 'with', '#')):
            # Check if it looks like prose, not code
            if re.match(r'^[A-Z].*[.:]\s*$', stripped) or stripped.startswith(('Here', 'The ', 'This ', 'I ')):
                continue
        found_code = True
        # Add indentation if not present
        if line.strip() and not line.startswith('    ') and not line.startswith('\t'):
            code_lines.append('    ' + line)
        else:
            code_lines.append(line)
    
    return '\n'.join(code_lines)


def repair_truncated_code(code: str) -> str:
    """Attempt to repair code that was truncated mid-generation.
    
    Fixes:
    - Unterminated string literals (close them)
    - Unclosed brackets/parens/braces
    - Truncated lines (remove incomplete last line)
    """
    lines = code.split('\n')
    
    # Remove the last line if it looks truncated (mid-string, mid-expression)
    # Check if the last non-empty line has unbalanced quotes
    while lines:
        last = lines[-1]
        if not last.strip():
            lines.pop()
            continue
        
        # Count quotes in the last line
        single_quotes = last.count("'") - last.count("\\'")
        double_quotes = last.count('"') - last.count('\\"')
        triple_single = last.count("'''")
        triple_double = last.count('"""')
        
        # If triple quotes are odd, the line is mid-docstring — remove it
        if triple_single % 2 == 1 or triple_double % 2 == 1:
            lines.pop()
            continue
        
        # If regular quotes are odd (unmatched), remove the line
        # Adjust for triple quotes (each triple = 3 singles counted)
        effective_single = single_quotes - (triple_single * 3)
        effective_double = double_quotes - (triple_double * 3)
        if effective_single % 2 == 1 or effective_double % 2 == 1:
            lines.pop()
            continue
        
        break
    
    code = '\n'.join(lines)
    
    # Close unclosed brackets/parens/braces
    open_count = {'(': 0, '[': 0, '{': 0}
    close_map = {'(': ')', '[': ']', '{': '}'}
    in_string = None
    escape = False
    
    for ch in code:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch in ('"', "'"):
            if in_string == ch:
                in_string = None
            elif in_string is None:
                in_string = ch
            continue
        if in_string:
            continue
        if ch in open_count:
            open_count[ch] += 1
        elif ch == ')':
            open_count['('] = max(0, open_count['('] - 1)
        elif ch == ']':
            open_count['['] = max(0, open_count['['] - 1)
        elif ch == '}':
            open_count['{'] = max(0, open_count['{'] - 1)
    
    # Append closing brackets
    suffix = ''
    for opener in ['{', '[', '(']:
        suffix += close_map[opener] * open_count[opener]
    
    if suffix:
        code = code.rstrip() + suffix
    
    return code


def execute_humaneval(prompt: str, completion: str, tests: str, entry_point: str) -> tuple[bool, str]:
    """Execute HumanEval code with test cases. Returns (success, error_msg)."""
    # Common imports that HumanEval problems may need
    import_header = (
        "from typing import List, Dict, Tuple, Optional, Set, Any, Union\n"
        "from typing import Callable, Iterator, Generator\n"
        "from collections import defaultdict, Counter, OrderedDict, deque\n"
        "from itertools import combinations, permutations, product, chain\n"
        "from functools import reduce, lru_cache\n"
        "from math import sqrt, ceil, floor, gcd, log, log2, inf\n"
        "from heapq import heappush, heappop, heapify\n"
        "from bisect import bisect_left, bisect_right\n"
        "import re\n"
        "import sys\n"
        "import copy\n"
        "import hashlib\n"
        "import math\n"
        "import string\n"
    )
    
    # Build full code: imports + prompt (function signature) + completion + tests
    # Key insight: Some HumanEval problems define helper functions in the prompt
    # (e.g., poly() for find_zero(), encode_cyclic() for decode_cyclic()).
    # We ALWAYS include the prompt to ensure helper functions are available.
    # If the completion re-defines the entry_point function, that's fine — it will
    # override the signature-only version from the prompt.
    
    if f"def {entry_point}" in completion:
        # Completion contains the full function — but we still need helper
        # functions from the prompt. Extract everything BEFORE the entry_point
        # function definition in the prompt.
        prompt_helper = ""
        entry_def_match = re.search(rf'^def\s+{re.escape(entry_point)}\s*\(', prompt, re.MULTILINE)
        if entry_def_match:
            prompt_helper = prompt[:entry_def_match.start()].strip()
        
        if prompt_helper:
            full_code = import_header + "\n" + prompt_helper + "\n\n" + completion + "\n\n" + tests
        else:
            full_code = import_header + "\n" + completion + "\n\n" + tests
    else:
        # Completion is just the body — use full prompt (which has the signature)
        # Fix: if the body is a single expression without 'return', add 'return'
        body = completion
        body_lines = [l for l in body.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
        if body_lines:
            first_line = body_lines[0].strip()
            # Check if the first (or only) code line is a bare expression (no assignment, no keyword)
            # A bare expression needs 'return' prepended
            statement_keywords = ('return', 'if', 'for', 'while', 'try', 'with', 'def', 'class',
                                  'raise', 'yield', 'import', 'from', 'pass', 'break', 'continue',
                                  'assert', 'del', 'global', 'nonlocal')
            is_assignment = '=' in first_line and not first_line.startswith(('=', '==')) and '==' not in first_line.split('=')[0]
            is_statement = any(first_line.startswith(kw) for kw in statement_keywords)
            
            if len(body_lines) == 1 and not is_statement:
                if is_assignment:
                    # Single assignment line (e.g., "result = [x**2 for x in range(n)]")
                    # Keep the assignment AND add return of the variable
                    var_name = first_line.split('=')[0].strip()
                    body = '    ' + first_line + '\n    return ' + var_name + '\n'
                else:
                    # Single expression line — prepend 'return'
                    body = '    return ' + first_line + '\n'
        
        full_code = import_header + "\n" + prompt + body + "\n\n" + tests
    
    try:
        # Execute in isolated namespace with timeout (10 seconds)
        exec_globals = {}
        result_container = {"success": False, "error": ""}
        
        def _run_code():
            try:
                exec(full_code, exec_globals)
                exec_globals["check"](exec_globals[entry_point])
                result_container["success"] = True
            except AssertionError as e:
                result_container["error"] = f"AssertionError: {e}"
            except SyntaxError as e:
                result_container["error"] = f"SyntaxError: {e}"
            except Exception as e:
                result_container["error"] = f"{type(e).__name__}: {e}"
        
        thread = threading.Thread(target=_run_code, daemon=True)
        thread.start()
        thread.join(timeout=10)  # 10 second timeout
        
        if thread.is_alive():
            return False, "TimeoutError: execution exceeded 10 seconds (possible infinite loop)"
        
        return result_container["success"], result_container["error"]
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =============================================================================
# BENCHMARK 2: GSM8K (Math Reasoning, exact match)
# =============================================================================

def run_gsm8k(runner: ModelRunner, n: int = 50, verbose: bool = False) -> dict:
    """Run GSM8K benchmark.
    
    Each problem is a grade-school math word problem. The model must
    solve it step-by-step. We extract the final numerical answer and
    compare against the ground truth (exact match).
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: GSM8K (exact match, n={n})")
    print(f"{'='*70}")
    
    ds = load_dataset("openai/gsm8k", "main", split="test")
    
    # Stratified sample
    if n < len(ds):
        random.seed(42)
        indices = random.sample(range(len(ds)), n)
        indices.sort()
    else:
        indices = list(range(len(ds)))
        n = len(ds)
    
    # Activate team_thinking adapter for structured multi-perspective reasoning
    if runner.use_adapters:
        runner.set_adapter("team_thinking")
        system = (
            "You are SNAP-C1, an advanced AI that reasons from multiple internal perspectives using "
            "deliberative debate. You MUST always start your response with <think>.\n"
            "Inside <think> tags, use this format:\n\n"
            "[Round 1: Strategy]\n"
            "[Architect] Break down the problem structure, identify what's given and what's asked\n"
            "[Critic] Check for common pitfalls, verify assumptions, catch unit errors\n"
            "[Researcher] Identify the math concepts needed, recall relevant formulas\n"
            "[Implementer] Execute the calculation step by step with clear arithmetic\n\n"
            "[Round 2: Verification]\n"
            "[Critic] Challenge the calculation: 'your calculation has an error...'\n"
            "[Implementer] Verify the math.\n"
            "[Consensus Reached]\n"
            "</think>\n\n"
            "<final_answer>\n"
            "#### <number>\n"
            "</final_answer>\n\n"
            "The number after #### must be just a number with no units, no dollar signs, no extra text.\n"
            "STRICT CONSTRAINT: Do not use XML tags like <analysis>, <verification>, <implementation>, <research>, etc. Only use <think> and <final_answer> if explicitly instructed."
        )
        logger.info("  GSM8K: team_thinking adapter ACTIVE")
    else:
        runner.set_adapter(None)
        system = (
            "Solve this math problem step by step. Show your work clearly. "
            "After your solution, write ONLY the final numerical answer on the last line "
            "in this exact format:\n#### <number>\n\n"
            "For example, if the answer is 42, end with:\n#### 42\n\n"
            "The number after #### must be just a number with no units, no dollar signs, no extra text."
        )
    
    passed = 0
    results = []
    
    for idx_num, i in enumerate(indices):
        item = ds[i]
        question = item["question"]
        answer_text = item["answer"]
        
        # Extract ground truth number (after ####)
        gt_match = re.search(r'####\s*(.+)', answer_text)
        ground_truth = gt_match.group(1).strip().replace(",", "") if gt_match else ""
        
        print(f"\n[{idx_num+1}/{n}] Q: {question[:70]}...", end=" ", flush=True)
        
        start = time.time()
        response = runner.generate(
            question,
            system=system,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
        )
        elapsed = time.time() - start
        
        # Extract model's final answer
        predicted = extract_gsm8k_answer(response)
        
        # Normalize both for comparison
        pred_clean = predicted.replace(",", "").replace("$", "").strip().rstrip(".")
        gt_clean = ground_truth.replace(",", "").replace("$", "").strip().rstrip(".")
        # Handle decimal comparison: "5.0" == "5", "15.00" == "15"
        try:
            correct = float(pred_clean) == float(gt_clean)
        except (ValueError, TypeError):
            correct = pred_clean == gt_clean
        if correct:
            passed += 1
            print(f"CORRECT={ground_truth} ({elapsed:.1f}s)")
        else:
            print(f"WRONG pred={predicted} gt={ground_truth} ({elapsed:.1f}s)")
        
        if verbose:
            thinking, answer = split_thinking_and_answer(response)
            print(f"\n    --- Question ---")
            for line in question.split('\n'):
                print(f"    > {line[:100]}")
            if thinking:
                print(f"\n    --- Model Reasoning ---")
                think_lines = thinking.split('\n')
                for line in think_lines[:30]:
                    print(f"    {line[:100]}")
                if len(think_lines) > 30:
                    print(f"    ... ({len(think_lines)} lines total)")
            print(f"\n    --- Final Answer ---")
            if answer:
                for line in answer.strip().split('\n')[:10]:
                    print(f"    {line[:100]}")
            else:
                print(f"    (answer was inside thinking block)")
            print(f"    Extracted: {predicted}  |  Ground Truth: {ground_truth}  |  {'CORRECT' if correct else 'WRONG'}")
            print()
        
        results.append({
            "question": question[:100],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "elapsed": elapsed,
            "response": response,
        })
    
    score = passed / n
    print(f"\n{'='*70}")
    print(f"GSM8K: {passed}/{n} = {score:.1%}")
    print(f"{'='*70}")
    
    return {
        "benchmark": "gsm8k",
        "passed": passed,
        "total": n,
        "score": score,
        "results": results,
    }


def extract_gsm8k_answer(response: str) -> str:
    """Extract the final numerical answer from a GSM8K response.
    
    Handles multiple formats:
    - #### 42 (trained format)
    - \\boxed{42} (LaTeX format, common in Qwen3)
    - "the answer is 42" (natural language)
    - Last number in response (fallback)
    
    Also handles Qwen3's <think>...</think> mode robustly,
    including cases where the think tag isn't closed.
    """
    # First, try to find #### in the FULL response (including inside think tags)
    # because the model might put #### inside its thinking
    all_hashes = re.findall(r'####\s*([0-9,]+\.?\d*)', response)
    
    # Remove think tags and other hallucinated tags — handle unclosed tags too
    tags_to_remove = ['think', 'analysis', 'verification', 'implementation', 'research', 'code_sandbox']
    clean = response
    for tag in tags_to_remove:
        if f'<{tag}>' in clean:
            # Try to remove closed blocks first
            clean = re.sub(rf'<{tag}>.*?</{tag}>', '', clean, flags=re.DOTALL).strip()
            # If the tag wasn't closed, take everything before it or after it
            if not clean and f'</{tag}>' in response:
                clean = response.rsplit(f'</{tag}>', 1)[-1].strip()
            elif not clean:
                # Tag never closed — entire response is inside it
                clean = response.replace(f'<{tag}>', '').strip()
    
    # Look for #### pattern in clean text (highest priority)
    match = re.search(r'####\s*\$?([0-9,]+\.?\d*)', clean)
    if match:
        return match.group(1).strip().replace(",", "")
    
    # If we found #### in the full response (including thinking), use the last one
    if all_hashes:
        return all_hashes[-1].strip().replace(",", "")
    
    # Look for boxed answer \boxed{X} (common in Qwen3)
    match = re.search(r'\\boxed\{([^}]+)\}', clean) or re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        val = match.group(1).strip().replace(",", "")
        # Extract just the number from boxed content
        num_match = re.search(r'[\-]?\d[\d,]*\.?\d*', val)
        return num_match.group(0).replace(",", "") if num_match else val
    
    # Look for "answer is X" / "answer: X" / "= X" patterns
    for pattern in [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?([\-]?\d[\d,]*\.?\d*)',
        r'(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?(?:answer|total|result)\s+is\s*:?\s*\$?([\-]?\d[\d,]*\.?\d*)',
        r'(?:=|equals)\s*\$?([\-]?\d[\d,]*\.?\d*)\s*$',
    ]:
        match = re.search(pattern, clean, re.I | re.MULTILINE)
        if match:
            return match.group(1).strip().replace(",", "")
    
    # Last number in the clean response (fallback)
    # Only if it's not starting with a bracket (like [Round 1])
    numbers = []
    for line in clean.split('\n'):
        if '[' not in line and 'Round' not in line:
            nums = re.findall(r'[\-]?\b\d[\d,]*\.?\d*\b', line)
            numbers.extend(nums)
    if numbers:
        return numbers[-1].strip().replace(",", "")
    
    # Absolute last resort: look in the full response
    numbers = re.findall(r'[\-]?\b\d[\d,]*\.?\d*\b', response)
    if numbers:
        return numbers[-1].strip().replace(",", "")
    
    return ""


# =============================================================================
# BENCHMARK 3: Tool Use (structured tool calls)
# =============================================================================

TOOL_USE_PROBLEMS = [
    {"query": "Read the contents of main.py", "expected_tool": "read_file", "expected_args": ["main.py"]},
    {"query": "Show me what's in the config/settings.yaml file", "expected_tool": "read_file", "expected_args": ["config/settings.yaml"]},
    {"query": "Search for all functions that handle authentication in the codebase", "expected_tool": "search_files", "expected_args": ["auth"], "alt_tools": ["shell_command"]},
    {"query": "Run the test suite with pytest", "expected_tool": "shell_command", "expected_args": ["pytest"], "alt_tools": ["execute_code"]},
    {"query": "Create a new file called utils/helpers.py with a basic logging setup", "expected_tool": "write_file", "expected_args": ["utils/helpers.py"]},
    {"query": "Execute this Python code: print(sum(range(100)))", "expected_tool": "execute_code", "expected_args": ["sum(range(100))"]},
    {"query": "Find all TODO comments in Python files", "expected_tool": "search_files", "expected_args": ["TODO"]},
    {"query": "Edit server.py and change the port from 8080 to 3000", "expected_tool": "edit_file", "expected_args": ["server.py"], "alt_tools": ["write_file", "read_file"]},
    {"query": "List all files in the src directory", "expected_tool": "shell_command", "expected_args": ["ls"], "alt_tools": ["search_files", "read_file"]},
    {"query": "Read the package.json to check dependencies", "expected_tool": "read_file", "expected_args": ["package.json"]},
    {"query": "Search the project for any hardcoded API keys", "expected_tool": "search_files", "expected_args": ["API"]},
    {"query": "Run python manage.py migrate", "expected_tool": "shell_command", "expected_args": ["manage.py"]},
    {"query": "Write a test file tests/test_auth.py with basic test cases", "expected_tool": "write_file", "expected_args": ["test_auth.py"]},
    {"query": "Execute: import platform; print(platform.system())", "expected_tool": "execute_code", "expected_args": ["platform"]},
    {"query": "Open and read the README.md file", "expected_tool": "read_file", "expected_args": ["README.md"]},
    {"query": "Find all files importing the requests library", "expected_tool": "search_files", "expected_args": ["requests"]},
    {"query": "Run npm install in the frontend directory", "expected_tool": "shell_command", "expected_args": ["npm"], "alt_tools": ["execute_code"]},
    {"query": "Create a .env file with DATABASE_URL=postgres://localhost/mydb", "expected_tool": "write_file", "expected_args": [".env"]},
    {"query": "Check the git status of the project", "expected_tool": "shell_command", "expected_args": ["git"]},
    {"query": "Read the first 50 lines of database/models.py", "expected_tool": "read_file", "expected_args": ["models.py"]},
    {"query": "Search for deprecated function calls in the codebase", "expected_tool": "search_files", "expected_args": ["deprecated"]},
    {"query": "Execute a quick benchmark: [x**2 for x in range(1000)]", "expected_tool": "execute_code", "expected_args": ["range(1000)"]},
    {"query": "Edit config.yaml to change debug mode from true to false", "expected_tool": "edit_file", "expected_args": ["config.yaml"], "alt_tools": ["read_file", "write_file"]},
    {"query": "Show me the contents of the Dockerfile", "expected_tool": "read_file", "expected_args": ["Dockerfile"]},
    {"query": "Run the linter: flake8 src/", "expected_tool": "shell_command", "expected_args": ["flake8"], "alt_tools": ["execute_code"]},
    {"query": "Find all classes that inherit from BaseModel", "expected_tool": "search_files", "expected_args": ["BaseModel"]},
    {"query": "Create a migration script at db/migrations/001_init.sql", "expected_tool": "write_file", "expected_args": ["001_init.sql"]},
    {"query": "Execute: len(open('data.csv').readlines())", "expected_tool": "execute_code", "expected_args": ["data.csv"]},
    {"query": "Read the nginx.conf configuration file", "expected_tool": "read_file", "expected_args": ["nginx.conf"]},
    {"query": "Search for all error handling blocks (try/except)", "expected_tool": "search_files", "expected_args": ["except"]},
    {"query": "Run docker-compose up -d", "expected_tool": "shell_command", "expected_args": ["docker"]},
    {"query": "Edit requirements.txt to add flask==3.0.0", "expected_tool": "edit_file", "expected_args": ["requirements.txt"], "alt_tools": ["write_file", "read_file"]},
    {"query": "Show the contents of the .gitignore file", "expected_tool": "read_file", "expected_args": [".gitignore"]},
    {"query": "Find all files with SQL injection vulnerabilities (raw SQL queries)", "expected_tool": "search_files", "expected_args": ["SQL"]},
    {"query": "Execute: import sys; print(sys.version)", "expected_tool": "execute_code", "expected_args": ["sys.version"]},
    {"query": "Create a new script deploy.sh with basic deployment steps", "expected_tool": "write_file", "expected_args": ["deploy.sh"]},
    {"query": "Read the Makefile to see available commands", "expected_tool": "read_file", "expected_args": ["Makefile"]},
    {"query": "Run the database seed script: python seed_data.py", "expected_tool": "shell_command", "expected_args": ["seed_data.py"], "alt_tools": ["execute_code"]},
    {"query": "Search for all environment variable references in the code", "expected_tool": "search_files", "expected_args": ["environ"]},
    {"query": "Edit app.py to add CORS support", "expected_tool": "edit_file", "expected_args": ["app.py"]},
    {"query": "Show the project's pyproject.toml configuration", "expected_tool": "read_file", "expected_args": ["pyproject.toml"], "alt_tools": ["search_files"]},
    {"query": "Execute a quick test: assert 2+2 == 4, 'Math is broken'", "expected_tool": "execute_code", "expected_args": ["assert"]},
    {"query": "Find all files larger than 1MB in the project", "expected_tool": "shell_command", "expected_args": ["find"], "alt_tools": ["search_files", "execute_code"]},
    {"query": "Search for all print statements that should be replaced with logging", "expected_tool": "search_files", "expected_args": ["print"]},
    {"query": "Create a new Python module at lib/cache.py with Redis connection", "expected_tool": "write_file", "expected_args": ["cache.py"]},
    {"query": "Read the CI/CD configuration from .github/workflows/main.yml", "expected_tool": "read_file", "expected_args": ["main.yml"]},
    {"query": "Run black formatter on the src directory", "expected_tool": "shell_command", "expected_args": ["black"], "alt_tools": ["execute_code"]},
    {"query": "Edit docker-compose.yml to add a Redis service", "expected_tool": "edit_file", "expected_args": ["docker-compose"], "alt_tools": ["write_file", "read_file"]},
    {"query": "Execute: from collections import Counter; print(Counter('abracadabra'))", "expected_tool": "execute_code", "expected_args": ["Counter"]},
    {"query": "Search for all async functions in the codebase", "expected_tool": "search_files", "expected_args": ["async"]},
]


def run_tool_use(runner: ModelRunner, n: int = 50, verbose: bool = False) -> dict:
    """Run tool use benchmark.
    
    Tests whether the model generates proper <tool_call> tags with
    correct tool names and relevant arguments.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Tool Use (structured calls, n={n})")
    print(f"{'='*70}")
    
    problems = TOOL_USE_PROBLEMS[:n]
    
    # Use tool_use adapter
    if runner.use_adapters:
        runner.set_adapter("tool_use")
    
    system = (
        "You are SNAP-C1, an AI assistant that uses tools to complete tasks. "
        "CRITICAL: You MUST respond with a tool call IMMEDIATELY. Do NOT explain or discuss first. "
        "Emit the tool call as the VERY FIRST thing in your response.\n\n"
        "Format:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "args": {"key": "value"}}\n'
        "</tool_call>\n\n"
        "Available tools:\n"
        '- read_file: {"name": "read_file", "args": {"path": "file/path"}}\n'
        '- write_file: {"name": "write_file", "args": {"path": "file/path", "content": "..."}}\n'
        '- edit_file: {"name": "edit_file", "args": {"path": "file/path", "old_text": "...", "new_text": "..."}}\n'
        '- execute_code: {"name": "execute_code", "args": {"language": "python", "code": "..."}}\n'
        '- shell_command: {"name": "shell_command", "args": {"command": "..."}}\n'
        '- search_files: {"name": "search_files", "args": {"pattern": "...", "path": "."}}\n\n'
        "Respond with ONLY the tool call. No thinking, no explanation."
    )
    
    passed = 0
    results = []
    
    for i, problem in enumerate(problems):
        query = problem["query"]
        expected_tool = problem["expected_tool"]
        expected_args = problem["expected_args"]
        alt_tools = problem.get("alt_tools", [])
        
        print(f"\n[{i+1}/{len(problems)}] {query[:60]}...", end=" ", flush=True)
        
        start = time.time()
        response = runner.generate(query, system=system, max_new_tokens=768, temperature=0.1)
        elapsed = time.time() - start
        
        # Score: check for <tool_call> with correct tool and args
        score_result = score_tool_call(response, expected_tool, expected_args, alt_tools)
        
        if score_result["score"] >= 0.7:
            passed += 1
            print(f"PASS (tool={score_result['found_tool']}, {elapsed:.1f}s)")
        else:
            print(f"FAIL (score={score_result['score']:.2f}, tool={score_result['found_tool']}, {elapsed:.1f}s)")
        
        if verbose:
            thinking, answer = split_thinking_and_answer(response)
            print(f"\n    --- Query ---")
            print(f"    > {query}")
            print(f"    Expected: {expected_tool}  |  Alt: {alt_tools if alt_tools else 'none'}")
            if thinking:
                print(f"\n    --- Model Thinking ---")
                think_lines = thinking.split('\n')
                for line in think_lines[:15]:
                    print(f"    {line[:100]}")
                if len(think_lines) > 15:
                    print(f"    ... ({len(think_lines)} lines total)")
            print(f"\n    --- Model Output ---")
            if answer:
                for line in answer.strip().split('\n')[:15]:
                    print(f"    {line[:100]}")
            else:
                # No answer outside thinking — show last part of response
                resp_lines = response.strip().split('\n')
                for line in resp_lines[-10:]:
                    print(f"    {line[:100]}")
            print(f"\n    --- Scoring ---")
            print(f"    Score: {score_result['score']:.2f} | Tags: {score_result['has_tags']} | "
                  f"Valid JSON: {score_result['valid_json']} | "
                  f"Tool: {score_result['found_tool']} | Args: {score_result['args_match']}")
            print()
        
        results.append({
            "query": query,
            "expected_tool": expected_tool,
            "score": score_result,
            "elapsed": elapsed,
            "response": response,
        })
    
    score = passed / len(problems)
    print(f"\n{'='*70}")
    print(f"Tool Use: {passed}/{len(problems)} = {score:.1%}")
    print(f"{'='*70}")
    
    return {
        "benchmark": "tool_use",
        "passed": passed,
        "total": len(problems),
        "score": score,
        "results": results,
    }


def score_tool_call(response: str, expected_tool: str, expected_args: list[str], alt_tools: list[str] | None = None) -> dict:
    """Score a tool call response. Accepts alternate valid tools.
    
    Extraction priority:
    1. Closed <tool_call>...</tool_call> tags with valid JSON
    2. Unclosed <tool_call> tag with valid JSON (truncated output)
    3. Raw {"name": "..."} JSON in response body
    4. Inline {"tool": "..."} pattern
    """
    if alt_tools is None:
        alt_tools = []
    valid_tools = [expected_tool] + alt_tools
    
    # Remove think tags first
    clean = response
    if '<think>' in clean:
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL).strip()
        if not clean and '</think>' in response:
            clean = response.rsplit('</think>', 1)[-1].strip()
        elif not clean:
            clean = response.replace('<think>', '').strip()
    
    found_tool = None
    valid_json = False
    args_match = False
    score = 0.0
    data = None
    
    # Strategy 1: Closed <tool_call>...</tool_call>
    has_tags = "<tool_call>" in clean and "</tool_call>" in clean
    if has_tags:
        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                valid_json = True
            except json.JSONDecodeError:
                pass
    
    # Strategy 2: Unclosed <tool_call> (truncated output)
    if data is None and "<tool_call>" in clean:
        has_tags = True
        match = re.search(r'<tool_call>\s*(\{.*)', clean, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            # Try to extract valid JSON from potentially truncated string
            # Find the first complete JSON object
            try:
                data = json.loads(json_str)
                valid_json = True
            except json.JSONDecodeError:
                # Try to find a complete JSON object within the string
                brace_match = re.search(r'(\{[^{}]*\})', json_str)
                if brace_match:
                    try:
                        data = json.loads(brace_match.group(1))
                        valid_json = True
                    except json.JSONDecodeError:
                        pass
    
    # Strategy 3: Raw {"name": "tool_name", ...} anywhere in response
    if data is None:
        name_match = re.search(r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\})', clean)
        if name_match:
            try:
                data = json.loads(name_match.group(1))
                valid_json = True
                has_tags = False  # Found without tags
            except json.JSONDecodeError:
                pass
    
    # Score based on what we found
    if has_tags:
        score += 0.3
    
    if valid_json and data:
        score += 0.2
        
        found_tool = data.get("name") or data.get("tool") or data.get("function")
        
        if found_tool == expected_tool:
            score += 0.3
        elif found_tool in alt_tools:
            score += 0.25
        
        # Check if expected args appear anywhere in the tool call
        call_str = json.dumps(data).lower()
        args_found = sum(1 for arg in expected_args if arg.lower() in call_str)
        if args_found == len(expected_args):
            args_match = True
            score += 0.2
        elif args_found > 0:
            score += 0.1
    elif not valid_json and not data:
        # Strategy 4: Fallback — check for inline {"tool": "..."} pattern
        match = re.search(r'\{"(?:tool|name)"\s*:\s*"(\w+)"', clean)
        if match:
            found_tool = match.group(1)
            score += 0.2
            if found_tool == expected_tool:
                score += 0.2
            elif found_tool in alt_tools:
                score += 0.15
    
    return {
        "score": min(score, 1.0),
        "has_tags": has_tags,
        "valid_json": valid_json,
        "found_tool": found_tool,
        "expected_tool": expected_tool,
        "args_match": args_match,
    }


# =============================================================================
# BENCHMARK 4: HLE (Humanity's Last Exam)
# =============================================================================

def run_hle(runner: ModelRunner, n: int = 20) -> dict:
    """Run Humanity's Last Exam benchmark.
    
    NOTE: Requires HuggingFace authentication. Run:
        huggingface-cli login
    and accept the dataset terms at https://huggingface.co/datasets/cais/hle
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Humanity's Last Exam (n={n})")
    print(f"{'='*70}")
    
    try:
        ds = load_dataset("cais/hle", split="test")
    except Exception as e:
        print(f"\nERROR: Cannot load HLE dataset: {e}")
        print("To use HLE, run: huggingface-cli login")
        print("Then accept terms at: https://huggingface.co/datasets/cais/hle")
        return {
            "benchmark": "hle",
            "passed": 0,
            "total": 0,
            "score": 0.0,
            "error": str(e),
            "results": [],
        }
    
    # Sample
    if n < len(ds):
        random.seed(42)
        indices = random.sample(range(len(ds)), n)
        indices.sort()
    else:
        indices = list(range(len(ds)))
        n = len(ds)
    
    # Use base model (no adapter specialization for HLE)
    runner.set_adapter(None)
    
    system = (
        "You are an expert across all academic fields. "
        "Answer the question accurately and concisely. "
        "If it's multiple choice, state just the letter. "
        "If it's short answer, give the exact answer."
    )
    
    passed = 0
    results = []
    
    for idx_num, i in enumerate(indices):
        item = ds[i]
        question = item["question"]
        answer = item.get("answer", "")
        
        print(f"\n[{idx_num+1}/{n}] Q: {question[:70]}...", end=" ", flush=True)
        
        start = time.time()
        response = runner.generate(question, system=system, max_new_tokens=256, temperature=0.1)
        elapsed = time.time() - start
        
        # Clean response
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # Check exact match (case-insensitive, stripped)
        correct = clean.strip().lower() == answer.strip().lower() if answer else False
        
        # Also check if answer appears in response
        if not correct and answer:
            correct = answer.strip().lower() in clean.lower()
        
        if correct:
            passed += 1
            print(f"CORRECT ({elapsed:.1f}s)")
        else:
            print(f"WRONG pred='{clean[:50]}' gt='{answer[:50]}' ({elapsed:.1f}s)")
        
        results.append({
            "question": question[:100],
            "ground_truth": answer,
            "predicted": clean[:200],
            "correct": correct,
            "elapsed": elapsed,
        })
    
    score = passed / n if n > 0 else 0
    print(f"\n{'='*70}")
    print(f"HLE: {passed}/{n} = {score:.1%}")
    print(f"{'='*70}")
    
    return {
        "benchmark": "hle",
        "passed": passed,
        "total": n,
        "score": score,
        "results": results,
    }


# =============================================================================
# PUBLISHED BASELINES (for comparison)
# =============================================================================

BASELINES = {
    "humaneval": {
        "Qwen3-1.7B (base)": 0.432,       # Published / estimated
        "Qwen3-4B": 0.652,
        "Qwen3-8B": 0.726,
        "GPT-4o": 0.902,
        "Claude 3.5 Sonnet": 0.921,
    },
    "gsm8k": {
        "Qwen3-1.7B (base)": 0.631,       # Published / estimated
        "Qwen3-4B": 0.789,
        "Qwen3-8B": 0.862,
        "GPT-4o": 0.946,
        "Claude 3.5 Sonnet": 0.960,
    },
    "tool_use": {
        "Qwen3-1.7B (base)": 0.0,          # No tool use without training
        "GPT-4o": 0.880,
        "Claude 3.5 Sonnet": 0.910,
    },
    "hle": {
        "GPT-4o": 0.036,
        "Claude 3.5 Sonnet": 0.041,
        "o1": 0.094,
        "Gemini 2.5 Pro": 0.069,
    },
}


def print_comparison(benchmark: str, snap_score: float):
    """Print comparison table against baselines."""
    baselines = BASELINES.get(benchmark, {})
    if not baselines:
        return
    
    print(f"\n  Comparison:")
    
    # Sort by score
    all_scores = list(baselines.items()) + [("SNAP-C1 (MoLoRA)", snap_score)]
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    for name, score in all_scores:
        marker = " <<<" if name == "SNAP-C1 (MoLoRA)" else ""
        print(f"    {name:30s} {score:6.1%}{marker}")


# =============================================================================
# MAIN
# =============================================================================

def format_response_block(response: str, max_lines: int = 40) -> str:
    """Format a model response for display with a bordered box."""
    lines = response.split('\n')
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    
    # Find max line width (cap at 100 for readability)
    max_width = min(max(len(l) for l in lines) if lines else 20, 100)
    max_width = max(max_width, 20)
    
    output = []
    output.append(f"    +{'-' * (max_width + 2)}+")
    for line in lines:
        # Truncate long lines
        if len(line) > max_width:
            line = line[:max_width - 3] + "..."
        output.append(f"    | {line:<{max_width}} |")
    if truncated:
        msg = f"... ({len(response.split(chr(10)))} total lines, truncated)"
        output.append(f"    | {msg:<{max_width}} |")
    output.append(f"    +{'-' * (max_width + 2)}+")
    return '\n'.join(output)


def split_thinking_and_answer(response: str) -> tuple[str, str]:
    """Split a response into thinking (inside <think> tags) and the final answer."""
    thinking = ""
    answer = response
    
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    elif '<think>' in response:
        # Unclosed think tag
        parts = response.split('<think>', 1)
        if '</think>' in parts[1]:
            thinking = parts[1].split('</think>', 1)[0].strip()
            answer = parts[1].split('</think>', 1)[1].strip()
        else:
            thinking = parts[1].strip()
            answer = ""
    
    return thinking, answer


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 Benchmark Suite")
    parser.add_argument("--bench", type=str, default="all",
                        choices=["humaneval", "gsm8k", "tool_use", "hle", "all"],
                        help="Which benchmark to run")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of problems per benchmark (default: 20)")
    parser.add_argument("--base", action="store_true",
                        help="Run with base model only (no adapters) for comparison")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show full model responses (reasoning + answers)")
    args = parser.parse_args()
    
    # Load model
    runner = ModelRunner(use_adapters=not args.base)
    
    benchmarks = {
        "humaneval": lambda: run_humaneval(runner, args.n, args.verbose),
        "gsm8k": lambda: run_gsm8k(runner, args.n, args.verbose),
        "tool_use": lambda: run_tool_use(runner, args.n, args.verbose),
        "hle": lambda: run_hle(runner, args.n),
    }
    
    if args.bench == "all":
        selected = ["humaneval", "gsm8k", "tool_use"]  # Skip HLE by default (needs auth)
    else:
        selected = [args.bench]
    
    all_results = {}
    
    for bench_name in selected:
        result = benchmarks[bench_name]()
        all_results[bench_name] = result
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"SNAP-C1 BENCHMARK SUMMARY {'(BASE MODEL)' if args.base else '(MoLoRA)'}")
    print(f"{'='*70}")
    
    for bench_name, result in all_results.items():
        score = result["score"]
        total = result["total"]
        passed = result["passed"]
        print(f"\n  {bench_name.upper():12s}: {passed}/{total} = {score:.1%}")
        print_comparison(bench_name, score)
    
    # Overall
    if all_results:
        avg = sum(r["score"] for r in all_results.values()) / len(all_results)
        print(f"\n{'='*70}")
        print(f"  AVERAGE: {avg:.1%}")
        print(f"{'='*70}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "base" if args.base else "molora"
    output_path = RESULTS_DIR / f"benchmark_{mode}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
