"""
agent_loop.py — SNAP-C1 V4 Agentic Tool-Use Loop
==================================================
Implements the multi-turn agent that allows the V4HyperAssembly model to
iteratively read files, search code, run tests, and apply patches rather than
performing a single-shot generation.

Architecture overview
---------------------
  V4Agent.solve(issue, repo_path)
      │
      ├─ Build prompt from history
      ├─ model.forward([prompt], generate=True)  ← torch.no_grad()
      ├─ decode generated_tokens with tokenizer.bpe.decode()
      ├─ parse_agent_output(text)
      │     ├─ <think>…</think>  → reasoning trace
      │     ├─ <tool_call>{…}</tool_call>  → call ToolRegistry
      │     └─ <patch>…</patch>  → done – return patch
      └─ repeat up to max_turns

Usage (standalone)
------------------
    python v4_core/inference/agent_loop.py \
        --weights v4_core/snapshot_v4_instruct.pt \
        --repo    /path/to/repo \
        --issue   "Fix IndexError in get_last_element when list is empty"
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Project-root bootstrap (works regardless of cwd)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent  # …/RX.AI
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_core.architecture.v4_assembly import V4HyperAssembly  # noqa: E402
from v4_core.data.bpe_tokenizer import HybridTokenDecoder      # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum bytes returned by read_file to avoid flooding the context window.
MAX_READ_BYTES: int = 8_000

#: Maximum characters returned by any tool result.
MAX_TOOL_OUTPUT: int = 2_000

#: Maximum characters kept in a single turn's prompt before truncation.
MAX_PROMPT_CHARS: int = 12_000

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are SNAP-C1, an expert software engineer whose sole job is to fix
    GitHub issues in real code repositories.

    ## Workflow
    1.  Think carefully inside <think>…</think> tags before acting.
    2.  Call exactly one tool per turn using a <tool_call> block.
    3.  Once you understand the bug and have constructed the fix, output the
        complete unified diff inside a <patch>…</patch> block to finish.

    ## Available tools

    | Tool          | Description                                           |
    |---------------|-------------------------------------------------------|
    | read_file     | Read a file by path. Args: {"path": "…"}              |
    | list_dir      | List directory contents. Args: {"path": "…"}          |
    | search_code   | Grep a query recursively. Args: {"query":"…","repo_path":"…"} |
    | run_tests     | Run a test command. Args: {"test_cmd":"…","repo_path":"…"}    |
    | apply_patch   | Apply a unified diff string. Args: {"patch_text":"…","repo_path":"…"} |
    | write_file    | Write content to a file. Args: {"path":"…","content":"…"}     |

    ## Response format examples

    ### Using a tool
    <think>I need to see the function definition first.</think>
    <tool_call>{"tool": "read_file", "args": {"path": "src/utils.py"}}</tool_call>

    ### Submitting the fix
    <think>The off-by-one is on line 42. Here is the minimal patch.</think>
    <patch>
    --- a/src/utils.py
    +++ b/src/utils.py
    @@ -40,7 +40,7 @@
     def get_last_element(lst):
    -    return lst[len(lst)]
    +    return lst[len(lst) - 1]
    </patch>

    Never output both a tool call and a patch in the same turn.
""")


# ===========================================================================
# ToolRegistry
# ===========================================================================

class ToolRegistry:
    """
    Collection of deterministic tools exposed to the agent.

    All methods return a ``(result: str, success: bool)`` tuple so the
    caller always has a uniform interface regardless of whether an
    exception occurred.
    """

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------
    def read_file(self, path: str) -> tuple[str, bool]:
        """
        Read *path* and return up to MAX_READ_BYTES of its content.

        Relative paths are resolved from the current working directory.
        """
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"[ERROR] File not found: {path}", False
            if not p.is_file():
                return f"[ERROR] Not a file: {path}", False

            raw = p.read_bytes()
            text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")

            truncated = len(raw) > MAX_READ_BYTES
            suffix = f"\n... [truncated – {len(raw)} bytes total]" if truncated else ""
            return text + suffix, True

        except PermissionError as exc:
            return f"[ERROR] Permission denied: {exc}", False
        except Exception as exc:
            return f"[ERROR] read_file failed: {exc}", False

    # ------------------------------------------------------------------
    # list_dir
    # ------------------------------------------------------------------
    def list_dir(self, path: str) -> tuple[str, bool]:
        """
        Return an annotated listing of *path* (files marked with sizes,
        directories marked with a trailing '/').
        """
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"[ERROR] Path not found: {path}", False
            if not p.is_dir():
                return f"[ERROR] Not a directory: {path}", False

            entries: list[str] = []
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    entries.append(f"  {child.name}/")
                else:
                    size = child.stat().st_size
                    entries.append(f"  {child.name}  ({size} bytes)")

            if not entries:
                return "(empty directory)", True

            return f"{path}\n" + "\n".join(entries), True

        except Exception as exc:
            return f"[ERROR] list_dir failed: {exc}", False

    # ------------------------------------------------------------------
    # search_code
    # ------------------------------------------------------------------
    def search_code(self, query: str, repo_path: str) -> tuple[str, bool]:
        """
        Recursively grep *query* inside *repo_path*.

        Uses the system ``grep`` (or ``findstr`` on Windows) and returns
        matching lines with ``file:line_number: content`` context,
        truncated to MAX_TOOL_OUTPUT characters.
        """
        try:
            if not Path(repo_path).is_dir():
                return f"[ERROR] repo_path is not a directory: {repo_path}", False

            # Build a cross-platform grep command
            if sys.platform.startswith("win"):
                # /c: treats the argument as a literal string (handles spaces in query);
                # escape any embedded double-quotes in the query first.
                safe_query = query.replace('"', '\\"')
                cmd = (
                    f'findstr /s /n /i /c:"{safe_query}" '
                    f'"{repo_path}\\*.py" '
                    f'"{repo_path}\\*.txt" '
                    f'"{repo_path}\\*.md"'
                )
                shell = True
            else:
                cmd = [
                    "grep", "-rn", "--include=*.py",
                    "--include=*.txt", "--include=*.md",
                    query, repo_path,
                ]
                shell = False

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                shell=shell,
            )

            output = (proc.stdout or "") + (proc.stderr or "")
            if not output.strip():
                return f"(no matches for '{query}' in {repo_path})", True

            # Truncate
            if len(output) > MAX_TOOL_OUTPUT:
                output = output[:MAX_TOOL_OUTPUT] + "\n... [truncated]"
            return output, True

        except subprocess.TimeoutExpired:
            return "[ERROR] search_code timed out after 15 s", False
        except Exception as exc:
            return f"[ERROR] search_code failed: {exc}", False

    # ------------------------------------------------------------------
    # run_tests
    # ------------------------------------------------------------------
    def run_tests(self, test_cmd: str, repo_path: str) -> tuple[str, bool]:
        """
        Execute *test_cmd* inside *repo_path* and return combined
        stdout+stderr, truncated to MAX_TOOL_OUTPUT characters.
        Timeout: 30 s.
        """
        try:
            if not Path(repo_path).is_dir():
                return f"[ERROR] repo_path is not a directory: {repo_path}", False

            proc = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=repo_path,
            )

            combined = proc.stdout + proc.stderr
            if len(combined) > MAX_TOOL_OUTPUT:
                combined = combined[:MAX_TOOL_OUTPUT] + "\n... [truncated]"

            success = proc.returncode == 0
            header = "[PASS]" if success else f"[FAIL rc={proc.returncode}]"
            return f"{header}\n{combined}", success

        except subprocess.TimeoutExpired:
            return "[ERROR] run_tests timed out after 30 s", False
        except Exception as exc:
            return f"[ERROR] run_tests failed: {exc}", False

    # ------------------------------------------------------------------
    # apply_patch
    # ------------------------------------------------------------------
    def apply_patch(self, patch_text: str, repo_path: str) -> tuple[str, bool]:
        """
        Apply *patch_text* (a unified diff string) to *repo_path* using
        ``patch -p1``.  Falls back to ``git apply`` if ``patch`` is
        unavailable.
        """
        import tempfile

        try:
            if not Path(repo_path).is_dir():
                return f"[ERROR] repo_path is not a directory: {repo_path}", False

            # Write patch to a temp file so we don't deal with stdin piping
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(patch_text)
                tmp_path = tmp.name

            try:
                # Prefer `patch -p1`; on Windows it is often absent → catch and fall back
                _patch_missing = False
                try:
                    proc = subprocess.run(
                        ["patch", "-p1", "--input", tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=repo_path,
                    )
                except FileNotFoundError:
                    _patch_missing = True
                    proc = None  # type: ignore[assignment]

                # Fall back to git apply if patch binary missing or returned non-zero
                if _patch_missing or proc.returncode != 0:
                    proc = subprocess.run(
                        ["git", "apply", "--reject", tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=repo_path,
                    )

                output = proc.stdout + proc.stderr
                if len(output) > MAX_TOOL_OUTPUT:
                    output = output[:MAX_TOOL_OUTPUT] + "\n... [truncated]"

                success = proc.returncode == 0
                return output or ("(patch applied cleanly)" if success else "(patch failed)"), success

            finally:
                Path(tmp_path).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return "[ERROR] apply_patch timed out after 15 s", False
        except Exception as exc:
            return f"[ERROR] apply_patch failed: {exc}", False

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------
    def write_file(self, path: str, content: str) -> tuple[str, bool]:
        """
        Write *content* to *path*, creating parent directories as needed.
        """
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"[OK] Wrote {len(content)} chars to {path}", True
        except PermissionError as exc:
            return f"[ERROR] Permission denied: {exc}", False
        except Exception as exc:
            return f"[ERROR] write_file failed: {exc}", False

    # ------------------------------------------------------------------
    # dispatch helper
    # ------------------------------------------------------------------
    def call(self, tool_name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """
        Dispatch *tool_name* with *args* dict.  Returns (result, success).
        """
        dispatch: dict[str, Any] = {
            "read_file":   self.read_file,
            "list_dir":    self.list_dir,
            "search_code": self.search_code,
            "run_tests":   self.run_tests,
            "apply_patch": self.apply_patch,
            "write_file":  self.write_file,
        }

        fn = dispatch.get(tool_name)
        if fn is None:
            return f"[ERROR] Unknown tool: '{tool_name}'", False

        try:
            return fn(**args)
        except TypeError as exc:
            return f"[ERROR] Bad arguments for tool '{tool_name}': {exc}", False


# ===========================================================================
# parse_agent_output
# ===========================================================================

def parse_agent_output(text: str) -> dict[str, Any]:
    """
    Parse the model's raw generation and classify it.

    Recognised patterns (first match wins, top-to-bottom priority):
    1. ``<patch>…</patch>``        → type="patch"
    2. ``<tool_call>{…}</tool_call>``  → type="tool_call"
    3. Anything else               → type="text"

    The ``<think>…</think>`` block is always extracted when present.

    Returns a dict with keys:
        type      : "patch" | "tool_call" | "text"
        tool      : str | None
        args      : dict | None
        content   : str  (patch text, raw text, or empty)
        reasoning : str  (content of <think> block, or "")
    """
    result: dict[str, Any] = {
        "type": "text",
        "tool": None,
        "args": None,
        "content": text.strip(),
        "reasoning": "",
    }

    # --- Extract <think> reasoning -------------------------------------
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        result["reasoning"] = think_match.group(1).strip()

    # --- Check for <patch> --------------------------------------------
    patch_match = re.search(r"<patch>(.*?)</patch>", text, re.DOTALL | re.IGNORECASE)
    if patch_match:
        result["type"] = "patch"
        result["content"] = patch_match.group(1).strip()
        return result

    # --- Check for <tool_call> ----------------------------------------
    tc_match = re.search(
        r"<tool_call>\s*(\{.*\})\s*</tool_call>", text, re.DOTALL | re.IGNORECASE
    )
    if tc_match:
        raw_json = tc_match.group(1).strip()
        try:
            payload = json.loads(raw_json)
            tool_name = str(payload.get("tool", ""))
            tool_args = payload.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            result["type"] = "tool_call"
            result["tool"] = tool_name
            result["args"] = tool_args
            result["content"] = raw_json
        except json.JSONDecodeError as exc:
            logger.warning(f"parse_agent_output: bad JSON in tool_call – {exc}")
            # Fall through to type="text"

    return result


# ===========================================================================
# format_observation
# ===========================================================================

def format_observation(tool_name: str, result: str, success: bool) -> str:
    """
    Format a tool result as a clean observation string for re-injection into
    the model's context.

    The format is intentionally terse: the model only needs the raw content
    plus a status indicator.
    """
    status = "OK" if success else "ERROR"
    separator = "─" * 60
    return (
        f"<observation tool='{tool_name}' status='{status}'>\n"
        f"{separator}\n"
        f"{result}\n"
        f"{separator}\n"
        f"</observation>"
    )


# ===========================================================================
# V4Agent
# ===========================================================================

class V4Agent:
    """
    SNAP-C1 V4 Agentic reasoning loop.

    The agent wraps a ``V4HyperAssembly`` model and a ``HybridTokenDecoder``
    tokenizer and orchestrates the following cycle:

        for turn in range(max_turns):
            prompt  = build_prompt(history)
            output  = model.forward([prompt], generate=True)
            tokens  = output["generated_tokens"]
            text    = tokenizer.bpe.decode(tokens)
            action  = parse_agent_output(text)
            if action["type"] == "patch":
                return patch
            if action["type"] == "tool_call":
                obs = tool_registry.call(...)
                history.append(obs)
        return best_effort_text
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: HybridTokenDecoder,
        device: torch.device | str,
        max_turns: int = 10,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_turns = max_turns
        self.tools = ToolRegistry()

        logger.info(
            f"V4Agent initialised — max_turns={max_turns}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, history: list[dict[str, str]]) -> str:
        """
        Concatenate *history* entries into a single prompt string.

        Each entry must have:
            role    : "system" | "user" | "assistant" | "tool"
            content : str

        The format mirrors a lightweight chat template so fine-tuned weights
        trained with the same template decompose it correctly at inference.
        """
        role_prefix = {
            "system":    "<<SYS>>",
            "user":      "<<USR>>",
            "assistant": "<<AST>>",
            "tool":      "<<OBS>>",
        }

        parts: list[str] = []
        for entry in history:
            role = entry.get("role", "user")
            prefix = role_prefix.get(role, "<<USR>>")
            content = entry.get("content", "").strip()
            parts.append(f"{prefix}\n{content}")

        # Close with the assistant turn marker so the model continues from here
        parts.append("<<AST>>")

        full_prompt = "\n\n".join(parts)

        # Guard: hard-truncate to MAX_PROMPT_CHARS to protect VRAM
        if len(full_prompt) > MAX_PROMPT_CHARS:
            full_prompt = full_prompt[-MAX_PROMPT_CHARS:]
            logger.debug(
                f"_build_prompt: prompt truncated to last {MAX_PROMPT_CHARS} chars"
            )

        return full_prompt

    def _generate_text(self, prompt: str) -> str:
        """
        Run the model in generation mode and decode the output tokens.

        Returns the decoded string, or an empty string on error.
        """
        try:
            with torch.no_grad():
                output = self.model([prompt], generate=True)

            generated_tokens = output.get("generated_tokens")
            if generated_tokens is None:
                logger.warning("_generate_text: model returned no generated_tokens")
                return ""

            # generated_tokens may be a tensor, a flat list, or a batched List[List[int]]
            if isinstance(generated_tokens, torch.Tensor):
                generated_tokens = generated_tokens.flatten().tolist()

            # Unwrap batch dimension: V4HyperAssembly returns List[List[int]]
            if (
                isinstance(generated_tokens, list)
                and generated_tokens
                and isinstance(generated_tokens[0], list)
            ):
                generated_tokens = generated_tokens[0]

            # Filter special/padding tokens (-1 is padding in V4's BPE scheme)
            bpe_tokens = [t for t in generated_tokens if 0 <= t < self.tokenizer.vocab_size]

            text = self.tokenizer.bpe.decode(bpe_tokens)
            return text

        except Exception as exc:
            logger.error(f"_generate_text error: {exc}")
            return ""

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def solve(
        self,
        issue: str,
        repo_path: str,
        instance_id: str = "unknown",
    ) -> dict[str, Any]:
        """
        Main entry point: attempt to fix *issue* inside *repo_path*.

        Parameters
        ----------
        issue        : Natural-language description of the bug.
        repo_path    : Absolute path to the repository root.
        instance_id  : SWE-Bench instance identifier (for logging / results).

        Returns
        -------
        A dict with keys:
            instance_id   : str
            patch         : str  (unified diff, or "" on failure)
            reasoning_trace : list[str]  (think-block content per turn)
            turns_used    : int
            success       : bool
        """
        logger.info(f"[{instance_id}] solve() started — repo={repo_path}")
        start_time = time.time()

        reasoning_trace: list[str] = []
        patch_text: str = ""
        last_generated: str = ""

        # ----------------------------------------------------------
        # Initialise conversation history
        # ----------------------------------------------------------
        history: list[dict[str, str]] = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": (
                f"Repository: {repo_path}\n\n"
                f"Issue:\n{issue}\n\n"
                "Please investigate the repository and produce a unified diff patch "
                "that fixes this issue."
            )},
        ]

        # ----------------------------------------------------------
        # Main agentic loop
        # ----------------------------------------------------------
        turn = 0
        for turn in range(1, self.max_turns + 1):
            logger.info(f"[{instance_id}] Turn {turn}/{self.max_turns}")

            # 1. Build the prompt from the current conversation history
            prompt = self._build_prompt(history)

            # 2. Run inference
            generated_text = self._generate_text(prompt)
            last_generated = generated_text  # save for best-effort fallback

            if not generated_text:
                logger.warning(f"[{instance_id}] Turn {turn}: empty generation, skipping")
                history.append({
                    "role": "tool",
                    "content": "[SYSTEM] Generation produced no output, please try again.",
                })
                continue

            # 3. Add the model's raw turn to history
            history.append({"role": "assistant", "content": generated_text})

            # 4. Parse the output
            parsed = parse_agent_output(generated_text)

            if parsed["reasoning"]:
                reasoning_trace.append(parsed["reasoning"])
                logger.debug(
                    f"[{instance_id}] Reasoning: "
                    + parsed["reasoning"][:120].replace("\n", " ")
                    + ("…" if len(parsed["reasoning"]) > 120 else "")
                )

            # 5a. Patch submitted → done
            if parsed["type"] == "patch":
                patch_text = parsed["content"]
                logger.success(
                    f"[{instance_id}] Patch received at turn {turn} "
                    f"({len(patch_text)} chars)"
                )
                break

            # 5b. Tool call → execute and feed back the result
            if parsed["type"] == "tool_call":
                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("args") or {}

                logger.info(
                    f"[{instance_id}] Tool call: {tool_name}({list(tool_args.keys())})"
                )
                tool_result, success = self.tools.call(tool_name, tool_args)
                observation = format_observation(tool_name, tool_result, success)

                history.append({"role": "tool", "content": observation})
                continue

            # 5c. Plain text (no tags) — try to extract a patch anyway
            # Look for a raw diff block in the output (heuristic)
            raw_diff_match = re.search(
                r"(--- .+?\n\+\+\+ .+?\n@@.+?@@.*)", generated_text, re.DOTALL
            )
            if raw_diff_match:
                candidate = raw_diff_match.group(1).strip()
                logger.info(
                    f"[{instance_id}] Heuristic patch extraction at turn {turn}"
                )
                patch_text = candidate
                break

            # Otherwise prompt the model to keep going
            history.append({
                "role": "tool",
                "content": (
                    "[SYSTEM] Your response did not contain a valid <tool_call> or "
                    "<patch> block.  Please use one of the available tools to "
                    "investigate further, or output the fix as <patch>…</patch>."
                ),
            })

        # ----------------------------------------------------------
        # Build result
        # ----------------------------------------------------------
        elapsed = time.time() - start_time
        success = bool(patch_text)

        if not success:
            logger.warning(
                f"[{instance_id}] No patch after {self.max_turns} turns "
                f"({elapsed:.1f}s).  Returning last generation as best effort."
            )
            patch_text = last_generated  # might still be useful

        logger.info(
            f"[{instance_id}] solve() done in {elapsed:.1f}s "
            f"| turns_used={min(turn, self.max_turns)} "
            f"| success={success}"
        )

        return {
            "instance_id":     instance_id,
            "patch":           patch_text,
            "reasoning_trace": reasoning_trace,
            "turns_used":      min(turn, self.max_turns),
            "success":         success,
        }


# ===========================================================================
# CLI entry-point
# ===========================================================================

def _load_model(weights_path: str, device: torch.device) -> V4HyperAssembly:
    """
    Load a ``V4HyperAssembly`` from a saved state-dict checkpoint.

    Gracefully falls back to a freshly initialised (un-trained) model if the
    checkpoint cannot be loaded or does not exist.
    """
    model = V4HyperAssembly(d_model=1024)

    if weights_path and Path(weights_path).exists():
        try:
            state = torch.load(weights_path, map_location=device)
            # State dicts are sometimes wrapped in {"model": ...}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            logger.success(f"Loaded weights from {weights_path}")
        except Exception as exc:
            logger.warning(f"Could not load weights ({exc}) — using random init")
    else:
        logger.warning(
            f"Weights path '{weights_path}' not found — using random init"
        )

    model.eval()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SNAP-C1 V4 Agent Loop — fix a GitHub issue in a local repo"
    )
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "v4_core" / "snapshot_v4_instruct.pt"),
        help="Path to V4HyperAssembly state-dict (.pt file)",
    )
    parser.add_argument(
        "--repo",
        default=str(PROJECT_ROOT),
        help="Absolute path to the repository to patch",
    )
    parser.add_argument(
        "--issue",
        default=(
            "Fix IndexError in get_last_element when list is non-empty: "
            "the function returns lst[len(lst)] which is always out of bounds "
            "instead of lst[len(lst) - 1]."
        ),
        help="Natural-language description of the bug to fix",
    )
    parser.add_argument(
        "--instance-id",
        default="test-instance-001",
        help="SWE-Bench instance identifier (for logging)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Maximum number of agent turns before giving up",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (e.g. 'cpu', 'cuda:0').  Auto-detected if omitted.",
    )
    args = parser.parse_args()

    # ---- Configure logger -------------------------------------------
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="DEBUG",
        colorize=True,
    )

    # ---- Device selection -------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # ---- Load model & tokenizer -------------------------------------
    logger.info("Loading V4HyperAssembly …")
    model = _load_model(args.weights, device)

    logger.info("Loading HybridTokenDecoder …")
    tokenizer = HybridTokenDecoder()

    # ---- Create agent & solve ---------------------------------------
    agent = V4Agent(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_turns=args.max_turns,
    )

    print("\n" + "=" * 70)
    print("  SNAP-C1 V4 — Agentic SWE-Bench Solver")
    print("=" * 70)
    print(f"  Issue    : {args.issue[:80]}{'…' if len(args.issue) > 80 else ''}")
    print(f"  Repo     : {args.repo}")
    print(f"  Max turns: {args.max_turns}")
    print("=" * 70 + "\n")

    result = agent.solve(
        issue=args.issue,
        repo_path=args.repo,
        instance_id=args.instance_id,
    )

    print("\n" + "=" * 70)
    print("  Result")
    print("=" * 70)
    print(f"  instance_id  : {result['instance_id']}")
    print(f"  success      : {result['success']}")
    print(f"  turns_used   : {result['turns_used']}")
    print(f"  reasoning    : {len(result['reasoning_trace'])} think-block(s) captured")

    if result["patch"]:
        print("\n--- PATCH ---")
        # Show first 2000 chars of the patch
        patch_preview = result["patch"][:2_000]
        print(patch_preview)
        if len(result["patch"]) > 2_000:
            print(f"… [{len(result['patch']) - 2_000} more chars]")
    else:
        print("\n(no patch generated)")

    print("=" * 70)
