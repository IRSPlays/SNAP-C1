"""
v5_core/architecture/code_introspector.py
==========================================
Pillar A: Code-as-Data Engine

Reads RX.AI's own source files as ASTs, builds a call-graph,
and prepares structured context for the self-modification loop.

Usage:
    introspector = CodeIntrospector("c:/Users/haziq/Documents/RX.AI")
    ctx = introspector.get_file_context("v5_core/training/terminal_loop.py")
    patch = introspector.apply_patch(ctx, new_ast_node)
"""

import ast
import os
import hashlib
import json
from pathlib import Path
from typing import Optional


# ─── AST Visitor: extract call graph ────────────────────────────────────────

class CallGraphVisitor(ast.NodeVisitor):
    """Walks an AST and records all function definitions and calls."""

    def __init__(self):
        self.functions = {}   # name -> {lineno, args, docstring}
        self.calls = {}       # caller -> [callee, ...]
        self._current_func = None

    def visit_FunctionDef(self, node):
        name = node.name
        docstring = ast.get_docstring(node) or ""
        args = [a.arg for a in node.args.args]
        has_annotations = any(a.annotation is not None for a in node.args.args)
        has_return_annotation = node.returns is not None
        self.functions[name] = {
            "lineno": node.lineno,
            "args": args,
            "docstring": docstring,
            "has_annotations": has_annotations or has_return_annotation,
        }
        prev = self._current_func
        self._current_func = name
        self.calls.setdefault(name, [])
        self.generic_visit(node)
        self._current_func = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if self._current_func is not None:
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            else:
                callee = "<complex>"
            self.calls[self._current_func].append(callee)
        self.generic_visit(node)


# ─── File Context ────────────────────────────────────────────────────────────

class FileContext:
    """Represents a parsed source file ready for modification."""

    def __init__(self, path: str, source: str, tree: ast.AST):
        self.path = path
        self.source = source
        self.tree = tree
        self._hash = hashlib.md5(source.encode()).hexdigest()

        visitor = CallGraphVisitor()
        visitor.visit(tree)
        self.functions = visitor.functions
        self.call_graph = visitor.calls

    def to_summary(self) -> dict:
        """Compact representation for injection into model context window."""
        return {
            "path": self.path,
            "hash": self._hash,
            "num_lines": len(self.source.splitlines()),
            "functions": list(self.functions.keys()),
            "call_graph": self.call_graph,
        }

    def __repr__(self):
        return f"FileContext({self.path}, {len(self.functions)} funcs)"


# ─── Patch ───────────────────────────────────────────────────────────────────

class Patch:
    """A proposed change to a source file."""

    def __init__(
        self,
        target_path: str,
        target_function: str,
        new_source: str,
        description: str = "",
    ):
        self.target_path = target_path
        self.target_function = target_function
        self.new_source = new_source
        self.description = description
        self.validated = False
        self.accepted = False

    def to_dict(self) -> dict:
        return {
            "target_path": self.target_path,
            "target_function": self.target_function,
            "description": self.description,
            "new_source_preview": self.new_source[:200],
            "validated": self.validated,
            "accepted": self.accepted,
        }


# ─── Main Introspector ───────────────────────────────────────────────────────

class CodeIntrospector:
    """
    The model's self-awareness module.

    Reads the RX.AI codebase as ASTs and provides:
    - Per-file context for the model's context window
    - A patch application pipeline with ast.unparse()
    - Call-graph navigation for understanding dependencies
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self._cache: dict[str, FileContext] = {}

    # ── Read ────────────────────────────────────────────────────────────────

    def get_file_context(self, relative_path: str) -> Optional[FileContext]:
        """Parse a file and return its FileContext (cached)."""
        full = self.root / relative_path
        if not full.exists():
            print(f"[CodeIntrospector] File not found: {full}")
            return None

        source = full.read_text(encoding="utf-8")
        cached = self._cache.get(relative_path)
        if cached and hashlib.md5(source.encode()).hexdigest() == cached._hash:
            return cached

        try:
            tree = ast.parse(source, filename=str(full))
        except SyntaxError as e:
            print(f"[CodeIntrospector] SyntaxError in {relative_path}: {e}")
            return None

        ctx = FileContext(str(full), source, tree)
        self._cache[relative_path] = ctx
        return ctx

    def get_all_contexts(self, subdir: str = "") -> list[FileContext]:
        """Parse all .py files under a subdirectory."""
        search_root = self.root / subdir if subdir else self.root
        contexts = []
        for py_file in sorted(search_root.rglob("*.py")):
            rel = py_file.relative_to(self.root)
            ctx = self.get_file_context(str(rel))
            if ctx:
                contexts.append(ctx)
        return contexts

    # ── Patch Application ───────────────────────────────────────────────────

    def apply_patch(self, patch: Patch) -> tuple[bool, str]:
        """
        Apply a Patch by rewriting the target function in the file.

        Returns:
            (success: bool, message: str)
        """
        ctx = self.get_file_context(
            str(Path(patch.target_path).relative_to(self.root))
        )
        if ctx is None:
            return False, f"Could not load file: {patch.target_path}"

        # Parse the new function source
        try:
            new_tree = ast.parse(patch.new_source)
        except SyntaxError as e:
            return False, f"Patch has SyntaxError: {e}"

        # Extract the new function node
        new_func_node = None
        for node in ast.walk(new_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == patch.target_function:
                    new_func_node = node
                    break
        if new_func_node is None:
            return False, f"Patch does not contain function '{patch.target_function}'"

        # Replace the matching function in the original tree
        found = False
        for node in ast.walk(ctx.tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == patch.target_function
            ):
                # Replace fields in-place
                for field, value in ast.iter_fields(new_func_node):
                    setattr(node, field, value)
                found = True
                break

        if not found:
            # Not an existing function: append at module level
            ctx.tree.body.append(new_func_node)

        # Unparse the modified tree
        try:
            new_source = ast.unparse(ctx.tree)
        except Exception as e:
            return False, f"ast.unparse failed: {e}"

        # Write back
        full = Path(patch.target_path)
        backup = str(full) + ".bak"
        full.rename(backup)  # backup first
        try:
            full.write_text(new_source, encoding="utf-8")
        except Exception as e:
            Path(backup).rename(full)  # restore backup
            return False, f"Write failed: {e}"

        # Invalidate cache
        self._cache.pop(
            str(Path(patch.target_path).relative_to(self.root)), None
        )
        patch.accepted = True
        return True, f"Patch applied: {patch.target_function} in {patch.target_path}"

    def revert_patch(self, patch: Patch) -> bool:
        """Restore from .bak backup if patch was rejected."""
        target = Path(patch.target_path)
        backup = Path(str(target) + ".bak")
        if backup.exists():
            target.unlink()
            backup.rename(target)
            self._cache.pop(
                str(target.relative_to(self.root)), None
            )
            return True
        return False

    # ── Codebase Summary ───────────────────────────────────────────────────

    def get_codebase_summary(self, subdir: str = "v5_core") -> dict:
        """Return a high-level summary of the codebase for context injection."""
        contexts = self.get_all_contexts(subdir)
        return {
            "total_files": len(contexts),
            "files": [ctx.to_summary() for ctx in contexts],
        }

    def dump_summary_json(self, out_path: str, subdir: str = "v5_core"):
        summary = self.get_codebase_summary(subdir)
        Path(out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[CodeIntrospector] Summary written to {out_path}")


# ─── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    introspector = CodeIntrospector(root)
    summary = introspector.get_codebase_summary("v5_core")
    print(json.dumps(summary, indent=2))
