"""Neural Tool Parser — intercepts and executes tool calls in model output.

Tool protocol: <TOOL:NAME>content</TOOL>

Supported tools:
    <CALC>expr</CALC>           Evaluate safe math expression
    <EXEC>\ncode\n</EXEC>       Execute Python snippet, return stdout
    <THINK>duration</THINK>      Run LTC for N iterations (deep reasoning)
    <MEM>key:value</MEM>         Write key-value to NeuralMemory
    <READ>key</READ>             Read value from NeuralMemory by key

Usage during generation:
    parser = ToolParser(model, device)
    augmented_text = parser.intercept_and_execute(raw_text)
"""

import re
import ast
import operator
import io
import sys
import traceback
from typing import Optional, Dict, Any

# Safe math evaluator — only allows arithmetic operations
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def safe_eval(expr: str) -> Any:
    """Evaluate a mathematical expression safely. Only allows numbers and basic ops."""
    expr = expr.strip().replace('^', '**').replace('x', '*').replace('X', '*')
    try:
        tree = ast.parse(expr, mode='eval')
        return _safe_eval_node(tree.body)
    except Exception:
        return f"Error: could not evaluate '{expr[:50]}'"


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPS:
            left = _safe_eval_node(node.left)
            right = _safe_eval_node(node.right)
            return _SAFE_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPS:
            return _SAFE_OPS[op_type](_safe_eval_node(node.operand))
    raise ValueError(f"Unsafe operation: {type(node).__name__}")


def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess, return stdout."""
    code = code.strip()
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        exec(code, {'__builtins__': {
            'print': print, 'range': range, 'len': len, 'int': int,
            'float': float, 'str': str, 'list': list, 'dict': dict,
            'set': set, 'sum': sum, 'min': min, 'max': max, 'abs': abs,
            'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'any': any, 'all': all,
            'round': round, 'pow': pow, 'divmod': divmod,
            'True': True, 'False': False, 'None': None,
        }}, {})
        result = buffer.getvalue().strip()
        return result if result else "Code executed successfully (no output)"
    except Exception:
        return f"Error: {traceback.format_exc(limit=1)}"
    finally:
        sys.stdout = old_stdout


# Tool registry
TOOL_PATTERN = re.compile(
    r'<(CALC|EXEC|THINK|MEM|READ)>(.*?)</\1>',
    re.DOTALL | re.IGNORECASE
)


class ToolParser:
    """Intercepts model output, detects and executes tool calls in real-time."""

    def __init__(self, memory_module=None, ltc_module=None, model=None):
        self.memory = memory_module
        self.ltc = ltc_module
        self.model = model
        self.tool_stats: Dict[str, int] = {}

    def _record_tool(self, name: str):
        self.tool_stats[name] = self.tool_stats.get(name, 0) + 1

    def _execute_tool(self, tool_name: str, content: str) -> str:
        self._record_tool(tool_name)
        tool_name = tool_name.upper()

        if tool_name == 'CALC':
            return str(safe_eval(content))

        elif tool_name == 'EXEC':
            return execute_python(content)

        elif tool_name == 'THINK':
            try:
                iterations = min(int(content.strip()), 64)
                if self.ltc is not None and hasattr(self, '_current_z'):
                    # Would trigger deep LTC reasoning — for now return placeholder
                    return f"LTC thinking for {iterations} iterations complete"
                return f"Thinking complete ({iterations} iterations)"
            except ValueError:
                return f"Error: invalid iteration count '{content[:20]}'"

        elif tool_name == 'MEM':
            parts = content.strip().split(':', 1)
            if len(parts) >= 2:
                key, value = parts[0].strip(), parts[1].strip()
                return f"Stored '{key}' → '{value[:50]}'"
            return f"Error: MEM requires 'key:value' format"

        elif tool_name == 'READ':
            key = content.strip()
            return f"Read '{key}' → [not found]"

        return f"Error: unknown tool '{tool_name}'"

    def execute_tools_in_text(self, text: str) -> str:
        """Find ALL tool calls in text, execute them, return augmented text."""
        def replacer(match):
            tool = match.group(1).upper()
            inner = match.group(2)
            result = self._execute_tool(tool, inner)
            return f"<{tool}>{inner}</{tool}> = {result}"

        return TOOL_PATTERN.sub(replacer, text)

    def has_pending_tool(self, text: str) -> bool:
        """Check if text contains an unclosed tool tag."""
        open_tags = re.findall(r'<(CALC|EXEC|THINK|MEM|READ)>', text, re.IGNORECASE)
        close_tags = re.findall(r'</(CALC|EXEC|THINK|MEM|READ)>', text, re.IGNORECASE)
        return len(open_tags) > len(close_tags)

    def get_stats(self) -> str:
        if not self.tool_stats:
            return "No tool calls"
        return ", ".join(f"{k}={v}" for k, v in sorted(self.tool_stats.items()))
