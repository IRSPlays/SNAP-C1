"""
SNAP-C1 Tool Executor
=======================
Executes tool calls emitted by the model in a sandboxed environment.

Supported tools:
- execute_code: Run Python/bash code
- read_file: Read file contents
- write_file: Write/create files
- edit_file: Edit specific text in files
- search_files: Search file contents with regex
- shell_command: Execute shell commands
- web_search: Search the web (placeholder)

Security:
- File operations are sandboxed to PROJECT_ROOT by default
- Code execution has configurable timeout
- Shell commands have configurable timeout
- Dangerous commands can be blocked
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent


class ToolExecutor:
    """Executes tool calls from SNAP-C1 model output."""
    
    # Commands that are blocked by default
    BLOCKED_COMMANDS = [
        "rm -rf /",
        "rm -rf ~",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",  # Fork bomb
        "shutdown",
        "reboot",
        "format",
    ]
    
    def __init__(
        self,
        sandbox_root: Path | None = None,
        code_timeout: int = 30,
        shell_timeout: int = 60,
        allow_network: bool = True,
    ):
        """Initialize tool executor.
        
        Args:
            sandbox_root: Root directory for file operations (default: PROJECT_ROOT)
            code_timeout: Max seconds for code execution
            shell_timeout: Max seconds for shell commands
            allow_network: Allow network access in code/shell execution
        """
        self.sandbox_root = sandbox_root or PROJECT_ROOT
        self.code_timeout = code_timeout
        self.shell_timeout = shell_timeout
        self.allow_network = allow_network
        
        # Registry of available tools
        self.tools = {
            "execute_code": self._execute_code,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "search_files": self._search_files,
            "shell_command": self._shell_command,
            "web_search": self._web_search,
        }
        
        logger.info(f"ToolExecutor initialized. Sandbox: {self.sandbox_root}")
    
    def execute(self, tool_name: str, args: dict) -> dict:
        """Execute a tool call.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            
        Returns:
            Result dict with status, output, and optional error
        """
        if tool_name not in self.tools:
            return {
                "status": "error",
                "error": f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}",
            }
        
        try:
            logger.info(f"Executing tool: {tool_name}")
            result = self.tools[tool_name](**args)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path within the sandbox."""
        resolved = (self.sandbox_root / path).resolve()
        
        # Security check: ensure path is within sandbox
        try:
            resolved.relative_to(self.sandbox_root.resolve())
        except ValueError:
            raise PermissionError(
                f"Path escapes sandbox: {path} resolves to {resolved}, "
                f"which is outside {self.sandbox_root}"
            )
        
        return resolved
    
    def _execute_code(self, language: str = "python", code: str = "") -> dict:
        """Execute Python or bash code."""
        if language not in ("python", "bash"):
            return {"status": "error", "error": f"Unsupported language: {language}"}
        
        # Write code to temp file
        suffix = ".py" if language == "python" else ".sh"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, dir=str(self.sandbox_root)
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            if language == "python":
                cmd = [sys.executable, temp_path]
            else:
                cmd = ["bash", temp_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.code_timeout,
                cwd=str(self.sandbox_root),
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Execution timed out after {self.code_timeout}s"}
        finally:
            os.unlink(temp_path)
    
    def _read_file(self, path: str, offset: int = 0, limit: int = 200) -> dict:
        """Read file contents."""
        resolved = self._resolve_path(path)
        
        if not resolved.exists():
            return {"status": "error", "error": f"File not found: {path}"}
        
        if not resolved.is_file():
            return {"status": "error", "error": f"Not a file: {path}"}
        
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Apply offset and limit
            selected = lines[offset:offset + limit]
            content = "".join(selected)
            
            return {
                "status": "success",
                "content": content,
                "total_lines": len(lines),
                "offset": offset,
                "lines_returned": len(selected),
            }
        except UnicodeDecodeError:
            return {"status": "error", "error": f"Cannot read binary file: {path}"}
    
    def _write_file(self, path: str, content: str) -> dict:
        """Write content to a file."""
        resolved = self._resolve_path(path)
        
        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        
        return {
            "status": "success",
            "bytes_written": len(content.encode("utf-8")),
            "path": str(resolved.relative_to(self.sandbox_root)),
        }
    
    def _edit_file(self, path: str, old_text: str, new_text: str) -> dict:
        """Replace specific text in a file."""
        resolved = self._resolve_path(path)
        
        if not resolved.exists():
            return {"status": "error", "error": f"File not found: {path}"}
        
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
        
        if old_text not in content:
            return {"status": "error", "error": "old_text not found in file"}
        
        count = content.count(old_text)
        new_content = content.replace(old_text, new_text)
        
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        return {
            "status": "success",
            "changes_made": count,
        }
    
    def _search_files(self, pattern: str, path: str = ".", include: str = "*") -> dict:
        """Search file contents using regex."""
        search_root = self._resolve_path(path)
        
        if not search_root.exists():
            return {"status": "error", "error": f"Directory not found: {path}"}
        
        matches = []
        regex = re.compile(pattern)
        
        for file_path in search_root.rglob(include):
            if not file_path.is_file():
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({
                                "file": str(file_path.relative_to(self.sandbox_root)),
                                "line": line_num,
                                "match": line.strip()[:200],
                            })
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return {
            "status": "success",
            "matches": matches,
            "total_matches": len(matches),
        }
    
    def _shell_command(self, command: str, workdir: str | None = None, timeout: int | None = None) -> dict:
        """Execute a shell command."""
        # Security check
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in command:
                return {"status": "error", "error": f"Blocked command pattern: {blocked}"}
        
        cwd = str(self._resolve_path(workdir)) if workdir else str(self.sandbox_root)
        timeout = timeout or self.shell_timeout
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Command timed out after {timeout}s"}
    
    def _web_search(self, query: str, num_results: int = 5) -> dict:
        """Web search placeholder.
        
        In production, this would integrate with a search API
        (SerpAPI, Brave Search, DuckDuckGo, etc.)
        """
        logger.warning("Web search is a placeholder. Integrate a search API for real results.")
        return {
            "status": "success",
            "note": "Web search not yet configured. Integrate a search API.",
            "query": query,
            "results": [],
        }
    
    def list_tools(self) -> list[dict]:
        """List all available tools with their descriptions."""
        descriptions = {
            "execute_code": "Execute Python or bash code in sandboxed environment",
            "read_file": "Read contents of a file",
            "write_file": "Write or create a file",
            "edit_file": "Replace specific text in a file",
            "search_files": "Search file contents with regex",
            "shell_command": "Execute a shell command",
            "web_search": "Search the web (requires API configuration)",
        }
        
        return [
            {"name": name, "description": descriptions.get(name, "")}
            for name in self.tools
        ]
