import json
import re
import time
from pathlib import Path
from loguru import logger
from .molora_pipeline import MoLORAPipeline
from .tool_executor import ToolExecutor

class FlowController:
    """
    Implements a strict "Flow Engineering" architecture.
    Forces the model through three phases:
    1. Gather Context (AST Search)
    2. Reproduce the Bug (Write a reproduction script)
    3. Fix and Verify (Propose patch until the script passes)
    """
    
    def __init__(self, pipeline: MoLORAPipeline, workspace_dir: str):
        self.pipeline = pipeline
        self.executor = ToolExecutor(sandbox_root=Path(workspace_dir))
        self.history = []
        
        # State tracking for DPO collector
        self.trajectory = {
            "issue": "",
            "search_log": [],
            "reproducer_log": [],
            "patch_log": [],
            "success": False
        }

    def _format_tools_prompt(self, phase: str) -> str:
        base = (
            "You are SNAP-C1, an autonomous software engineer tasked with fixing a bug reported in an issue.\n"
            "You MUST use the deliberative debate format inside <think> tags before taking an action.\n"
            "Example:\n<think>\n[Round 1: Strategy]\n...\n</think>\n"
            "After your thought process, generate EXACLTY ONE tool call."
        )
        
        if phase == "search":
            return base + (
                "\n\nPHASE 1: GATHER CONTEXT\n"
                "Your current goal is to understand the issue and locate the buggy file.\n"
                "Available tools:\n"
                "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"grep -rn 'search string' .\"}}</tool_call>\n"
                "  <tool_call>{\"name\": \"read_file\", \"kwargs\": {\"filepath\": \"path/to/file.py\"}}</tool_call>\n"
                "Explore the codebase until you fully understand the context. Then use a special tool to transition to Phase 2:\n"
                "  <tool_call>{\"name\": \"transition_phase\", \"kwargs\": {\"next_phase\": \"reproduce\"}}</tool_call>"
            )
        elif phase == "reproduce":
            return base + (
                "\n\nPHASE 2: REPRODUCE THE BUG\n"
                "Your current goal is to write a single Python script (`reproduce.py`) that demonstrates the exact bug described in the issue.\n"
                "If the issue shows expected vs actual output, your script MUST assert the expected output.\n"
                "If the issue describes an exception, your script MUST trigger that exception.\n"
                "Available tools:\n"
                "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"cat << 'EOF' > reproduce.py\\n# Python script here\\nEOF\"}}</tool_call>\n"
                "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"python reproduce.py\"}}</tool_call>\n"
                "When `python reproduce.py` successfully demonstrates the bug or crashes, transition to Phase 3:\n"
                "  <tool_call>{\"name\": \"transition_phase\", \"kwargs\": {\"next_phase\": \"fix\"}}</tool_call>"
            )
        elif phase == "fix":
            return base + (
                "\n\nPHASE 3: FIX AND VERIFY\n"
                "Your current goal is to write a patch to fix the bug, and verify it by running `python reproduce.py`.\n"
                "Available tools:\n"
                "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"sed -i 's/old/new/g' file.py\"}}</tool_call>\n"
                "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"cat << 'EOF' > patch.diff\\n...\\nEOF && git apply patch.diff\"}}</tool_call>\n"
                "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"python reproduce.py\"}}</tool_call>\n"
                "If the reproduction script passes without errors, generating the final patch:\n"
                "  <final_answer>\nDiff contents here\n</final_answer>"
            )
        
        return base

    def run_issue(self, issue_description: str, max_steps: int = 20) -> dict:
        """Solves a single SWE-bench issue using the flow architecture."""
        self.trajectory["issue"] = issue_description
        current_phase = "search"
        
        logger.info(f"Starting FlowController on issue (Phase: {current_phase})")
        
        self.history = [
            {"role": "user", "content": f"ISSUE DESCRIPTION:\n{issue_description}\n\nBegin your analysis."}
        ]
        
        step = 0
        while step < max_steps:
            logger.info(f"Step {step+1}/{max_steps} [{current_phase}]")
            
            # Format system prompt dynamically
            system_prompt = self._format_tools_prompt(current_phase)
            self.history[0]["content"] = system_prompt # Hacky way to set system prompt at runtime ?
            
            # 1. Model Generation
            self.pipeline._set_adapter("team_thinking")
            
            # Since _generate expects a list of dicts directly
            prompt_messages = [m for m in self.history if m["role"] != "system"]
            if "content" in self.history[0]: # Prepend the active system prompt
                 prompt_messages.insert(0, {"role": "system", "content": system_prompt})
            
            logger.info("Generating response (Streaming) ...")
            start = time.time()
            streamer = self.pipeline._generate(prompt_messages, max_new_tokens=1024, stream=True)
            
            response_chunks = []
            print(f"\n--- [Model Output (Phase: {current_phase})] ---")
            for chunk in streamer:
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            print("\n----------------------------------\n")
            
            response = "".join(response_chunks).strip()
            # Clean trailing special tokens (_generate does this for non-stream, but we must do it manually here)
            response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
            
            logger.debug(f"Generation took {time.time()-start:.2f}s")
            
            self.history.append({"role": "assistant", "content": response})
            
            # Log to trajectory
            if current_phase == "search":
                self.trajectory["search_log"].append(response)
            elif current_phase == "reproduce":
                self.trajectory["reproducer_log"].append(response)
            elif current_phase == "fix":
                self.trajectory["patch_log"].append(response)
            
            # 2. Extract and Execute Actions
            if "<final_answer>" in response:
                logger.info("Model declared final answer. Verifying...")
                self.trajectory["success"] = True # Assume success for now, evaluated later
                break
                
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                logger.warning("No tool calls found. Forcing continuation.")
                self.history.append({"role": "user", "content": "You didn't generate a tool call or transition. Please take an action."})
            else:
                for tc in tool_calls:
                    name = tc.get("name")
                    kwargs = tc.get("kwargs", {})
                    
                    if name == "transition_phase":
                        next_phase = kwargs.get("next_phase")
                        if next_phase in ["reproduce", "fix"]:
                            current_phase = next_phase
                            logger.info(f"Transitioning to Phase: {current_phase}")
                            self.history.append({"role": "user", "content": f"Transitioned to {current_phase} phase successfully."})
                        continue
                    
                    try:
                        result = self.executor.execute(name, kwargs)
                    except Exception as e:
                        result = f"Error executing {name}: {e}"
                        
                    logger.info(f"Executed {name}. Output len: {len(result)}")
                    self.history.append({
                        "role": "user", 
                        "content": f"Tool Execution Result ({name}):\n{result[:1000]}"
                    })
            
            step += 1
            
        return self.trajectory

    def _extract_tool_calls(self, content: str) -> list[dict]:
        """Parses <tool_call> JSON from generated text."""
        matches = re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
        calls = []
        for m in matches:
            try:
                calls.append(json.loads(m.group(1).strip()))
            except json.JSONDecodeError:
                logger.error(f"Malformed tool call JSON: {m.group(1)}")
        return calls
