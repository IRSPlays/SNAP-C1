import os
import sys
import time
import json
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from inference.molora_pipeline import MoLORAPipeline
from inference.tool_executor import ToolExecutor
from memory.memory_manager import MemoryManager

class CuriosityEngine:
    """
    Phase 8 AGI: Open-Ended Curiosity & Auto-Tool Creation.
    Runs when the model is idle. Drives the model to explore new concepts,
    write its own tools, and store the findings in ChromaDB.
    """
    def __init__(self, pipeline: MoLORAPipeline):
        self.pipeline = pipeline
        self.sandbox_dir = PROJECT_ROOT / "sandbox" / "curiosity"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Give the explorer ultimate power in the sandbox
        self.executor = ToolExecutor(sandbox_root=self.sandbox_dir)
        self.memory = MemoryManager()
        
        self.history = []

    def run_exploration_cycle(self):
        """Runs one full cycle of curiosity-driven exploration."""
        logger.info("Starting Open-Ended Curiosity Cycle...")
        
        system_prompt = (
            "You are SNAP-C1, an AGI exploring an open sandbox environment.\n"
            "You are currently idle, so your goal is to expand your capabilities by learning a new skill or writing a new tool.\n"
            "You MUST use the deliberative debate format inside <think> tags before taking an action.\n\n"
            "Available tools:\n"
            "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"cat << 'EOF' > new_tool.py\\n# code here\\nEOF\"}}</tool_call>\n"
            "  <tool_call>{\"name\": \"run_command\", \"kwargs\": {\"command\": \"python new_tool.py\"}}</tool_call>\n"
            "  <tool_call>{\"name\": \"store_skill\", \"kwargs\": {\"description\": \"...\", \"code\": \"...\"}}</tool_call>\n\n"
            "1. Brainstorm an API you want to learn, or a script you want to build (e.g., a regex tester, a web scraper, a math theorem prover).\n"
            "2. Write the tool to a file.\n"
            "3. Run it to test it.\n"
            "4. When it works, use `store_skill` to save it to your permanent memory.\n"
            "Generate ONLY ONE tool call per turn."
        )

        # Inject context from memory so it doesn't relearn the same thing
        memories = self.memory.get_context_injection("recent tools learned", max_tokens=200)
        
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Begin exploration.\n\n{memories}"}
        ]
        
        self.pipeline._set_adapter("team_thinking")
        
        # Run a 10-step exploration loop
        for step in range(10):
            logger.info(f"Curiosity Step {step+1}/10")
            
            # Generate takes a list of message dicts directly in _generate
            prompt_messages = [m for m in self.history if m["role"] != "system"]
            prompt_messages.insert(0, {"role": "system", "content": system_prompt})
            
            logger.info("Generating response (Streaming) ...")
            streamer = self.pipeline._generate(prompt_messages, max_new_tokens=1024, stream=True)
            
            response_chunks = []
            print(f"\n--- [Curiosity Engine Output (Step: {step+1})] ---")
            for chunk in streamer:
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            print("\n----------------------------------\n")
            
            response = "".join(response_chunks).strip()
            response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
            self.history.append({"role": "assistant", "content": response})
            
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                self.history.append({"role": "user", "content": "You did not use a tool. Please explore!"})
                continue
                
            for tc in tool_calls:
                name = tc.get("name")
                kwargs = tc.get("kwargs", {})
                
                # Special intercept for the AGI tool creation
                if name == "store_skill":
                    desc = kwargs.get("description", "")
                    code = kwargs.get("code", "")
                    mem_id = self.memory.store_skill(
                        skill_description=desc, 
                        example=code, 
                        domain="auto-generated-tool"
                    )
                    logger.success(f"AGI Breakthrough: Model invented and stored new tool! (Memory ID: {mem_id})")
                    return # Cycle complete!
                
                elif name == "run_command":
                    # Map to the ToolExecutor's shell command (we use run_command alias in prompts)
                    name = "shell_command"
                
                try:
                    result = self.executor.execute(name, kwargs)
                except Exception as e:
                    result = f"Error: {e}"
                    
                logger.info(f"Sandbox Executed: {name}")
                self.history.append({"role": "user", "content": f"Tool Execution Result:\n{str(result)[:1000]}"})
                
        logger.warning("Curiosity cycle ended without finalizing a new skill.")

    def _extract_tool_calls(self, content: str) -> list[dict]:
        import re
        matches = re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
        calls = []
        for m in matches:
            try:
                calls.append(json.loads(m.group(1).strip()))
            except json.JSONDecodeError:
                pass
        return calls

if __name__ == "__main__":
    pipe = MoLORAPipeline(adapters=["team_thinking", "self_correction", "tool_use"])
    engine = CuriosityEngine(pipe)
    engine.run_exploration_cycle()
