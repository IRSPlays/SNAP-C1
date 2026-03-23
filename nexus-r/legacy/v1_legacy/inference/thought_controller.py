"""
SNAP-C1 v2 Thought Controller: The "System 2" Brain
===================================================
Replaces linear generation with a recursive, self-directed thought loop.
This controller acts as the 'Operating System' for the model's cognition.

State Machine:
1. IDLE -> 2. THINKING (Streaming) -> 3. ACTION (Research/Code) -> 4. REFLECTION -> 5. ANSWER

Key Features:
- Parses <think>, <research>, <code_sandbox>, <final_answer> tokens in real-time.
- Executes tools autonomously via public API and injects results back into context.
- Maintains 'working memory' (thought trace) across the session.
- Integrates with Episodic Memory (Hippocampus) for retrieval and storage.
- Logs full experience traces for Self-Evolution (DPO training).
- STREAMS thoughts to stdout in real-time.
- PREVENTS INFINITE LOOPS with action tracking and convergence pressure.
"""

import re
import sys
import time
from typing import Generator, Any
from pathlib import Path

from loguru import logger
import yaml

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from inference.tool_executor import ToolExecutor

try:
    from memory.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None

try:
    from training.experience_collector import ExperienceCollector
except ImportError:
    ExperienceCollector = None

class ThoughtController:
    """Controls the recursive thought loop of SNAP-C1 v2."""
    
    def __init__(self, model_pipeline, verbose: bool = True):
        self.model = model_pipeline
        self.tools = ToolExecutor(sandbox_root=PROJECT_ROOT)
        self.verbose = verbose
        
        # Initialize Memory
        if MemoryManager:
            self.memory = MemoryManager()
        else:
            self.memory = None
            
        # Initialize Experience Collector (for Self-Evolution)
        if ExperienceCollector:
            self.collector = ExperienceCollector()
        else:
            self.collector = None
        
        # Load system prompt
        prompt_path = PROJECT_ROOT / "config" / "prompts" / "recursive_system.yaml"
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                self.system_prompt = yaml.safe_load(f)["system_prompt"]
        else:
            logger.warning("System prompt not found, using default.")
            self.system_prompt = "You are SNAP-C1 v2. Think recursively."

        # Regex patterns for tags
        self.patterns = {
            "research": re.compile(r"<research>(.*?)</research>", re.DOTALL),
            "code_sandbox": re.compile(r"<code_sandbox>(.*?)</code_sandbox>", re.DOTALL),
            "tool_call": re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL),
            "memory_store": re.compile(r"<memory_store>(.*?)</memory_store>", re.DOTALL),
            "final_answer": re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL),
            "think": re.compile(r"<think>(.*?)</think>", re.DOTALL),
        }

    def run(self, user_query: str, max_steps: int = 15) -> str:
        """Execute the recursive thought loop."""
        
        # Track the full thought trace for self-learning
        thought_trace = ""
        
        # 1. Retrieve Context from Memory (with safe fallback)
        memory_context = ""
        memories = [] # Initialize safely
        if self.memory:
            try:
                # Search for relevant past experiences/facts
                memories = self.memory.recall(user_query, n_results=3)
                if memories:
                    memory_context = "\n\n<relevant_memories>\n"
                    for m in memories:
                        memory_context += f"- {m['content']} (confidence: {m['metadata'].get('confidence', 0.5):.2f})\n"
                    memory_context += "</relevant_memories>"
                    if self.verbose:
                        print(f"[Memory] Retrieved {len(memories)} relevant memories.")
            except Exception as e:
                logger.error(f"Memory retrieval failed: {e}")

        # Initialize context with memory injection
        messages = [
            {"role": "system", "content": self.system_prompt + memory_context},
            {"role": "user", "content": user_query}
        ]
        
        step_count = 0
        execution_success = False
        execution_failures = 0
        duplicate_actions = 0
        action_history = set()  # Track executed actions to prevent loops
        
        while step_count < max_steps:
            step_count += 1
            if self.verbose:
                print(f"\n[Step {step_count}/{max_steps}] ", end="", flush=True)
            
            # --- Convergence Pressure ---
            if step_count == 6:
                msg = "\n[System] You have done sufficient research and thinking. Please synthesize your findings and provide the <final_answer> now."
                messages.append({"role": "user", "content": msg})
                if self.verbose:
                    print("\n[System] Injecting soft convergence pressure...")
            elif step_count == 10:
                msg = "\n[System] STOP THINKING. You have enough information. OUTPUT <final_answer> IMMEDIATELY."
                messages.append({"role": "user", "content": msg})
                if self.verbose:
                    print("\n[System] Injecting HARD convergence pressure...")

            # STREAMING GENERATION
            streamer = self.model._generate(messages, max_new_tokens=768, stream=True)
            
            chunk = ""
            for new_text in streamer:
                chunk += new_text
                # Real-time print to stdout
                sys.stdout.write(new_text)
                sys.stdout.flush()
            
            # Accumulate trace
            thought_trace += chunk
            messages.append({"role": "assistant", "content": chunk})
            
            # Parse for actions
            action_taken = False
            
            # 1. Check for <research>
            research_match = self.patterns["research"].search(chunk)
            if research_match:
                query = research_match.group(1).strip()
                action_key = f"research:{query}"
                
                if action_key in action_history:
                    observation = "\n[System] You already searched for this. Do not repeat actions. Provide <final_answer>."
                    duplicate_actions += 1
                    if self.verbose:
                        print("\n[System] Duplicate action blocked.")
                else:
                    result = self.tools.execute("web_search", {"query": query})
                    res_content = result.get("results", []) or result.get("note", str(result))
                    observation = f"\n<observation>\nResearch Result for '{query}':\n{res_content}\n</observation>\n"
                    action_history.add(action_key)
                
                messages.append({"role": "user", "content": observation})
                thought_trace += observation
                action_taken = True

            # 2. Check for <code_sandbox>
            code_match = self.patterns["code_sandbox"].search(chunk)
            if code_match:
                code = code_match.group(1).strip()
                action_key = f"code:{code}"
                
                if action_key in action_history:
                    observation = "\n[System] You already ran this exact code. Do not repeat actions. Provide <final_answer>."
                    duplicate_actions += 1
                    if self.verbose:
                        print("\n[System] Duplicate action blocked.")
                else:
                    result = self.tools.execute("execute_code", {"language": "python", "code": code})
                    output = result.get("output", "") + result.get("stderr", "")
                    if result.get("status") == "success":
                        execution_success = True
                    else:
                        execution_failures += 1
                    observation = f"\n<observation>\nCode Execution Output:\n{output}\n</observation>\n"
                    action_history.add(action_key)

                messages.append({"role": "user", "content": observation})
                thought_trace += observation
                action_taken = True

            # 3. Check for <tool_call> (Generic JSON tools)
            tool_match = self.patterns["tool_call"].search(chunk)
            if tool_match:
                import json
                try:
                    tool_json = tool_match.group(1).strip()
                    tool_data = json.loads(tool_json)
                    tool_name = tool_data.get("tool")
                    tool_args = tool_data.get("args", {})
                    
                    action_key = f"tool:{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                    
                    if action_key in action_history:
                        observation = "\n[System] You already executed this tool call. Do not repeat actions."
                    else:
                        if self.verbose:
                            print(f"\n[Action] Executing tool: {tool_name}")
                        
                        result = self.tools.execute(tool_name, tool_args)
                        observation = f"\n<observation>\nTool Output:\n{result}\n</observation>\n"
                        action_history.add(action_key)
                        execution_success = True # Assuming tool execution is a success event
                    
                    messages.append({"role": "user", "content": observation})
                    thought_trace += observation
                    action_taken = True
                    
                except json.JSONDecodeError:
                    observation = "\n[System] Invalid JSON in <tool_call>. Please format as valid JSON."
                    messages.append({"role": "user", "content": observation})
                    thought_trace += observation
                    action_taken = True

            # 4. Check for <memory_store>
            mem_match = self.patterns["memory_store"].search(chunk)
            if mem_match:
                fact = mem_match.group(1).strip()
                
                if self.memory:
                    self.memory.store_fact(fact, source="thought_loop", confidence=0.9)
                    messages.append({"role": "user", "content": "\n<system_note>Memory stored successfully.</system_note>\n"})
                action_taken = True

            # 4. Check for <final_answer>
            final_match = self.patterns["final_answer"].search(chunk)
            if final_match:
                final_answer = final_match.group(1).strip()
                
                if self.collector:
                    self.collector.log(
                        instruction=user_query,
                        thought_trace=thought_trace,
                        final_answer=final_answer,
                        feedback={
                            "code_execution_success": execution_success,
                            "execution_failures": execution_failures,
                            "duplicate_actions": duplicate_actions,
                            "steps": step_count,
                            "memory_used": bool(memories) if self.memory else False
                        }
                    )
                
                return final_answer
            
            # If no action, loop continues
            if not action_taken:
                if "<final_answer>" not in chunk:
                    continue
                else:
                    break

        # Fallback: return the last chunk if max steps reached
        if step_count >= max_steps:
             return f"[Max steps reached] Last thought: {chunk}"

        return "Error: Loop terminated unexpectedly."
