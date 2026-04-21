"""
SNAP-C1 Unified Inference Pipeline
=====================================
The main inference pipeline that orchestrates all SNAP-C1 capabilities:
1. Memory retrieval (inject relevant context)
2. Team Thinking (multi-perspective reasoning)
3. Tool Use (structured tool calls with execution)
4. Self-Correction (review → fix → validate)
5. Memory storage (learn from interaction)

This is the brain of SNAP-C1 — where all the LoRA adapters,
memory system, and tool executor come together.

Usage:
    from inference.pipeline import SNAPPipeline
    
    pipeline = SNAPPipeline()
    response = pipeline.run("Help me debug this code...")
"""

import json
import re
import time
from pathlib import Path
from typing import Generator

import torch
import yaml
from loguru import logger

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_manager import MemoryManager
from inference.tool_executor import ToolExecutor
from inference.lmstudio_backend import LMStudioBackend

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


class SNAPPipeline:
    """The unified SNAP-C1 inference pipeline."""
    
    SYSTEM_PROMPT = (
        "You are SNAP-C1 (Self-Neural Adaptive Processing - Core 1), "
        "an advanced AI that reasons from multiple internal perspectives, "
        "self-corrects its outputs, uses tools when needed, and learns from interactions.\n\n"
        "When thinking through complex problems, use structured internal reasoning:\n"
        "- [Architect]: High-level design and structure\n"
        "- [Critic]: Challenge assumptions, find flaws\n"
        "- [Researcher]: Gather evidence, cite facts\n"
        "- [Implementer]: Practical execution, concrete steps\n"
        "- [Synthesizer]: Merge perspectives, form consensus\n\n"
        "When using tools, emit structured tool calls:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "args": {"key": "value"}}\n'
        "</tool_call>\n\n"
        "After generating responses, review your own output for errors:\n"
        "<review>Check for errors</review>\n"
        "<fix>Apply corrections if needed</fix>\n"
        "<validate>Verify the fix</validate>"
    )
    
    def __init__(
        self,
        adapters: list[str] | None = None,
        enable_memory: bool = True,
        enable_tools: bool = True,
        max_tool_rounds: int = 5,
    ):
        """Initialize the SNAP-C1 pipeline.
        
        Args:
            adapters: List of adapter names to load (default: all available)
            enable_memory: Enable persistent memory system
            enable_tools: Enable tool execution
            max_tool_rounds: Maximum number of tool call rounds per query
        """
        self.max_tool_rounds = max_tool_rounds
        self.enable_tools = enable_tools
        
        # Load base config
        with open(CONFIG_DIR / "base_model.yaml", "r") as f:
            self.base_config = yaml.safe_load(f)
        
        # Load LM Studio Backend
        lm_cfg = self.base_config.get("lmstudio", {})
        self.backend = LMStudioBackend(
            base_url=lm_cfg.get("base_url", "http://localhost:1234/v1"),
            model=lm_cfg.get("model", "local-model")
        )
        
        if not self.backend.check_connection():
            logger.warning("LM Studio connection failed. Please ensure the Local Server is running!")
        
        # Initialize memory system
        self.memory = MemoryManager() if enable_memory else None
        
        # Initialize tool executor
        self.tool_executor = ToolExecutor() if enable_tools else None
        
        logger.info("SNAP-C1 Pipeline initialized.")
    
    def _build_messages(self, user_input: str, conversation_history: list[dict] | None = None) -> list[dict]:
        """Build the message list with system prompt, memory, and conversation history."""
        messages = []
        
        # System prompt with memory context
        system_content = self.SYSTEM_PROMPT
        
        if self.memory:
            memory_context = self.memory.get_context_injection(user_input, max_tokens=512)
            if memory_context:
                system_content += f"\n\n{memory_context}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _generate(self, messages: list[dict]) -> str:
        """Generate a response using the LM Studio backend."""
        # Generation config
        inf_cfg = self.base_config.get("inference", {})
        
        # Use our simplified backend which talks to the loaded GGUF model via API
        response = self.backend.generate(messages, **inf_cfg)
        
        return response
    
    def _extract_tool_calls(self, response: str) -> list[dict]:
        """Extract tool calls from the response."""
        tool_calls = []
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        
        matches = re.finditer(pattern, response, re.DOTALL)
        for match in matches:
            try:
                tool_call = json.loads(match.group(1))
                tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call: {e}")
        
        return tool_calls
    
    def _process_tool_calls(self, response: str, messages: list[dict]) -> str:
        """Process tool calls in the response and continue generation."""
        if not self.tool_executor or not self.enable_tools:
            return response
        
        full_response = response
        
        for round_num in range(self.max_tool_rounds):
            tool_calls = self._extract_tool_calls(full_response)
            
            if not tool_calls:
                break
            
            logger.info(f"Tool round {round_num + 1}: {len(tool_calls)} calls")
            
            # Execute each tool call
            tool_results = []
            for tc in tool_calls:
                result = self.tool_executor.execute(tc["name"], tc.get("args", {}))
                tool_results.append(result)
                logger.info(f"  Tool: {tc['name']} → {result.get('status', 'unknown')}")
            
            # Add tool results to response and continue
            results_text = ""
            for result in tool_results:
                results_text += f"\n<tool_result>\n{json.dumps(result)}\n</tool_result>\n"
            
            # Continue generation with tool results
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "user", "content": f"Tool results:{results_text}\nContinue based on these results."})
            
            continuation = self._generate(messages)
            full_response += results_text + continuation
        
        return full_response
    
    def run(
        self,
        user_input: str,
        conversation_history: list[dict] | None = None,
        stream: bool = False,
    ) -> str:
        """Run the full SNAP-C1 pipeline on a user input.
        
        Args:
            user_input: The user's message
            conversation_history: Previous messages in the conversation
            stream: Whether to stream the response (not yet implemented)
            
        Returns:
            The model's response
        """
        start_time = time.time()
        
        # Build messages with memory context
        messages = self._build_messages(user_input, conversation_history)
        
        # Generate initial response
        response = self._generate(messages)
        
        # Process any tool calls and append results
        if self.enable_tools and "<tool_call>" in response:
            full_response = self._process_tool_calls(response, messages)
            # Find the final AI response after tool outputs
            # Assuming the last chunk of the full response contains the final answer
            response = full_response
        
        # Store interaction in memory
        if self.memory:
            self.memory.store_conversation(
                user_msg=user_input,
                assistant_msg=response,
                quality_score=0.5,  # Default; self-eval would adjust this
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Pipeline complete in {elapsed:.2f}s ({len(response)} chars)")
        
        return response
    
    def chat(self):
        """Interactive chat mode."""
        print("=" * 60)
        print("  SNAP-C1 Interactive Chat")
        print("  Type 'quit' to exit, 'stats' for memory stats")
        print("=" * 60)
        
        history = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "stats":
                if self.memory:
                    stats = self.memory.stats()
                    print(json.dumps(stats, indent=2))
                else:
                    print("Memory system not enabled.")
                continue
            
            response = self.run(user_input, conversation_history=history)
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # Keep history manageable
            if len(history) > 20:
                history = history[-20:]
            
            print(f"\nSNAP-C1: {response}")


def main():
    """Launch SNAP-C1 in interactive chat mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SNAP-C1 Inference Pipeline")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory system")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool execution")
    parser.add_argument("--adapters", nargs="+", default=None, help="Specific adapters to load")
    parser.add_argument("--query", type=str, default=None, help="Single query (non-interactive)")
    args = parser.parse_args()
    
    pipeline = SNAPPipeline(
        adapters=args.adapters,
        enable_memory=not args.no_memory,
        enable_tools=not args.no_tools,
    )
    
    if args.query:
        response = pipeline.run(args.query)
        print(response)
    else:
        pipeline.chat()


if __name__ == "__main__":
    main()
