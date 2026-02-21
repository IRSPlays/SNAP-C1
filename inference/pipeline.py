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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_manager import MemoryManager
from inference.tool_executor import ToolExecutor

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
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(adapters)
        
        # Initialize memory system
        self.memory = MemoryManager() if enable_memory else None
        
        # Initialize tool executor
        self.tool_executor = ToolExecutor() if enable_tools else None
        
        logger.info("SNAP-C1 Pipeline initialized.")
    
    def _load_model(self, adapters: list[str] | None = None):
        """Load the base model with optional LoRA adapters."""
        model_name = self.base_config["model"]["name"]
        quant_cfg = self.base_config["quantization"]
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )
        
        logger.info(f"Loading base model: {model_name}")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",  # Left padding for generation
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Build max_memory map for tight VRAM
        max_memory = self.base_config.get("hardware", {}).get("max_memory", None)
        compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        
        # Load adapters if available
        if adapters is None:
            # Auto-detect available adapters
            adapters = []
            for adapter_name in ["team_thinking", "self_correction", "tool_use"]:
                adapter_path = ADAPTERS_DIR / adapter_name / "final"
                if adapter_path.exists():
                    adapters.append(adapter_name)
        
        for adapter_name in adapters:
            adapter_path = ADAPTERS_DIR / adapter_name / "final"
            if adapter_path.exists():
                logger.info(f"Loading adapter: {adapter_name}")
                model = PeftModel.from_pretrained(
                    model,
                    str(adapter_path),
                    adapter_name=adapter_name,
                )
            else:
                logger.warning(f"Adapter not found: {adapter_path}")
        
        model.eval()
        logger.info(f"Model ready with adapters: {adapters or 'none'}")
        
        return model, tokenizer
    
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
        """Generate a response from the model."""
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.base_config["tokenizer"]["max_length"],
        ).to(self.model.device)
        
        # Generation config
        inf_cfg = self.base_config["inference"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=inf_cfg["max_new_tokens"],
                temperature=inf_cfg["temperature"],
                top_p=inf_cfg["top_p"],
                top_k=inf_cfg["top_k"],
                repetition_penalty=inf_cfg["repetition_penalty"],
                do_sample=inf_cfg["do_sample"],
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
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
        
        # Process any tool calls
        if self.enable_tools and "<tool_call>" in response:
            response = self._process_tool_calls(response, messages)
        
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
