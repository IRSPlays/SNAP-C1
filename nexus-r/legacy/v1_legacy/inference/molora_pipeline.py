"""
SNAP-C1 MoLoRA Inference Pipeline (Mixture of LoRA Experts)
=============================================================
The core SNAP-C1 innovation: instead of merging LoRA adapters into base
weights (which causes catastrophic interference), this pipeline:

1. Loads the base Qwen3-1.7B model ONCE in 4-bit
2. Loads all 3 LoRA adapters simultaneously via PEFT
3. Routes each query through the MoLoRA router to select the optimal adapter
4. Switches adapters at inference time with model.set_adapter() (zero overhead)
5. Generates the response with the selected expert adapter active

This eliminates the merge interference problem entirely while keeping
all capabilities accessible from a single model instance.

Usage:
    from inference.molora_pipeline import MoLORAPipeline
    
    pipeline = MoLORAPipeline()
    response = pipeline.run("Should I use microservices or monolith?")
    # → Routes to team_thinking adapter, generates multi-perspective response
"""

import json
import re
import sys
import io
import time
from threading import Thread
from pathlib import Path

import torch
import yaml
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.molora_router import MoLoRARouter, RoutingDecision
# Import ThoughtController for v2
try:
    from inference.thought_controller import ThoughtController
except ImportError:
    ThoughtController = None

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


# Capability-specific system prompts — each adapter gets a focused prompt
# that matches what it was trained on
SYSTEM_PROMPTS = {
    "team_thinking": (
        "You are SNAP-C1, an advanced AI that reasons from multiple internal perspectives. "
        "When analyzing problems, use structured multi-perspective reasoning with these personas:\n\n"
        "[Architect]: High-level design, system structure, patterns\n"
        "[Critic]: Challenge assumptions, find flaws, identify risks\n"
        "[Researcher]: Evidence-based analysis, cite facts, compare approaches\n"
        "[Implementer]: Practical execution, concrete steps, code examples\n"
        "[Synthesizer]: Merge all perspectives into a coherent recommendation\n\n"
        "Always use all 5 personas with their exact tags. Provide substantive analysis "
        "under each persona, not just surface-level observations."
    ),
    "self_correction": (
        "You are SNAP-C1 (Self-Neural Adaptive Processing - Core 1), "
        "an advanced AI that reasons from multiple perspectives, "
        "self-corrects its outputs, and uses tools when needed. "
        "Think deeply before responding.\n\n"
        "When reviewing and correcting code, you MUST use ALL THREE sections in this EXACT order:\n\n"
        "1. <review>\\nIdentify bugs, errors, edge cases. Include: Error type, severity, location.\\n</review>\n"
        "2. <fix>\\nProvide the complete corrected code with all issues fixed.\\n</fix>\n"
        "3. <validate>\\nVerify the fix works. Include: Verification status, confidence, reasoning.\\n</validate>\n\n"
        "IMPORTANT: You must output ALL THREE sections (<review>, <fix>, <validate>) in every response. "
        "Never stop after just <review>. Always continue to <fix> and <validate>."
    ),
    "tool_use": (
        "You are SNAP-C1, an advanced AI with tool use capabilities. "
        "When you need to interact with files, run code, or search, emit structured tool calls:\n\n"
        "<tool_call>\n"
        '{"tool": "tool_name", "args": {"key": "value"}}\n'
        "</tool_call>\n\n"
        "Available tools:\n"
        '- read_file: {"tool": "read_file", "args": {"path": "file/path"}}\n'
        '- write_file: {"tool": "write_file", "args": {"path": "file/path", "content": "..."}}\n'
        '- edit_file: {"tool": "edit_file", "args": {"path": "file/path", "old": "...", "new": "..."}}\n'
        '- execute_code: {"tool": "execute_code", "args": {"language": "python", "code": "..."}}\n'
        '- shell_command: {"tool": "shell_command", "args": {"command": "..."}}\n'
        '- search_files: {"tool": "search_files", "args": {"pattern": "...", "path": "."}}\n'
        '- web_search: {"tool": "web_search", "args": {"query": "..."}}\n\n'
        "Always use the exact <tool_call> format. You can chain multiple tool calls."
    ),
    "base": (
        "You are SNAP-C1 (Self-Neural Adaptive Processing - Core 1), "
        "an advanced AI assistant. Provide clear, accurate, and helpful responses."
    ),
}


class MoLORAPipeline:
    """MoLoRA (Mixture of LoRA Experts) inference pipeline.
    
    Loads the base model once and dynamically switches between LoRA adapters
    based on the query type. This avoids catastrophic interference from
    merging adapters into base weights.
    """
    
    AVAILABLE_ADAPTERS = ["team_thinking", "self_correction", "tool_use"]
    
    # Per-adapter max_new_tokens — some capabilities need more room
    ADAPTER_MAX_TOKENS = {
        "team_thinking": 1536,      # 5 personas need space
        "self_correction": 1536,    # review + fix + validate = 3 sections
        "tool_use": 1024,           # tool calls are compact
        "base": 1024,
    }
    
    def __init__(
        self,
        adapters: list[str] | None = None,
        verbose: bool = True,
    ):
        """Initialize the MoLoRA pipeline.
        
        Args:
            adapters: List of adapter names to load (default: all available)
            verbose: Print routing decisions and scores
        """
        self.verbose = verbose
        self.router = MoLoRARouter()
        
        # Load base config
        with open(CONFIG_DIR / "base_model.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        # Load model with adapters
        self.adapters_loaded: list[str] = []
        self.model, self.tokenizer = self._load_model(adapters or self.AVAILABLE_ADAPTERS)
        self.current_adapter: str | None = None
        
        logger.info(f"MoLoRA Pipeline ready. Adapters: {self.adapters_loaded}")
    
    def _load_model(self, adapter_names: list[str]):
        """Load base model in 4-bit and attach all LoRA adapters."""
        model_name = self.config["model"]["name"]
        quant_cfg = self.config["quantization"]
        
        logger.info(f"Loading base model: {model_name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Load each adapter via PEFT
        first_adapter = True
        for name in adapter_names:
            adapter_path = ADAPTERS_DIR / name / "final"
            if not adapter_path.exists():
                logger.warning(f"Adapter not found: {adapter_path}, skipping")
                continue
            
            logger.info(f"Loading adapter: {name} from {adapter_path}")
            
            if first_adapter:
                model = PeftModel.from_pretrained(
                    model, str(adapter_path), adapter_name=name,
                )
                first_adapter = False
            else:
                model.load_adapter(str(adapter_path), adapter_name=name)
            
            self.adapters_loaded.append(name)
        
        model.eval()
        
        # Log VRAM usage
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"VRAM: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return model, tokenizer
    
    def _set_adapter(self, adapter_name: str):
        """Switch to a specific adapter (zero overhead — just pointer swap)."""
        if adapter_name == "base":
            # Disable all adapters to use base model
            self.model.disable_adapter_layers()
            self.current_adapter = "base"
        elif adapter_name in self.adapters_loaded:
            # Re-enable adapter layers if they were disabled
            self.model.enable_adapter_layers()
            self.model.set_adapter(adapter_name)
            self.current_adapter = adapter_name
        else:
            logger.warning(f"Adapter '{adapter_name}' not loaded, using base model")
            self.model.disable_adapter_layers()
            self.current_adapter = "base"
    
    def _generate(
        self, 
        messages: list[dict], 
        max_new_tokens: int | None = None,
        stream: bool = False
    ) -> str | TextIteratorStreamer:
        """Generate a response from the model with the current adapter active.
        
        Args:
            messages: Chat history
            max_new_tokens: Token limit
            stream: If True, returns a TextIteratorStreamer instead of string.
        """
        inf_cfg = self.config["inference"]
        
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["tokenizer"]["max_length"],
        ).to(self.model.device)
        
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens or inf_cfg["max_new_tokens"],
            temperature=inf_cfg["temperature"],
            top_p=inf_cfg["top_p"],
            top_k=inf_cfg["top_k"],
            repetition_penalty=inf_cfg["repetition_penalty"],
            do_sample=inf_cfg["do_sample"],
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer
            
            # Run generation in a separate thread so we can iterate the stream
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            
            return streamer
        
        # Standard blocking generation
        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)
        
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # Clean trailing special tokens
        response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
        
        return response
    
    def run(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        force_adapter: str | None = None,
        recursive: bool = True,  # v2 Default: Recursive Thought Loop
    ) -> dict:
        """Run a query through the SNAP-C1 v2 recursive thought loop.
        
        Args:
            query: The user's input
            recursive: If True, use ThoughtController (System 2 loop).
                       If False, use legacy single-pass generation.
        """
        start_time = time.time()
        
        # Step 1: Initialize ThoughtController if recursive mode
        if recursive and ThoughtController:
            # v2 Logic: Use the recursive controller
            controller = ThoughtController(self, verbose=self.verbose)
            try:
                # Execute the recursive loop
                final_answer = controller.run(query)
                elapsed = time.time() - start_time
                
                return {
                    "response": final_answer,
                    "routing": {"adapter": "recursive_v2", "reasoning": "Recursive Thought Loop"},
                    "adapter_used": self.current_adapter,
                    "generation_time": elapsed,
                    "response_length": len(final_answer),
                }
            except Exception as e:
                logger.error(f"Recursive loop failed: {e}. Falling back to legacy pipeline.")
                # Fallback to legacy logic below
        
        # --- LEGACY v1 LOGIC (Fallback) ---
        
        # Step 1: Route the query
        if force_adapter:
            routing = RoutingDecision(
                primary=force_adapter,
                scores={force_adapter: 1.0},
                reasoning=f"Forced adapter: {force_adapter}",
            )
        else:
            routing = self.router.route(query)
        
        if self.verbose:
            print(f"\n[MoLoRA Router] {routing.reasoning}")
            print(f"[MoLoRA Router] Scores: {', '.join(f'{k}={v:.2f}' for k, v in routing.scores.items())}")
        
        # Step 2: Set the active adapter
        adapter_name = routing.primary
        self._set_adapter(adapter_name)
        
        if self.verbose:
            print(f"[MoLoRA Router] Active adapter: {self.current_adapter}")
        
        # Step 3: Build messages with capability-specific system prompt
        system_prompt = SYSTEM_PROMPTS.get(adapter_name, SYSTEM_PROMPTS["base"])
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": query})
        
        # Step 4: Generate response with per-adapter token limit
        adapter_tokens = self.ADAPTER_MAX_TOKENS.get(adapter_name, 1024)
        response = self._generate(messages, max_new_tokens=adapter_tokens)
        
        elapsed = time.time() - start_time
        
        result = {
            "response": response,
            "routing": {
                "adapter": adapter_name,
                "scores": routing.scores,
                "reasoning": routing.reasoning,
                "multi_adapter": routing.multi_adapter,
            },
            "adapter_used": self.current_adapter,
            "generation_time": elapsed,
            "response_length": len(response),
        }
        
        if self.verbose:
            print(f"[MoLoRA] Generated {len(response)} chars in {elapsed:.2f}s")
        
        return result
    
    def test_all_capabilities(self) -> dict:
        """Run comprehensive tests on all capabilities through the router.
        
        Returns a detailed results dict with scores per capability.
        """
        from training.test_inference import (
            score_team_thinking,
            score_self_correction,
            score_tool_use,
            TEST_PROMPTS,
        )
        
        print("=" * 70)
        print("SNAP-C1 MoLoRA INFERENCE TEST")
        print("=" * 70)
        
        results = {}
        
        for capability, prompts in TEST_PROMPTS.items():
            print(f"\n{'=' * 70}")
            print(f"TESTING: {capability.upper()}")
            print(f"{'=' * 70}")
            
            cap_results = []
            
            for i, prompt in enumerate(prompts):
                print(f"\n--- Prompt {i+1} ---")
                print(f"User: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                
                # Run through MoLoRA pipeline
                result = self.run(prompt, force_adapter=capability, recursive=False)
                response = result["response"]
                
                print(f"\nAssistant output ({len(response)} chars):")
                print("-" * 40)
                print(response[:1500])
                if len(response) > 1500:
                    print(f"... [{len(response) - 1500} more chars]")
                print("-" * 40)
                
                # Score
                if capability == "team_thinking":
                    score = score_team_thinking(response)
                elif capability == "self_correction":
                    score = score_self_correction(response)
                else:
                    score = score_tool_use(response)
                
                print(f"Score: {score}")
                cap_results.append({
                    "prompt": prompt[:80],
                    "score": score,
                    "output_len": len(response),
                    "routing": result["routing"],
                    "generation_time": result["generation_time"],
                })
            
            results[capability] = cap_results
        
        # Summary
        print(f"\n{'=' * 70}")
        print("MoLoRA TEST SUMMARY")
        print(f"{'=' * 70}")
        
        for cap, cap_results in results.items():
            avg_score = sum(r["score"]["score"] for r in cap_results) / len(cap_results)
            avg_time = sum(r["generation_time"] for r in cap_results) / len(cap_results)
            print(f"\n{cap}:")
            print(f"  Average score: {avg_score:.2f}")
            print(f"  Average time:  {avg_time:.1f}s")
            for r in cap_results:
                print(f"  - {r['prompt'][:60]}: score={r['score']['score']:.2f}, len={r['output_len']}")
                if "missing_personas" in r["score"] and r["score"]["missing_personas"]:
                    print(f"    Missing: {r['score']['missing_personas']}")
                if "missing_tags" in r["score"] and r["score"]["missing_tags"]:
                    print(f"    Missing: {r['score']['missing_tags']}")
        
        overall = sum(
            sum(r["score"]["score"] for r in cap_results) / len(cap_results)
            for cap_results in results.values()
        ) / len(results)
        
        print(f"\n{'=' * 70}")
        print(f"MoLoRA OVERALL SCORE: {overall:.2f} / 1.00")
        print(f"(Compare: Merged model scored 0.03 / 1.00)")
        print(f"{'=' * 70}")
        
        # Save results
        results_path = PROJECT_ROOT / "adapters" / "test_results_molora.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")
        
        return results
    
    def test_router_accuracy(self) -> dict:
        """Test the router's ability to classify queries correctly."""
        test_cases = [
            # (query, expected_adapter)
            ("Should I use microservices or monolith for my startup?", "team_thinking"),
            ("Compare REST vs GraphQL for a mobile app backend", "team_thinking"),
            ("What are the tradeoffs between SQL and NoSQL databases?", "team_thinking"),
            ("Evaluate different caching strategies for high-traffic APIs", "team_thinking"),
            
            ("Fix this buggy function: def sort(lst): return lst.sort()", "self_correction"),
            ("Review this code and find the bug: while True: pass", "self_correction"),
            ("What's wrong with this implementation? Previous attempt: ...", "self_correction"),
            ("Debug this error: IndexError: list index out of range", "self_correction"),
            
            ("Show me the contents of config.yaml", "tool_use"),
            ("Run python test.py and show the output", "tool_use"),
            ("Search for all files containing 'TODO' in the project", "tool_use"),
            ("Create a new file called utils.py with a helper function", "tool_use"),
            
            ("What is the capital of France?", "base"),
            ("Tell me a joke", "base"),
            ("Hello, how are you?", "base"),
        ]
        
        print(f"\n{'=' * 70}")
        print("MoLoRA ROUTER ACCURACY TEST")
        print(f"{'=' * 70}")
        
        correct = 0
        total = len(test_cases)
        
        for query, expected in test_cases:
            decision = self.router.route(query)
            is_correct = decision.primary == expected
            correct += int(is_correct)
            
            status = "OK" if is_correct else "WRONG"
            print(f"  [{status}] '{query[:60]}...' -> {decision.primary} (expected: {expected})")
            if not is_correct:
                print(f"         Scores: {', '.join(f'{k}={v:.2f}' for k, v in decision.scores.items())}")
        
        accuracy = correct / total
        print(f"\nRouter accuracy: {correct}/{total} ({accuracy:.0%})")
        
        return {"correct": correct, "total": total, "accuracy": accuracy}


def main():
    """Run MoLoRA tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SNAP-C1 MoLoRA Pipeline")
    parser.add_argument("--test", action="store_true", help="Run all capability tests")
    parser.add_argument("--router-test", action="store_true", help="Test router accuracy")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--adapter", type=str, help="Force a specific adapter")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()
    
    if args.router_test:
        # Router test doesn't need the model
        router = MoLoRARouter()
        pipeline = type('Dummy', (), {'router': router})()
        MoLORAPipeline.test_router_accuracy(pipeline)
        return
    
    pipeline = MoLORAPipeline()
    
    if args.test:
        pipeline.test_all_capabilities()
    elif args.query:
        result = pipeline.run(args.query, force_adapter=args.adapter, recursive=False)
        print(f"\n{result['response']}")
    elif args.chat:
        print("=" * 60)
        print("  SNAP-C1 MoLoRA Chat")
        print("  Type 'quit' to exit")
        print("=" * 60)
        
        history = []
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            
            if not user_input or user_input.lower() == "quit":
                break
            
            result = pipeline.run(user_input, conversation_history=history, recursive=False)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["response"]})
            
            if len(history) > 20:
                history = history[-20:]
            
            print(f"\nSNAP-C1: {result['response']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
