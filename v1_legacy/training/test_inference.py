"""
SNAP-C1 Inference Test Script
==============================
Tests the merged model for all 3 capabilities:
1. Team Thinking — expects [Architect], [Critic], [Researcher], [Implementer], [Synthesizer] personas
2. Self-Correction — expects <review>, <fix>, <validate> tags
3. Tool Use — expects <tool_call> tags with proper JSON

Loads merged model in 4-bit, runs test prompts, scores outputs.
"""

import json
import re
import sys
import io
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
MERGED_DIR = PROJECT_ROOT / "adapters" / "merged"

SYSTEM_PROMPT = (
    "You are SNAP-C1 (Self-Neural Adaptive Processing - Core 1), "
    "an advanced AI that reasons from multiple perspectives, "
    "self-corrects its outputs, and uses tools when needed. "
    "Think deeply before responding."
)

# Test prompts for each capability
TEST_PROMPTS = {
    "team_thinking": [
        "Should I use a monolithic or microservices architecture for a startup MVP?",
        "What's the best way to handle database migrations in production?",
    ],
    "self_correction": [
        "Write a function to find the second largest number in a list.\n\nPrevious attempt:\ndef second_largest(lst):\n    return sorted(lst)[-2]\n\nPlease review and correct this response.",
        "Write a binary search function.\n\nPrevious attempt:\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\n\nPlease review and correct this response.",
    ],
    "tool_use": [
        "Show me the contents of config.yaml",
        "Find all Python files that contain 'TODO' comments in the project",
    ],
}

# Scoring criteria
def score_team_thinking(output: str) -> dict:
    """Check for persona tags in output. Accepts [Persona] or **Persona** formats."""
    persona_names = ["Architect", "Critic", "Researcher", "Implementer", "Synthesizer"]
    found = []
    missing = []
    for name in persona_names:
        # Accept [Architect], **Architect**, or **Architect:
        if f"[{name}]" in output or f"**{name}**" in output or f"**{name}:" in output:
            found.append(name)
        else:
            missing.append(name)
    score = len(found) / len(persona_names)
    return {
        "score": score,
        "found_personas": found,
        "missing_personas": missing,
        "has_think_tag": "<think>" in output,
    }

def score_self_correction(output: str) -> dict:
    """Check for structured review/fix/validate tags."""
    tags = ["<review>", "</review>", "<fix>", "</fix>", "<validate>", "</validate>"]
    found = [t for t in tags if t in output]
    score = len(found) / len(tags)
    # Also check for actual code in the fix
    has_code = "def " in output or "return " in output or "```" in output
    return {
        "score": score,
        "found_tags": found,
        "missing_tags": [t for t in tags if t not in found],
        "has_code": has_code,
    }

def score_tool_use(output: str) -> dict:
    """Check for proper tool_call tags with JSON."""
    has_tool_call = "<tool_call>" in output and "</tool_call>" in output
    # Try to extract and parse JSON from tool_call
    valid_json = False
    tool_name = None
    if has_tool_call:
        match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                valid_json = True
                tool_name = data.get("tool") or data.get("name") or data.get("function")
            except json.JSONDecodeError:
                pass
    
    score = 0.0
    if has_tool_call:
        score += 0.5
    if valid_json:
        score += 0.3
    if tool_name:
        score += 0.2
    
    return {
        "score": score,
        "has_tool_call_tags": has_tool_call,
        "valid_json": valid_json,
        "tool_name": tool_name,
    }


def main():
    print("=" * 70)
    print("SNAP-C1 INFERENCE TEST — Round 3 (completion_only_loss + 8 epochs)")
    print("=" * 70)
    
    # Load model in 4-bit
    print(f"\nLoading merged model from {MERGED_DIR}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    # Run tests
    results = {}
    
    for capability, prompts in TEST_PROMPTS.items():
        print(f"\n{'=' * 70}")
        print(f"TESTING: {capability.upper()}")
        print(f"{'=' * 70}")
        
        cap_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1} ---")
            print(f"User: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode only the generated tokens
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=False)
            
            # Clean up — remove trailing special tokens but keep structural tags
            response_clean = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
            
            print(f"\nAssistant output ({len(response_clean)} chars):")
            print("-" * 40)
            # Print first 1500 chars
            print(response_clean[:1500])
            if len(response_clean) > 1500:
                print(f"... [{len(response_clean) - 1500} more chars]")
            print("-" * 40)
            
            # Score
            if capability == "team_thinking":
                score = score_team_thinking(response_clean)
            elif capability == "self_correction":
                score = score_self_correction(response_clean)
            else:
                score = score_tool_use(response_clean)
            
            print(f"\nScore: {score}")
            cap_results.append({"prompt": prompt[:80], "score": score, "output_len": len(response_clean)})
        
        results[capability] = cap_results
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    
    for cap, cap_results in results.items():
        avg_score = sum(r["score"]["score"] for r in cap_results) / len(cap_results)
        print(f"\n{cap}:")
        print(f"  Average score: {avg_score:.2f}")
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
    print(f"OVERALL SCORE: {overall:.2f} / 1.00")
    print(f"{'=' * 70}")
    
    # Save results
    results_path = PROJECT_ROOT / "adapters" / "test_results_round3.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    return overall, results


if __name__ == "__main__":
    overall, results = main()
