import os
import sys
import torch
from loguru import logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from v4_core.architecture.v4_assembly import V4HyperAssembly
from v4_core.data.bpe_tokenizer import HybridTokenDecoder
import argparse

def main():
    parser = argparse.ArgumentParser(description="V4 SNAP-C1 Direct Code Generation Baseline")
    parser.add_argument("--weights", type=str, required=True, help="Path to base V4 weights")
    parser.add_argument("--instruct", type=str, help="Path to fine-tuned AST Decoder Head")
    args = parser.parse_args()

    # Load Model
    logger.info("loading SNAP-C1 V4 HyperAssembly...")
    model = V4HyperAssembly()
    device = model.device
    
    logger.info(f"Loading base structural weights from {args.weights}")
    state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
    
    # Strip torch.compile '_orig_mod.' prefixes saved by the trainer
    # AND explicitly strip the old 512-dim random AST noise from the base snapshot
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('._orig_mod.', '.')
        if "ast_geometry_decoder" not in clean_key:
            clean_state_dict[clean_key] = v
        
    # Use strict=False because the AST Decoder had broken dimensions during 
    # the base training run.
    model.load_state_dict(clean_state_dict, strict=False)
    
    # NEW: Overlay the actually trained Instruction Head!
    if args.instruct:
        logger.info(f"Overlaying Fine-Tuned Code Generation Head: {args.instruct}")
        instruct_weights = torch.load(args.instruct, map_location='cpu', weights_only=True)
        model.load_state_dict(instruct_weights, strict=False)
        
    model.eval()
    
    # Load Tokenizer for decoding the output
    tokenizer = HybridTokenDecoder()

    print("\n" + "="*60)
    print("  SNAP-C1 V4 — Direct Code Generation Test (Baseline)")
    print("="*60 + "\n")

    # A simple, isolated syntax bug that requires fixing a loop boundary
    test_prompt = """
Fix the off-by-one Error in this Python function.

```python
def get_last_element(items):
    # Bug: causes IndexError because it doesn't subtract 1
    return items[len(items)]
```
"""

    print("--- INPUT PROMPT ---")
    print(test_prompt.strip())
    print("\n--- GENERATING... ---")

    with torch.no_grad():
        # Pass generate=True to activate the AST auto-regressive decoding loop
        output = model([test_prompt], generate=True)
        
        generated_tokens = output["generated_tokens"][0] # Batch 0
        loss_confidence = output["loss_logits"][0].item()
        time_steps = output["time_steps"]
        experts = output["experts_used"]
        
        # Decode the raw BPE integer array back into Python source text
        raw_code = tokenizer.bpe.decode(generated_tokens)
        
        print("\n--- SYNAPSE METADATA ---")
        print(f"ODE Equilibrium Cycles: {time_steps}")
        print(f"SSD Experts Activated:  {experts}")
        print(f"Structural Confidence:  {loss_confidence:.4f}")
        
        print("\n--- V4 RAW GENERATED CODE TENSORS ---")
        print(raw_code)

if __name__ == "__main__":
    main()
