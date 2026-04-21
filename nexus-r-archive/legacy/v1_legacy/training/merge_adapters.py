"""
SNAP-C1 LoRA Adapter Merge Utility
====================================
Merges multiple LoRA adapters into a single adapter or into the base model weights.

Strategies:
1. Sequential merge: Apply adapters one by one with scaling
2. Weighted merge: Combine adapter weights with custom ratios
3. Full merge: Bake adapters into base model weights (for export)

Usage:
    python merge_adapters.py --adapters team_thinking self_correction tool_use --strategy weighted
    python merge_adapters.py --adapters team_thinking --merge-into-base  # For GGUF export
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


def load_base_config() -> dict:
    """Load the base model configuration."""
    with open(CONFIG_DIR / "base_model.yaml", "r") as f:
        return yaml.safe_load(f)


def load_base_model(merge_into_base: bool = False):
    """Load the base model. 
    
    If merge_into_base=True, load in full precision (no quantization) 
    so we can save merged weights.
    """
    base_config = load_base_config()
    model_name = base_config["model"]["name"]
    
    logger.info(f"Loading base model: {model_name}")
    
    if merge_into_base:
        # Full precision for merging into base weights
        # Use CPU to avoid VRAM limits — merging doesn't need GPU
        logger.info("Loading in fp16 on CPU for merge (avoids VRAM limits)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        # 4-bit for just adapter merging (saves VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    return model, tokenizer


def get_adapter_path(adapter_name: str) -> Path:
    """Get the path to a trained adapter's final checkpoint."""
    final_path = ADAPTERS_DIR / adapter_name / "final"
    if final_path.exists():
        return final_path
    
    # Fallback: look for any checkpoint
    adapter_dir = ADAPTERS_DIR / adapter_name
    if adapter_dir.exists():
        checkpoints = sorted(adapter_dir.glob("checkpoint-*"))
        if checkpoints:
            return checkpoints[-1]  # Latest checkpoint
    
    logger.error(f"No trained adapter found for: {adapter_name}")
    logger.error(f"Looked in: {final_path} and {adapter_dir}")
    sys.exit(1)


def merge_sequential(model, adapter_names: list[str], weights: list[float] | None = None):
    """Merge adapters sequentially by loading each one on top."""
    if weights is None:
        weights = [1.0] * len(adapter_names)
    
    for name, weight in zip(adapter_names, weights):
        adapter_path = get_adapter_path(name)
        logger.info(f"Loading adapter: {name} (weight={weight}) from {adapter_path}")
        
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            adapter_name=name,
        )
        
        # Scale adapter weights
        if weight != 1.0:
            for param_name, param in model.named_parameters():
                if name in param_name and "lora_" in param_name:
                    param.data *= weight
    
    return model


def merge_into_base_model(model, adapter_names: list[str], output_dir: Path):
    """Merge adapters into base model weights (for GGUF export).
    
    This creates a full model with LoRA weights baked in.
    Requires loading in full precision (not quantized).
    """
    for name in adapter_names:
        adapter_path = get_adapter_path(name)
        logger.info(f"Merging adapter into base: {name} from {adapter_path}")
        
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # Bake LoRA into base weights
    
    # Save merged model
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    logger.info(f"Merged model saved to: {output_dir}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 Adapter Merge Utility")
    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="Adapter names to merge (e.g., team_thinking self_correction tool_use)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for each adapter (default: all 1.0)",
    )
    parser.add_argument(
        "--strategy",
        choices=["sequential", "weighted"],
        default="sequential",
        help="Merge strategy",
    )
    parser.add_argument(
        "--merge-into-base",
        action="store_true",
        help="Merge adapters into base model weights (for GGUF export). Requires more VRAM.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: adapters/merged/)",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else ADAPTERS_DIR / "merged"
    
    # Validate adapters exist
    for name in args.adapters:
        get_adapter_path(name)  # Will exit if not found
    
    # Load base model
    model, tokenizer = load_base_model(merge_into_base=args.merge_into_base)
    
    if args.merge_into_base:
        model = merge_into_base_model(model, args.adapters, output_dir)
    else:
        model = merge_sequential(model, args.adapters, args.weights)
        # Save multi-adapter config
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "adapters": args.adapters,
            "weights": args.weights or [1.0] * len(args.adapters),
            "strategy": args.strategy,
        }
        with open(output_dir / "merge_config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Merge config saved to: {output_dir / 'merge_config.json'}")
    
    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Merge complete!")


if __name__ == "__main__":
    main()
