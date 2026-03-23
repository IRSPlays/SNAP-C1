"""
SNAP-C1 Universal LoRA Training Script (CPU Edition)
=====================================================
Trains any SNAP-C1 LoRA adapter (team_thinking, self_correction, tool_use)
on Qwen3-4B using LoRA on CPU (float16).

Hardware: AMD Ryzen 5 7600 (16GB RAM) — no CUDA/ROCm available.
After training, merge adapters and export to GGUF for LM Studio inference.

Usage:
    python train_lora.py --config config/team_thinking.yaml
    python train_lora.py --config config/self_correction.yaml
    python train_lora.py --config config/tool_use.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_base_config() -> dict:
    """Load the base model configuration."""
    return load_config(str(CONFIG_DIR / "base_model.yaml"))


def load_model_and_tokenizer(base_config: dict):
    """Load model on CPU in float16 for LoRA training.
    
    Qwen3-4B in float16 = ~8GB RAM, leaving ~6GB for training overhead.
    No quantization needed (bitsandbytes is CUDA-only).
    """
    model_name = base_config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    logger.info("Running on CPU (AMD RX 7600 — no CUDA/ROCm support)")
    
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_free = psutil.virtual_memory().available / (1024**3)
    logger.info(f"System RAM: {ram_gb:.1f} GB total, {ram_free:.1f} GB free")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=base_config["model"]["trust_remote_code"],
        padding_side=base_config["tokenizer"]["padding_side"],
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load in float16 on CPU — ~8GB for 4B model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=base_config["model"]["trust_remote_code"],
        low_cpu_mem_usage=True,
    )
    
    # Enable gradient checkpointing to save memory
    hw_config = base_config.get("hardware", {})
    if hw_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled (saves ~40% memory)")
    
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    
    ram_after = psutil.virtual_memory().available / (1024**3)
    logger.info(f"Free RAM after model load: {ram_after:.1f} GB")
    
    return model, tokenizer


def setup_lora(model, adapter_config: dict):
    """Apply LoRA adapter to model."""
    lora_cfg = adapter_config["lora"]
    
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = model.num_parameters()
    logger.info(f"LoRA applied. Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    return model, peft_config


def load_dataset(adapter_config: dict) -> tuple[Dataset, Dataset | None]:
    """Load training and evaluation datasets from JSONL files."""
    data_cfg = adapter_config["data"]
    data_path = PROJECT_ROOT / data_cfg["path"]
    
    train_path = data_path / data_cfg["train_split"]
    eval_path = data_path / data_cfg.get("eval_split", "eval.jsonl")
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run 'python training/generate_data.py' first to create training data.")
        sys.exit(1)
    
    # Load JSONL
    def load_jsonl(path: Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    
    train_records = load_jsonl(train_path)
    logger.info(f"Loaded {len(train_records)} training examples from {train_path}")
    
    eval_records = None
    if eval_path.exists():
        eval_records = load_jsonl(eval_path)
        logger.info(f"Loaded {len(eval_records)} eval examples from {eval_path}")
    
    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    
    return train_dataset, eval_dataset


def format_chat_template(example: dict, tokenizer) -> str:
    """Format a training example into Qwen3 chat template.
    
    Supports two formats:
    1. Simple: {"instruction": "...", "output": "..."}
    2. With context: {"instruction": "...", "initial_response": "...", "output": "..."}
    """
    messages = []
    
    # System message for SNAP-C1
    system_msg = (
        "You are SNAP-C1 (Self-Neural Adaptive Processing - Core 1), "
        "an advanced AI that reasons from multiple perspectives, "
        "self-corrects its outputs, and uses tools when needed. "
        "Think deeply before responding."
    )
    messages.append({"role": "system", "content": system_msg})
    
    # User instruction
    instruction = example["instruction"]
    if "initial_response" in example:
        instruction += f"\n\nPrevious attempt:\n{example['initial_response']}\n\nPlease review and correct this response."
    
    messages.append({"role": "user", "content": instruction})
    
    # Assistant output
    messages.append({"role": "assistant", "content": example["output"]})
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return formatted


def train(config_path: str, resume_from: str | None = None):
    """Main training function."""
    # Load configs
    adapter_config = load_config(config_path)
    base_config = load_base_config()
    
    adapter_name = adapter_config["adapter"]["name"]
    logger.info(f"=== SNAP-C1 Training: {adapter_name} ===")
    logger.info(f"Description: {adapter_config['adapter']['description']}")
    logger.info("Mode: CPU LoRA (no quantization, float16)")
    
    # Setup — no quantization on CPU
    model, tokenizer = load_model_and_tokenizer(base_config)
    model, peft_config = setup_lora(model, adapter_config)
    
    # Load data
    train_dataset, eval_dataset = load_dataset(adapter_config)
    
    # Format dataset with chat template
    def format_fn(example):
        return {"text": format_chat_template(example, tokenizer)}
    
    train_dataset = train_dataset.map(format_fn)
    if eval_dataset:
        eval_dataset = eval_dataset.map(format_fn)
    
    # Training arguments — optimized for CPU with 16GB RAM
    train_cfg = adapter_config["training"]
    output_dir = PROJECT_ROOT / "adapters" / adapter_name
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=1,        # Tiny batch for 16GB RAM
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_length=base_config["tokenizer"]["max_length"],
        fp16=False,                           # CPU doesn't support fp16 training natively
        bf16=False,
        optim="adamw_torch",                  # Standard optimizer for CPU
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        seed=train_cfg["seed"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
        packing=False,
        # Eval
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg["save_steps"] if eval_dataset else None,
        # CPU-specific
        no_cuda=True,                         # Force CPU training
        dataloader_num_workers=0,             # Avoid multiprocessing overhead
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    logger.info("Starting training on CPU... (this will take a while)")
    logger.info("Tip: Monitor RAM usage — if it exceeds 15GB, reduce max_length in config")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    
    # Save final adapter
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Adapter saved to: {final_path}")
    
    # Save training metrics
    metrics = trainer.state.log_history
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    logger.info(f"=== Training complete: {adapter_name} ===")
    logger.info("Next step: Run merge_adapters.py then export_gguf.py to create your SNAP-C1 GGUF model")
    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 LoRA Training (CPU)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to adapter config YAML (e.g., config/team_thinking.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    args = parser.parse_args()
    
    # Validate config exists
    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
    
    train(args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
