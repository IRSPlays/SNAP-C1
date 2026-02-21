"""
SNAP-C1 Universal QLoRA Training Script
========================================
Trains any SNAP-C1 LoRA adapter (team_thinking, self_correction, tool_use)
on Qwen3-8B using QLoRA (4-bit quantization + Low-Rank Adaptation).

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
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
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


def setup_quantization(base_config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for 4-bit QLoRA."""
    quant_cfg = base_config["quantization"]
    
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
    
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        llm_int8_enable_fp32_cpu_offload=True,  # Required: keeps CPU-offloaded layers in fp32
    )


def load_model_and_tokenizer(base_config: dict, bnb_config: BitsAndBytesConfig):
    """Load Qwen3-4B in 4-bit with tokenizer. Optimized for 4GB VRAM with CPU offloading."""
    model_name = base_config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    
    # Clear CUDA cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB total, {free_mem:.1f} GB free)")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=base_config["model"]["trust_remote_code"],
        padding_side=base_config["tokenizer"]["padding_side"],
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Build max_memory map — key to CPU offloading
    # Limits GPU usage so some layers spill to CPU RAM
    max_memory = None
    hw_config = base_config.get("hardware", {})
    if "max_memory" in hw_config:
        max_memory = {}
        for k, v in hw_config["max_memory"].items():
            # Convert string keys like "0" to int for GPU indices
            try:
                max_memory[int(k)] = v
            except ValueError:
                max_memory[k] = v  # "cpu" stays as string
        logger.info(f"Memory map: {max_memory}")
    
    # Use fp16 for RTX 2050
    compute_dtype = getattr(torch, base_config["quantization"]["bnb_4bit_compute_dtype"])
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",              # auto distributes across GPU + CPU
        trust_remote_code=base_config["model"]["trust_remote_code"],
        dtype=compute_dtype,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    
    # Log device distribution
    if hasattr(model, 'hf_device_map'):
        devices = set(str(v) for v in model.hf_device_map.values())
        gpu_layers = sum(1 for v in model.hf_device_map.values() if str(v) == '0')
        cpu_layers = sum(1 for v in model.hf_device_map.values() if str(v) == 'cpu')
        logger.info(f"Device map: {gpu_layers} layers on GPU, {cpu_layers} layers on CPU")
    
    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=hw_config.get("gradient_checkpointing", True),
    )
    
    # Log VRAM after loading
    if torch.cuda.is_available():
        free_after = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"Free VRAM after model load: {free_after:.2f} GB")
    
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params before LoRA: {trainable:,}")
    
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
    
    # Setup
    bnb_config = setup_quantization(base_config)
    model, tokenizer = load_model_and_tokenizer(base_config, bnb_config)
    model, peft_config = setup_lora(model, adapter_config)
    
    # Load data
    train_dataset, eval_dataset = load_dataset(adapter_config)
    
    # Format dataset with chat template
    def format_fn(example):
        return {"text": format_chat_template(example, tokenizer)}
    
    train_dataset = train_dataset.map(format_fn)
    if eval_dataset:
        eval_dataset = eval_dataset.map(format_fn)
    
    # Training arguments
    train_cfg = adapter_config["training"]
    output_dir = PROJECT_ROOT / "adapters" / adapter_name
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_length=train_cfg["max_seq_length"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        seed=train_cfg["seed"],
        gradient_checkpointing=base_config["hardware"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",  # No wandb/tensorboard for now
        dataset_text_field="text",
        completion_only_loss=True,  # CRITICAL: Only compute loss on assistant response, not system/user tokens
        packing=False,
        # Eval
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg["save_steps"] if eval_dataset else None,
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
    logger.info("Starting training...")
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
    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 QLoRA Training")
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
