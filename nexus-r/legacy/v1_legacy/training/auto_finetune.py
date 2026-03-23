"""
SNAP-C1 v2 Auto-Finetune Engine: The "Nightly Build"
====================================================
Automatically fine-tunes the model on its own successful experiences.
This closes the loop: Experience -> Memory -> Weights.

Process:
1. Load high-quality experiences from `data/self_improving/buffer.jsonl`.
2. Format them for SFT (Supervised Fine-Tuning).
3. Train the LoRA adapter (1-2 epochs) to bake in the new reasoning patterns.
4. Save as a new adapter version.

Usage:
    python training/auto_finetune.py --adapter team_thinking --threshold 0.8
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from loguru import logger
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "self_improving"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
BUFFER_FILE = DATA_DIR / "buffer.jsonl"


def load_experiences(min_score: float = 0.8) -> list[dict]:
    """Load high-quality experiences from the buffer."""
    experiences = []
    if not BUFFER_FILE.exists():
        logger.warning("No experience buffer found.")
        return []

    with open(BUFFER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("score", 0) >= min_score:
                    # Format for SFT: "User: <instruction>
Assistant: <thought_trace>
<final_answer>..."
                    text = (
                        f"<|im_start|>user
{data['instruction']}<|im_end|>
"
                        f"<|im_start|>assistant
{data['thought_trace']}
"
                        f"<final_answer>{data['final_answer']}</final_answer><|im_end|>"
                    )
                    experiences.append({"text": text})
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(experiences)} high-quality experiences (score >= {min_score})")
    return experiences


def train_update(adapter_name: str, experiences: list[dict]):
    """Run SFT on the new experiences."""
    if not experiences:
        logger.error("No data to train on.")
        return

    # Load Base Model (Qwen3-1.7B)
    model_name = "Qwen/Qwen3-1.7B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Existing Adapter to Continue Training
    adapter_path = ADAPTERS_DIR / adapter_name / "final"
    if adapter_path.exists():
        logger.info(f"Resuming from existing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        logger.warning(f"Adapter {adapter_name} not found. Starting fresh LoRA.")
        # (In production, we should probably fail here or init new LoRA config)

    # Create Dataset
    train_dataset = Dataset.from_list(experiences)

    # Configure Training
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = ADAPTERS_DIR / f"{adapter_name}_v2_{timestamp}"
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=2,              # Quick micro-finetune
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,              # Low LR to preserve knowledge
        logging_steps=1,
        save_strategy="no",              # Only save at end
        report_to="none",
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting Self-Evolution Training...")
    trainer.train()

    # Save
    final_path = ADAPTERS_DIR / adapter_name / "final"
    
    # Backup old adapter
    if final_path.exists():
        backup_path = ADAPTERS_DIR / adapter_name / f"backup_{timestamp}"
        shutil.move(str(final_path), str(backup_path))
        logger.info(f"Backed up old adapter to {backup_path}")

    model.save_pretrained(str(final_path))
    logger.info(f"Updated adapter saved to {final_path}")
    logger.success("Self-Evolution Complete. The model has learned.")


def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 v2 Auto-Finetune")
    parser.add_argument("--adapter", type=str, default="team_thinking", help="Adapter to update")
    parser.add_argument("--threshold", type=float, default=0.8, help="Min score for experiences")
    args = parser.parse_args()

    experiences = load_experiences(args.threshold)
    if len(experiences) < 10:
        logger.warning(f"Only {len(experiences)} experiences found. Recommend >= 10 for training.")
        # Proceed anyway for demo purposes, but in prod we'd wait
    
    train_update(args.adapter, experiences)

if __name__ == "__main__":
    main()
