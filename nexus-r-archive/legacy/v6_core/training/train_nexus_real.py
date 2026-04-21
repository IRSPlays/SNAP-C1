"""
NEXUS V6 Training Script - Real Data Training
==============================================

Trains NexusV6 on coding/reasoning data to verify:
1. Model can learn from real data
2. Loss decreases over time
3. Architecture innovations work in practice

Efficiency goal: Beat models 10-100x size through:
- Depth-adaptive experts (only compute needed)
- Latent concept discovery (focus on relevant concepts)
- Mamba + attention hybrid (efficient sequence modeling)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project to path
sys.path.insert(0, '/workspaces/SNAP-C1')

from v6_core.architecture.nexus_v6 import (
    NexusV6, build_nexus_small, build_nexus_tiny
)


class ToolUseDataset(Dataset):
    """Dataset from tool_use JSONL files - coding/reasoning tasks."""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                instruction = obj.get('instruction', '')
                output = obj.get('output', '')
                
                # Format: instruction [EOS] output [EOS]
                text = f"{instruction}\n\n{output}"
                ids = tokenizer.encode(text, truncation=True, max_length=max_length)
                self.samples.append(ids)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ids = self.samples[idx]
        
        # Input is all but last token, target is all but first (causal LM)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        
        return {'input_ids': input_ids, 'labels': labels}


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for basic language model training."""
    
    def __init__(self, tokenizer, max_length: int = 128, max_samples: int = 10000):
        from datasets import load_dataset
        
        print("Loading TinyStories dataset...")
        self.dataset = load_dataset('roneneldan/TinyStories', split='train')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = min(max_samples, len(self.dataset))
        
        print(f"Using {self.max_samples} samples")
    
    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        
        return {'input_ids': input_ids, 'labels': labels}


def collate_fn(batch):
    """Collate function - pad sequences to same length."""
    max_len = max(len(x['input_ids']) for x in batch)
    
    input_ids = []
    labels = []
    
    for example in batch:
        pad_len = max_len - len(example['input_ids'])
        input_ids.append(torch.cat([
            example['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        labels.append(torch.cat([
            example['labels'],
            torch.zeros(pad_len, dtype=torch.long) - 100  # ignore index
        ]))
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }


class NexusTrainer:
    """Trainer for NexusV6 with WSD schedule and efficiency optimizations."""
    
    def __init__(
        self,
        model: NexusV6,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        peak_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_steps: int = 50,
        total_steps: int = 1000,
        gradient_accumulation: int = 4,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gradient_accumulation = gradient_accumulation
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Dataloader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True
        )
        
        self.eval_loader = None
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                drop_last=True
            )
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Track losses for monitoring
        self.loss_history = []
    
    def get_lr(self):
        """WSD: Warmup -> Stable -> Decay schedule."""
        step = self.global_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / self.warmup_steps
        else:
            # Cosine decay to min_lr
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def train_step(self, batch):
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits, info = self.model(input_ids)
        
        # Cross entropy loss (ignore index -100)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Scale for gradient accumulation
        loss = loss / self.gradient_accumulation
        
        # Backward
        loss.backward()
        
        return loss.item() * self.gradient_accumulation
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"NEXUS V6 Training")
        print(f"{'='*60}")
        print(f"Steps: {self.total_steps}")
        print(f"Batch size: {self.batch_size}, Grad accum: {self.gradient_accumulation}")
        print(f"Effective batch: {self.batch_size * self.gradient_accumulation}")
        print(f"LR: {self.peak_lr:.1e} -> {self.min_lr:.1e}")
        print(f"{'='*60}\n")
        
        self.model.train()
        optimizer = self.optimizer
        
        total_loss = 0.0
        start_time = time.time()
        accum_loss = 0.0
        step_loss = 0.0
        
        data_iter = iter(self.train_loader)
        
        while self.global_step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            # Update LR
            lr = self.get_lr()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward
            loss = self.train_step(batch)
            step_loss += loss
            
            # Gradient accumulation
            if (self.global_step + 1) % self.gradient_accumulation == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Increment
                self.global_step += 1
                accum_loss += step_loss
                step_loss = 0.0
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = accum_loss / self.log_interval
                    self.loss_history.append(avg_loss)
                    elapsed = time.time() - start_time
                    
                    # Estimate tokens/sec
                    tokens_per_sec = (
                        self.batch_size * self.gradient_accumulation * 
                        self.log_interval * 128  # approximate seq len
                    ) / elapsed if elapsed > 0 else 0
                    
                    # Get any info from model (expert usage, etc.)
                    info_str = ""
                    
                    print(
                        f"Step {self.global_step}/{self.total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Tokens/sec: {tokens_per_sec:.0f} | "
                        f"Time: {elapsed:.1f}s"
                    )
                    
                    accum_loss = 0.0
                    start_time = time.time()
                
                # Eval
                if self.eval_loader and self.global_step % self.eval_interval == 0:
                    eval_loss = self.evaluate()
                    print(f"\n*** Eval Loss: {eval_loss:.4f} ***\n")
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint("best")
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % 500 == 0 and self.global_step > 0:
                    self.save_checkpoint(f"step_{self.global_step}")
        
        print("\nTraining complete!")
        self.save_checkpoint("final")
        
        # Print final loss trend
        if len(self.loss_history) > 10:
            early = sum(self.loss_history[:len(self.loss_history)//4]) / (len(self.loss_history)//4)
            late = sum(self.loss_history[-len(self.loss_history)//4:]) / (len(self.loss_history)//4)
            print(f"\nLoss trend: {early:.4f} -> {late:.4f} ({'↓ improving' if late < early else '↑ worse'})")
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits, _ = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        path = self.checkpoint_dir / f"nexus_{name}.pt"
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'loss_history': self.loss_history,
        }, path)
        print(f"Saved: {path}")


def train_tiny():
    """Train on TinyStories for quick validation."""
    print("\n" + "="*60)
    print("Training on TinyStories (quick validation)")
    print("="*60 + "\n")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model - use tiny for quick CPU training
    model = build_nexus_tiny()
    
    # Resize vocab to match tokenizer
    print(f"Resizing vocab from {model.vocab_size} to {tokenizer.vocab_size}")
    model.resize_token_embeddings(tokenizer.vocab_size)
    
    # Dataset
    train_ds = TinyStoriesDataset(tokenizer, max_length=128, max_samples=5000)
    eval_ds = TinyStoriesDataset(tokenizer, max_length=128, max_samples=500)
    
    # Trainer
    trainer = NexusTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        batch_size=4,
        peak_lr=1e-3,
        warmup_steps=20,
        total_steps=200,
        gradient_accumulation=2,
        log_interval=5,
        eval_interval=50,
        checkpoint_dir="./checkpoints/nexus_tiny"
    )
    
    trainer.train()


def train_tool_use():
    """Train on tool_use data (coding/reasoning)."""
    print("\n" + "="*60)
    print("Training on Tool Use Data (coding/reasoning)")
    print("="*60 + "\n")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    model = build_nexus_tiny()
    model.resize_token_embeddings(tokenizer.vocab_size)
    
    # Dataset
    train_path = '/workspaces/SNAP-C1/data/tool_use/train.jsonl'
    eval_path = '/workspaces/SNAP-C1/data/tool_use/eval.jsonl'
    
    train_ds = ToolUseDataset(train_path, tokenizer, max_length=256)
    eval_ds = ToolUseDataset(eval_path, tokenizer, max_length=256)
    
    # Trainer
    trainer = NexusTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        batch_size=2,
        peak_lr=5e-4,
        warmup_steps=30,
        total_steps=500,
        gradient_accumulation=4,
        log_interval=10,
        eval_interval=100,
        checkpoint_dir="./checkpoints/nexus_tool_use"
    )
    
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['tiny', 'tool_use'], default='tiny',
                        help='Dataset to train on')
    args = parser.parse_args()
    
    if args.dataset == 'tiny':
        train_tiny()
    else:
        train_tool_use()