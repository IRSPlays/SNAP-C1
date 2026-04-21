"""
GSM8K + MBPP Fine-tuning for V7
================================

Fine-tune V7 on reasoning tasks to test if architecture has reasoning potential.

Data:
- GSM8K: Grade school math word problems  
- MBPP: Python coding problems

Goal: See if V7 can learn to solve these problems, not just predict text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import json
import math
import os

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nexus_v1 import NexusV7


class ReasoningTokenizer:
    """Simple but consistent tokenizer."""

    def __init__(self):
        chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~\n\t")
        self.vocab = {c: i + 4 for i, c in enumerate(chars)}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.vocab['<bos>'] = 2
        self.vocab['<eos>'] = 3
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"Tokenizer vocab size: {self.vocab_size}")

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        ids = [self.vocab.get(c, 1) for c in text]
        ids = [2] + ids[:max_len - 2] + [3]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.inv_vocab.get(i, '<unk>') for i in ids if i not in [0, 2, 3])


class ReasoningDataset(Dataset):
    """Dataset for reasoning tasks."""

    def __init__(self, gsm8k_path: str, mbpp_path: str, tokenizer: ReasoningTokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        with open(gsm8k_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(('math', item))

        with open(mbpp_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(('code', item))

        print(f"Loaded {len(self.data)} reasoning problems")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        task_type, item = self.data[idx]

        if task_type == 'math':
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        else:
            text = f"Problem: {item.get('text', item.get('prompt', ''))}\nSolution:\n{item.get('code', item.get('canonical_solution', ''))}"

        input_ids = self.tokenizer.encode(text, self.max_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'task_type': task_type,
        }


def build_reasoning_model(vocab_size: int):
    """Build V7 with custom vocab size."""
    model = NexusV7(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=6,
        num_q_heads=4,
        num_kv_heads=2,
        d_ffn=1024,
        dropout=0.1,
        max_seq_len=256,
        pad_token_id=0
    )
    return model


def train_reasoning(model, train_loader, val_loader, num_epochs=3, lr=3e-4, device='cuda', save_path=None):
    """Fine-tune V7 on reasoning tasks."""
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

    print(f"\n{'='*60}")
    print(f"FINE-TUNING V7 ON REASONING TASKS")
    print(f"{'='*60}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Device: {device}")
    print(f"LR: {lr}, Epochs: {num_epochs}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)

            optimizer.zero_grad()
            result = model(input_ids, labels=input_ids)
            loss = result['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                perplexity = math.exp(min(avg_loss, 100))
                lr_now = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1} Step {global_step}: loss={avg_loss:.4f}, ppl={perplexity:.1f}, lr={lr_now:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        val_result = validate(model, val_loader, device)
        val_ppl = math.exp(min(val_result['loss'], 100))

        print(f"\nEpoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_result['loss']:.4f}, val_ppl={val_ppl:.1f}")

        if val_result['loss'] < best_loss:
            best_loss = val_result['loss']
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model (val_loss={best_loss:.4f})")

    return model


@torch.no_grad()
def validate(model, val_loader, device='cuda'):
    """Validate model."""
    model.eval()
    total_loss = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        result = model(input_ids, labels=input_ids)
        total_loss += result['loss'].item()

    avg_loss = total_loss / len(val_loader)
    return {'loss': avg_loss}


@torch.no_grad()
def test_generation(model, tokenizer: ReasoningTokenizer, device='cuda'):
    """Test model generation on reasoning prompts."""
    model.eval()
    print(f"\n{'='*60}")
    print(f"GENERATION TEST")
    print(f"{'='*60}")

    prompts = [
        "Question: If John has 5 apples and gives 3 to Mary, how many does he have left? Answer:",
        "Question: A store has 12 apples. They sell 7. How many are left? Answer:",
        "Problem: Write a function that returns the sum of two numbers. Solution:",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        input_ids = torch.tensor([tokenizer.encode(prompt, 128)]).to(device)
        gen_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
        generated = tokenizer.decode(gen_ids[0].tolist())
        print(f"Generated: {generated[:200]}...")


def main():
    data_dir = 'C:/Users/Haziq/Documents/SNAP-C1/nexus-r/nexus_v1/training/data'
    save_path = 'C:/Users/Haziq/Documents/SNAP-C1/nexus-r/nexus_v1/training/v7_reasoning_best.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Building tokenizer...")
    tokenizer = ReasoningTokenizer()

    print("\nLoading datasets...")
    train_dataset = ReasoningDataset(
        os.path.join(data_dir, 'gsm8k_train.jsonl'),
        os.path.join(data_dir, 'humaneval_test.jsonl'),
        tokenizer,
        max_len=256
    )

    test_dataset = ReasoningDataset(
        os.path.join(data_dir, 'gsm8k_test.jsonl'),
        os.path.join(data_dir, 'humaneval_test.jsonl'),
        tokenizer,
        max_len=256
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=8)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\nBuilding model...")
    model = build_reasoning_model(tokenizer.vocab_size)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    print("\nTest forward pass...")
    x = torch.randint(0, tokenizer.vocab_size, (2, 64))
    result = model(x, labels=x)
    print(f"Forward OK, loss={result['loss'].item():.4f}")

    print("\nStarting fine-tuning...")
    model = train_reasoning(model, train_loader, val_loader, num_epochs=3, lr=3e-4, device=device, save_path=save_path)

    print("\nFine-tuning complete!")
    test_generation(model, tokenizer, device)
    print("\nDone!")


if __name__ == '__main__':
    main()