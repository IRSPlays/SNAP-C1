"""
Vanilla Transformer Baseline
============================

Standard transformer with learned positional encoding, LayerNorm, MHA, and standard FFN.
Matches V7's size: 6 layers, 4 heads, 256 dim, 1024 FFN.

This is the CONTROL experiment - V7 must beat THIS to prove architectural advantage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import json
import math
import os


class VanillaTokenizer:
    """Same tokenizer as reasoning trainer."""

    def __init__(self):
        chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~\n\t")
        self.vocab = {c: i + 4 for i, c in enumerate(chars)}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.vocab['<bos>'] = 2
        self.vocab['<eos>'] = 3
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        ids = [self.vocab.get(c, 1) for c in text]
        ids = [2] + ids[:max_len - 2] + [3]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.inv_vocab.get(i, '<unk>') for i in ids if i not in [0, 2, 3])


class VanillaDataset(Dataset):
    """Same dataset as reasoning trainer."""

    def __init__(self, gsm8k_path: str, mbpp_path: str, tokenizer: VanillaTokenizer, max_len: int = 256):
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

        print(f"Loaded {len(self.data)} problems")

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


class VanillaAttention(nn.Module):
    """Standard multi-head attention with learned positions."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out_proj(attn.transpose(1, 2).contiguous().view(B, T, D))


class VanillaBlock(nn.Module):
    """Standard transformer block."""

    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = VanillaAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.w2(F.gelu(self.w1(self.norm2(x)))))
        return x


class VanillaTransformer(nn.Module):
    """Standard transformer - CONTROL model."""

    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 6,
                 num_heads: int = 4, d_ffn: int = 1024, dropout: float = 0.1,
                 max_seq_len: int = 256, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            VanillaBlock(d_model, num_heads, d_ffn, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, return_loss: bool = True):
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x = self.dropout(self.embedding(input_ids) + self.pos_embedding(pos))

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {'logits': logits}

        if labels is not None and return_loss:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
            result['loss'] = loss

        return result

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > 256:
                input_ids = input_ids[:, -256:]

            result = self.forward(input_ids, return_loss=False)
            logits = result['logits']
            next_logits = logits[:, -1, :] / temperature

            v, _ = torch.topk(next_logits, min(50, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def train_vanilla(model, train_loader, val_loader, num_epochs=3, lr=3e-4, device='cuda', save_path=None):
    """Train vanilla transformer."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

    print(f"\n{'='*60}")
    print(f"TRAINING VANILLA TRANSFORMER (CONTROL)")
    print(f"{'='*60}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Device: {device}")
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
                print(f"Epoch {epoch+1} Step {global_step}: loss={avg_loss:.4f}, ppl={perplexity:.1f}")

        avg_loss = epoch_loss / len(train_loader)
        val_result = validate_vanilla(model, val_loader, device)
        val_ppl = math.exp(min(val_result['loss'], 100))

        print(f"\nEpoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_result['loss']:.4f}, val_ppl={val_ppl:.1f}")

        if val_result['loss'] < best_loss:
            best_loss = val_result['loss']
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model (val_loss={best_loss:.4f})")

    return model


@torch.no_grad()
def validate_vanilla(model, val_loader, device='cuda'):
    """Validate model."""
    model.eval()
    total_loss = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        result = model(input_ids, labels=input_ids)
        total_loss += result['loss'].item()

    return {'loss': total_loss / len(val_loader)}


def main():
    data_dir = 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/nexus-r/nexus_v1/training/data'
    save_path = 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/nexus-r/nexus_v1/training/vanilla_best.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = VanillaTokenizer()

    train_dataset = VanillaDataset(
        os.path.join(data_dir, 'gsm8k_train.jsonl'),
        os.path.join(data_dir, 'humaneval_test.jsonl'),
        tokenizer,
        max_len=256
    )

    test_dataset = VanillaDataset(
        os.path.join(data_dir, 'gsm8k_test.jsonl'),
        os.path.join(data_dir, 'humaneval_test.jsonl'),
        tokenizer,
        max_len=256
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=8)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = VanillaTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_layers=6,
        num_heads=4,
        d_ffn=1024,
        dropout=0.1,
        max_seq_len=256
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    print("\nTest forward pass...")
    x = torch.randint(0, tokenizer.vocab_size, (2, 64))
    result = model(x, labels=x)
    print(f"Forward OK, loss={result['loss'].item():.4f}")

    print("\nStarting training...")
    model = train_vanilla(model, train_loader, val_loader, num_epochs=3, lr=3e-4, device=device, save_path=save_path)

    print("\nVanilla training complete!")
    print(f"Best model saved to: {save_path}")


if __name__ == '__main__':
    main()