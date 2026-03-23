"""
Quick Vanilla Re-train for Comparison
=====================================

Train vanilla with SAME architecture as the baseline script used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import json
import math
import os


class VanillaTokenizer:
    def __init__(self):
        chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~\n\t")
        self.vocab = {c: i + 4 for i, c in enumerate(chars)}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.vocab['<bos>'] = 2
        self.vocab['<eos>'] = 3
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, max_len: int = 256) -> list:
        ids = [self.vocab.get(c, 1) for c in text]
        ids = [2] + ids[:max_len - 2] + [3]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids[:max_len]


class VanillaDataset(Dataset):
    def __init__(self, gsm8k_path, mbpp_path, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.data = []
        with open(gsm8k_path) as f:
            for line in f:
                item = json.loads(line)
                self.data.append(('math', item))
        with open(mbpp_path) as f:
            for line in f:
                item = json.loads(line)
                self.data.append(('code', item))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        task_type, item = self.data[idx]
        if task_type == 'math':
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        else:
            text = f"Problem: {item.get('text', item.get('prompt', ''))}\nSolution:\n{item.get('code', '')}"
        return {'input_ids': torch.tensor(self.tokenizer.encode(text))}


class VanillaAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(attn.transpose(1, 2).contiguous().view(B, T, D))


class VanillaBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = VanillaAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.w2(F.gelu(self.w1(self.norm2(x)))))
        return x


class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=5, num_heads=4, d_ffn=1024, dropout=0.1, max_seq_len=256, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([VanillaBlock(d_model, num_heads, d_ffn, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None, return_loss=True):
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
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=self.pad_token_id)
            result['loss'] = loss
        return result

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
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


def main():
    data_dir = 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/nexus-r/nexus_v1/training/data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = VanillaTokenizer()
    train_dataset = VanillaDataset(os.path.join(data_dir, 'gsm8k_train.jsonl'), os.path.join(data_dir, 'humaneval_test.jsonl'), tokenizer)
    test_dataset = VanillaDataset(os.path.join(data_dir, 'gsm8k_test.jsonl'), os.path.join(data_dir, 'humaneval_test.jsonl'), tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=8)
    
    model = VanillaTransformer(vocab_size=tokenizer.vocab_size, d_model=256, num_layers=5, num_heads=4, d_ffn=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * 3)
    
    print(f"Vanilla params: {sum(p.numel() for p in model.parameters()):,}")
    
    best_loss = float('inf')
    for epoch in range(3):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            optimizer.zero_grad()
            result = model(input_ids, labels=input_ids)
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch+1} Step {batch_idx}: loss={result['loss'].item():.4f}")
        
        val_loss = sum(m['input_ids'].numel() for m in val_loader)  # just count
        total_loss = 0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            result = model(input_ids, labels=input_ids)
            total_loss += result['loss'].item()
        val_loss = total_loss / len(val_loader)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, val_ppl={math.exp(val_loss):.1f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/nexus-r/nexus_v1/training/vanilla_v2_best.pt')
    
    print(f"Done! Best val_loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()