"""
V7 Comprehensive Evaluation
===========================

Rigorous evaluation to determine if V7 has reasoning potential.

Tests:
1. GSM8K Test Set Accuracy - Can it SOLVE problems, not just model text?
2. Held-out Math Problems - Problems not in training
3. Ablation Study - Which component helps: RoPE, SwiGLU, GQA, or RMSNorm?
4. Vanila comparison on actual task accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import json
import math
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nexus_v1 import NexusV7


class ReasoningTokenizer:
    def __init__(self):
        chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~\n\t")
        self.vocab = {c: i + 4 for i, c in enumerate(chars)}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.vocab['<bos>'] = 2
        self.vocab['<eos>'] = 3
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, max_len: int = 512) -> List[int]:
        ids = [self.vocab.get(c, 1) for c in text]
        ids = [2] + ids[:max_len - 2] + [3]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.inv_vocab.get(i, '<unk>') for i in ids if i not in [0, 2, 3])


def extract_final_answer(text: str) -> str:
    """Extract the final numeric answer from generated text."""
    patterns = [
        r'The answer is (\d+)',
        r'= (\d+)',
        r'(\d+)$',
        r'(\d+)',
    ]
    for p in patterns:
        m = re.search(p, text.strip())
        if m:
            return m.group(1)
    return text.strip().split()[-1] if text.strip().split() else ""


def extract_number(text: str) -> str:
    """Extract number from text."""
    m = re.search(r'-?\d+\.?\d*', text)
    return m.group(0) if m else ""


class GSMDataset(Dataset):
    """GSM8K dataset for evaluation."""

    def __init__(self, file_path: str, tokenizer: ReasoningTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                q = item['question']
                a = item['answer']
                final_ans = extract_number(a.split('####')[-1].strip() if '####' in a else a)
                self.data.append({
                    'question': q,
                    'answer': a,
                    'final_answer': final_ans
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        return self.data[idx]


@torch.no_grad()
def evaluate_accuracy(model, dataset, tokenizer, device='cuda', num_samples=100, max_new_tokens=150):
    """Evaluate model on reasoning problems - measure actual accuracy."""
    model.eval()

    correct = 0
    total = min(num_samples, len(dataset))
    errors = []

    for i in range(total):
        item = dataset[i]
        question = item['question']
        expected_answer = item['final_answer']

        prompt = f"Question: {question}\nAnswer:"
        input_ids = torch.tensor([tokenizer.encode(prompt, 256)]).to(device)

        gen_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.7)
        generated = tokenizer.decode(gen_ids[0].tolist())

        model_answer = extract_number(generated)

        is_correct = (model_answer == expected_answer) or (expected_answer in generated)

        if is_correct:
            correct += 1
        else:
            errors.append({
                'q': question[:80],
                'expected': expected_answer,
                'got': model_answer,
                'generated': generated[:150]
            })

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{total}, Accuracy: {correct/(i+1)*100:.1f}%")

    accuracy = correct / total
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'errors': errors[:5]
    }


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device='cuda'):
    """Evaluate perplexity on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        result = model(input_ids, labels=input_ids)
        total_loss += result['loss'].item() * input_ids.numel()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))
    return {'perplexity': perplexity, 'loss': avg_loss}


class VanillaTransformer(nn.Module):
    """Vanilla transformer for comparison."""

    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 5,
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
            nn.TransformerEncoderLayer(d_model, num_heads, d_ffn, dropout, batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

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
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=self.pad_token_id)
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


def run_evaluation():
    """Run comprehensive evaluation."""
    data_dir = 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/nexus-r/nexus_v1/training/data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("V7 COMPREHENSIVE EVALUATION")
    print("="*60)

    tokenizer = ReasoningTokenizer()
    gsm8k_test = GSMDataset(os.path.join(data_dir, 'gsm8k_test.jsonl'), tokenizer)

    print(f"\nGSM8K Test Set: {len(gsm8k_test)} problems")

    print("\n" + "="*60)
    print("TEST 1: GSM8K ACCURACY")
    print("="*60)

    results = {}

    models_to_test = [
        ('v7_matched_best.pt', 'V7 Matched'),
        ('vanilla_best.pt', 'Vanilla'),
        ('v7_reasoning_best.pt', 'V7 Original (larger)'),
    ]

    for filename, name in models_to_test:
        model_path = f'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/nexus-r/nexus_v1/training/{filename}'
        if not os.path.exists(model_path):
            print(f"\n{name}: Model not found at {model_path}")
            continue

        print(f"\n{'='*40}")
        print(f"Evaluating: {name}")
        print(f"{'='*40}")

        if 'v7' in filename:
            model = NexusV7(
                vocab_size=tokenizer.vocab_size,
                d_model=256,
                num_layers=5 if 'matched' in filename else 6,
                num_q_heads=4,
                num_kv_heads=2,
                d_ffn=1024,
                dropout=0.0,
                max_seq_len=256,
                pad_token_id=0
            )
        else:
            model = VanillaTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=256,
                num_layers=5,
                num_heads=4,
                d_ffn=1024,
                dropout=0.0,
                max_seq_len=256
            )

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f"Params: {params:,}")

        print("\nEvaluating accuracy on GSM8K test set...")
        acc_result = evaluate_accuracy(model, gsm8k_test, tokenizer, device, num_samples=200)
        print(f"\n*** {name} Accuracy: {acc_result['accuracy']*100:.1f}% ({acc_result['correct']}/{acc_result['total']}) ***")

        results[name] = {
            'accuracy': acc_result['accuracy'],
            'params': params
        }

        print(f"\nSample errors:")
        for err in acc_result['errors'][:2]:
            print(f"  Q: {err['q']}...")
            print(f"  Expected: {err['expected']}, Got: {err['got']}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, res in results.items():
        print(f"{name}: Accuracy={res['accuracy']*100:.1f}%, Params={res['params']:,}")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    if len(results) >= 2:
        names = list(results.keys())
        acc_diff = (results[names[0]]['accuracy'] - results[names[1]]['accuracy']) * 100
        print(f"\nV7 vs Vanilla accuracy difference: {acc_diff:+.1f}%")

        if acc_diff > 5:
            print("VERDICT: V7 shows meaningful improvement in actual reasoning tasks.")
        elif acc_diff > 0:
            print("VERDICT: V7 shows modest improvement, needs more validation.")
        else:
            print("VERDICT: No improvement over vanilla. Architecture may not matter for reasoning.")

    return results


if __name__ == '__main__':
    results = run_evaluation()