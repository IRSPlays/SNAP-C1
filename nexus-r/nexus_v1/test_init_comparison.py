"""
Training comparison: OLD vs NEW weight initialization
=====================================================
This script runs actual training experiments to measure the impact
of depth-scaled weight initialization.
"""

import torch
import torch.nn as nn
import math
import sys
import time
import json

sys.path.insert(0, '.')

from nexus_v1 import NexusV7
from tokenizer import TiktokenTokenizer, download_tiny_shakespeare
from scheduler import create_scheduler

# Check DirectML availability
try:
    import torch_directml
    if torch_directml.is_available():
        DEVICE = torch_directml.device()
        DEVICE_TYPE = 'directml'
        print(f'Using AMD RX 7600 GPU: {DEVICE}')
except:
    DEVICE = torch.device('cpu')
    DEVICE_TYPE = 'cpu'
    print('Using CPU')

# Download data
print('Downloading TinyShakespeare...')
text = download_tiny_shakespeare()
print(f'Downloaded {len(text)} chars')

# Create tokenizer
tokenizer = TiktokenTokenizer()
print(f'Tokenizer vocab size: {tokenizer.vocab_size}')

# Tokenize
max_len = 256
stride = 128
tokens = tokenizer.encode(text, max_len=1000000, add_special_tokens=False)
print(f'Total tokens: {len(tokens)}')

# Create sequences
sequences = []
for i in range(0, len(tokens), stride):
    end_idx = min(i + max_len, len(tokens))
    seq = tokens[i:end_idx]
    if len(seq) < max_len:
        seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
    sequences.append(seq)

print(f'Created {len(sequences)} sequences')

# OLD initialization function (the original one)
def old_init_weights(model):
    """Original initialization: std=0.02 for all layers."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Create model with OLD initialization
print('\n=== Testing OLD initialization (std=0.02 flat) ===')
model_old = NexusV7(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    num_layers=6,
    num_q_heads=4,
    num_kv_heads=2,
    d_ffn=1024,
    dropout=0.1,
    max_seq_len=max_len,
).to(DEVICE)
# Apply old initialization (re-initialize after model creation)
old_init_weights(model_old)

# Create model with NEW initialization (built-in)
print('=== Testing NEW initialization (depth-scaled) ===')
model_new = NexusV7(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    num_layers=6,
    num_q_heads=4,
    num_kv_heads=2,
    d_ffn=1024,
    dropout=0.1,
    max_seq_len=max_len,
).to(DEVICE)
# New initialization is applied by default in __init__

# Verify initialization differences
print('\nWeight initialization comparison:')
for name, param in model_old.named_parameters():
    if 'q_proj.weight' in name:
        old_std = param.data.std().item()
        break
for name, param in model_new.named_parameters():
    if 'q_proj.weight' in name:
        new_std = param.data.std().item()
        break
print(f'OLD q_proj std: {old_std:.6f}')
print(f'NEW q_proj std: {new_std:.6f}')
print(f'Scaling factor: {new_std/old_std:.4f}')

# Training settings
batch_size = 4
num_steps = 200
peak_lr = 3e-4

# Create optimizers
def create_optimizer(model):
    return torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n.lower()], 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n.lower()], 'weight_decay': 0.0}
    ], lr=peak_lr, betas=(0.9, 0.95))

optimizer_old = create_optimizer(model_old)
optimizer_new = create_optimizer(model_new)

# WSD scheduler
scheduler_old = create_scheduler('wsd', optimizer_old, warmup_steps=20, stable_steps=100, decay_steps=80, peak_lr=peak_lr, min_lr=peak_lr*0.1)
scheduler_new = create_scheduler('wsd', optimizer_new, warmup_steps=20, stable_steps=100, decay_steps=80, peak_lr=peak_lr, min_lr=peak_lr*0.1)


# Training function
def train_model(model, optimizer, scheduler, name, num_steps=200):
    model.train()
    losses = []
    grad_norms = []

    data_idx = 0
    start_time = time.time()

    for step in range(num_steps):
        # Get batch
        batch_indices = [(data_idx + i) % len(sequences) for i in range(batch_size)]
        batch_tokens = [sequences[i] for i in batch_indices]
        input_ids = torch.tensor(batch_tokens, dtype=torch.long).to(DEVICE)
        data_idx = (data_idx + batch_size) % len(sequences)

        optimizer.zero_grad()

        # Forward
        result = model(input_ids, labels=input_ids)
        loss = result['loss']

        # Backward
        loss.backward()

        # Track gradients
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            avg_gn = sum(grad_norms[-50:]) / len(grad_norms[-50:])
            elapsed = time.time() - start_time
            print(f'{name} step {step:3d}: loss={avg_loss:.4f}, grad_norm={avg_gn:.4f}, lr={scheduler.get_lr()[0]:.2e}, time={elapsed:.1f}s')

    return losses, grad_norms


# Train both models
print('\n--- Training with OLD initialization ---')
losses_old, gn_old = train_model(model_old, optimizer_old, scheduler_old, 'OLD')

print('\n--- Training with NEW initialization ---')
losses_new, gn_new = train_model(model_new, optimizer_new, scheduler_new, 'NEW')

# Compare results
print('\n=== FINAL RESULTS ===')
print(f'OLD initialization:')
print(f'  Initial loss: {losses_old[0]:.4f}')
print(f'  Final loss: {losses_old[-1]:.4f}')
print(f'  Improvement: {(losses_old[0] - losses_old[-1]):.4f} ({(1 - losses_old[-1]/losses_old[0])*100:.1f}%)')
print(f'  Final grad_norm: {gn_old[-1]:.4f}')
print(f'  Avg grad_norm: {sum(gn_old)/len(gn_old):.4f}')

print(f'\nNEW initialization:')
print(f'  Initial loss: {losses_new[0]:.4f}')
print(f'  Final loss: {losses_new[-1]:.4f}')
print(f'  Improvement: {(losses_new[0] - losses_new[-1]):.4f} ({(1 - losses_new[-1]/losses_new[0])*100:.1f}%)')
print(f'  Final grad_norm: {gn_new[-1]:.4f}')
print(f'  Avg grad_norm: {sum(gn_new)/len(gn_new):.4f}')

# Save results for research doc
results = {
    'old_init': {
        'initial_loss': losses_old[0],
        'final_loss': losses_old[-1],
        'improvement_pct': (1 - losses_old[-1]/losses_old[0])*100,
        'final_grad_norm': gn_old[-1],
        'avg_grad_norm': sum(gn_old)/len(gn_old),
        'all_losses': losses_old,
        'all_grad_norms': gn_old
    },
    'new_init': {
        'initial_loss': losses_new[0],
        'final_loss': losses_new[-1],
        'improvement_pct': (1 - losses_new[-1]/losses_new[0])*100,
        'final_grad_norm': gn_new[-1],
        'avg_grad_norm': sum(gn_new)/len(gn_new),
        'all_losses': losses_new,
        'all_grad_norms': gn_new
    }
}

with open('../research_init_comparison.json', 'w') as f:
    json.dump(results, f)
print('\nResults saved to research_init_comparison.json')

# Print which is better
if losses_new[-1] < losses_old[-1]:
    improvement = (losses_old[-1] - losses_new[-1]) / losses_old[-1] * 100
    print(f'\nWINNER: NEW initialization by {improvement:.1f}%')
else:
    degradation = (losses_new[-1] - losses_old[-1]) / losses_old[-1] * 100
    print(f'\nWINNER: OLD initialization by {degradation:.1f}%')