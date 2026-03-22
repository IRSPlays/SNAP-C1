# SNAP-C1 NEXUS V6 - Handoff to GPU Machine

## Quick Start on GPU Machine

```bash
cd /workspaces/SNAP-C1

# Test model works
python -c "
from v6_core.architecture.nexus_v6 import build_nexus_small
import torch
model = build_nexus_small()
x = torch.randint(0, 32000, (2, 64))
logits, info = model(x)
print(f'Output: {logits.shape}')
"

# Run benchmark on untrained model
python v6_core/training/benchmark_nexus.py

# Train on TinyStories
python v6_core/training/train_nexus_real.py --dataset tiny
```

## Architecture Status

**NEXUS V6** - Located at `v6_core/architecture/nexus_v6.py` (940 lines)

### What Works
- Forward/backward pass without NaN
- WSD (Warmup-Stable-Decay) learning rate schedule
- Top-K Sparse MoE with load balancing
- Depth-adaptive gating
- Mamba + Attention hybrid

### Model Sizes
| Size | Params | Layers | Experts |
|------|--------|--------|---------|
| Tiny | 68M | 8 | 4 |
| Small | 157M | 16 | 6 |
| Medium | 462M | 24 | 8 |
| Large | 1.26B | 32 | 12 |

## CRITICAL BUGS - MUST FIX BEFORE GPU TRAINING

### 1. MambaSSM Sequential Loop (lines 108-151)
**Problem:** `for t in range(T)` is sequential - 100x slower than parallel
**Fix:** Replace with parallel scan or use `selective SSM` from Mamba-2

### 2. Hebbian Layer Breaks Autograd (lines 317-326)
**Problem:** Data-dependent control flow (`if self.trace.abs().sum() > 0`)
**Fix:** Remove or rewrite as learnable adapter

### 3. Tree-Guided Evolution Dead Code (lines 502-560)
**Problem:** Never called in forward pass, 3.7M params wasted
**Fix:** Remove or integrate properly into training loop

### 4. "10-100x more efficient" Claim is INVALID
**Reality:** 68M vs GPT-2's 124M = 0.55x size
**Do not claim 10-100x until proven with benchmarks**

## Training Results - FAILED

### CPU Training (What We Did)
- Trained 68M model on TinyStories for 1000 steps
- Loss: 10.3 → 0.0002 (MEMORIZATION, not learning)
- WikiText-2 perplexity: 86M (WORSE than random)
- GPT-2 baseline: 64 perplexity on same data
- 0/4 reasoning tasks passed
- Model outputs: "vandalism vandalism vandalism..."

### Root Causes
1. TinyStories too simple - model memorized
2. Learning rate too high for small dataset
3. No regularization (dropout, weight decay not helping)
4. Architecture bugs (sequential Mamba, broken Hebbian)

## Files Created

```
v6_core/
├── architecture/
│   ├── nexus_v6.py           # Main architecture (NEEDS FIXES)
│   └── __init__.py           # Exports
├── training/
│   ├── benchmark_nexus.py    # Benchmark suite
│   ├── train_nexus_real.py   # Training script
│   └── validate_nexus.py     # Validation
```

## Checkpoints Saved

```
checkpoints/
├── nexus_tiny_trained.pt     # FAILED - memorization
├── nexus_tiny_step200.pt
├── nexus_tiny_step400.pt
├── nexus_tiny_step600.pt
├── nexus_tiny_step800.pt
├── nexus_tiny_step1000.pt
└── training_history.json
```

## Benchmark Results (Untrained)

```
NEXUS (68M): PPL 86,650
GPT-2 (124M): PPL 64.33
Reasoning: 0/4 passed
```

## What to Do Next on GPU

### 1. Fix Architecture First
```python
# In nexus_v6.py:
# - Replace sequential Mamba with proper parallel SSM
# - Remove or fix Hebbian layer
# - Remove dead Tree-Guided Evolution code
# - Add proper gradient checkpointing
```

### 2. Train Properly
```bash
# Use diverse dataset, not just TinyStories
# Train for proper epochs, not just steps
# Save checkpoints every N epochs
# Evaluate on held-out data during training
```

### 3. Benchmark
```bash
# Compare against GPT-2 at same size
# Compare against 10x larger models
# Measure inference speed
# Test on reasoning benchmarks (GSM8K, etc.)
```

## Key Lessons

1. **Loss going to 0 is BAD** - means memorization, not learning
2. **TinyStories too simple** - need diverse, challenging data
3. **CPU training is slow** - 2.5s/step, 42min for 1000 steps
4. **GPU will be 10-100x faster** - enables proper training
5. **Architecture has fundamental bugs** - must fix before training

## Honest Assessment

**Current NEXUS V6 is NOT AGI, NOT 10-100x efficient, NOT ready for production.**

It's a failed experiment that:
- Has correct components (MoE, attention, SSM)
- But implementation has critical bugs
- Training collapsed to memorization
- No reasoning capability demonstrated

**To make it work:**
1. Fix MambaSSM with parallel scan
2. Remove broken Hebbian layer
3. Train on diverse, challenging data
4. Use proper regularization
5. Benchmark against established models

---

*Generated: March 22, 2026*
*Commit: 958e6cc*
