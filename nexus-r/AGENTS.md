# NEXUS-R Development Guide

## Project Overview

NEXUS-R is a reasoning-capable AI architecture project. Current version: **nexus_v1** (based on V7 architecture).

**Goal:** Build an AI that can reason, plan, and self-improve - not just predict tokens.

---

## Key Principles

1. **Validate before innovating** - prove each component works before adding complexity
2. **Brutal honesty** - if something doesn't work, say so
3. **Incremental improvement** - add one innovation at a time
4. **Real benchmarks** - synthetic data proves nothing

---

## Architecture: nexus_v1

Based on V7, proven to learn on real data.

### Components (Keep These!)
- Flash Attention (F.scaled_dot_product_attention)
- RoPE positional encoding
- SwiGLU FFN
- GQA (Grouped Query Attention)
- RMSNorm
- Cosine LR + warmup

### What's Working
- Learning: Yes (perplexity 4.7-5.5 on TinyShakespeare)
- No memorization: Yes (train/val gap ~1.2x)
- Generation: Yes (coherent Shakespeare-style text)

### What Needs Validation
- GPU training at scale (RTX 6000 Ada pending)
- Benchmark on reasoning tasks (GSM8K, MATH)
- Training curves at 10k+ steps

---

## Development Workflow

### Before Adding Innovations

1. **Validate current state**
   - Train for 10k+ steps
   - Confirm perplexity improves
   - Check train/val gap

2. **Benchmark on real tasks**
   - TinyShakespeare (done)
   - GSM8K (TODO)
   - Other reasoning tasks

3. **Profile performance**
   - Tokens/second
   - Memory usage
   - GPU utilization

### Adding Components

**For each new innovation:**
1. Implement in isolation
2. Test on small scale
3. Validate improvement vs baseline
4. If no improvement → remove
5. If improvement → keep and document

### Validation Protocol

For each innovation, measure:
```
Baseline (nexus_v1):
  - Val perplexity: ~5.5
  - Train/val gap: ~1.2x
  - GSM8K accuracy: TBD

Innovation test:
  - Same data, same compute
  - Must beat baseline on BOTH perplexity AND task accuracy
  - Reject if: worse on either
```

---

## File Organization

```
nexus-r/
├── nexus_v1/
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── nexus_v1.py      # Main model (COPY FROM V7)
│   │   ├── flash_attention.py
│   │   ├── rms_norm.py
│   │   ├── rope.py
│   │   └── swiglu.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── data.py
│   │   └── scheduler.py
│   │
│   ├── evaluation/
│   │   ├── benchmark.py
│   │   └── perplexity.py
│   │
│   └── docs/
│       ├── architecture.md
│       └── roadmap.md
│
└── legacy/   # v1-v7, DO NOT MODIFY
```

---

## Code Style

### Imports
```python
# Standard library first
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

# Local imports last
from architecture.xxx import SomeClass
```

### Naming
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `NexusV1`, `FlashAttention` |
| Functions | snake_case | `build_nexus_small` |
| Variables | snake_case | `expert_weights` |
| Constants | UPPER_SNAKE | `MAX_SEQ_LEN` |

### Testing

```python
def test_my_component():
    """Template for component tests."""
    model = MyComponent()
    x = torch.randn(2, 32, 128)
    
    # Forward
    out = model(x)
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.shape == (2, 32, 128)
    
    # Backward
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(model.weight.grad).any()
```

---

## Training Commands

### Quick Test (Tiny Model)
```bash
cd nexus-r/nexus_v1
python training/train.py --model tiny --steps 1000
```

### Full Training (Small Model)
```bash
python training/train.py --model small --steps 10000 --lr 3e-4
```

### Benchmark
```bash
python evaluation/benchmark.py --tasks tiny_shakespeare
```

---

## Common Issues

### Memory OOM
- Reduce batch size
- Use gradient checkpointing
- Reduce sequence length

### Loss NaN
- Lower learning rate
- Check for inf in inputs
- Verify gradient flow

### Not Learning
- Check data pipeline (is data correct?)
- Verify LR is reasonable
- Ensure batch size isn't too small

---

## Next Steps (Priority Order)

1. **Validate on RTX 6000 Ada** - GPU training
2. **Train for 10k+ steps** - Characterize convergence
3. **Benchmark on GSM8K** - Reasoning tasks
4. **Add Working Memory** - Key missing component

---

## Contact

Project by Haziq. Built from scratch, learned from v1-v7 failures.
