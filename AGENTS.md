# AGENTS.md - SNAP-C1 Development Guide

## Project Overview

SNAP-C1 is a from-scratch neural architecture for self-improving code agents. The main codebase is in `v6_core/` with the current focus on NEXUS V6 architecture in `v6_core/architecture/nexus_v6.py`.

---

## Build & Test Commands

### Quick Test (Single File)
```bash
# Test NEXUS V6 forward pass
python -c "
from v6_core.architecture.nexus_v6 import build_nexus_small
import torch
model = build_nexus_small()
x = torch.randint(0, 32000, (2, 64))
logits, info = model(x)
print(f'Output: {logits.shape}')
"

# Smoke test V6 (from v6_core directory)
cd /workspaces/SNAP-C1/v6_core && python test_v6_smoke.py

# Run validation suite (if timeout issues, reduce num_steps)
cd /workspaces/SNAP-C1 && python v6_core/training/validate_nexus.py
```

### Full Test Suite
```bash
# All root-level tests
cd /workspaces/SNAP-C1
python test_device.py
python test_softmax.py
python test_scatter.py

# Architecture smoke test
cd /workspaces/SNAP-C1/v6_core && python test_v6_smoke.py
```

### Training
```bash
# NEXUS V6 training with WSD scheduler
cd /workspaces/SNAP-C1
python -c "
from v6_core.architecture.nexus_v6 import build_nexus_small, WSDTrainer
import torch
model = build_nexus_small()
trainer = WSDTrainer(model, peak_lr=1e-3, warmup_steps=100, total_steps=10000)
batch = {'input_ids': torch.randint(0, 32000, (2, 64)), 'labels': torch.randint(0, 32000, (2, 64))}
loss = trainer.step(batch)
print(f'Loss: {loss:.4f}')
"
```

### Requirements Installation
```bash
pip install -r requirements.txt
```

---

## Code Style Guidelines

### Imports
```python
# Standard library first
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

# Local imports last
from v6_core.architecture.some_module import SomeClass
```

### Type Annotations
- Use `typing` module: `Optional[str]`, `Tuple[int, int]`, `Dict[str, torch.Tensor]`
- For PyTorch: `torch.Tensor` (not just `Tensor`)
- Return types in docstrings when type hints aren't used

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `NexusV6`, `TreeGuidedEvolution` |
| Functions | snake_case | `build_nexus_small`, `get_layer_priority` |
| Variables | snake_case | `expert_weights`, `hidden_states` |
| Constants | UPPER_SNAKE | `MAX_SEQ_LEN`, `NUM_EXPERTS` |
| Private | _leading_underscore | `_internal_state`, `_compute_attention` |

### Docstrings
```python
class MyModule(nn.Module):
    """
    Brief description of what this module does.

    Longer explanation if needed, covering:
    - Key algorithms or techniques
    - Input/output contracts
    - Any side effects or state changes

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of what is returned.
    """
```

### Error Handling
```python
# Use explicit checks with clear messages
if x is None:
    raise ValueError("x must not be None")

# For tensor shape issues, include actual vs expected
if actual_shape != expected_shape:
    raise RuntimeError(f"Expected shape {expected_shape}, got {actual_shape}")

# Catch and re-raise with context
try:
    result = risky_operation()
except Exception as e:
    raise RuntimeError(f"Failed to do X: {e}") from e
```

### Module Structure
```python
"""
Module Title
============

Brief description of module purpose.

Key components:
- Component A: what it does
- Component B: what it does
"""

import torch
import torch.nn as nn
# ...

# ============================================================================
# SECTION: Component Group Name
# ============================================================================

class ComponentA(nn.Module):
    """Docstring for component."""
    pass

# ============================================================================
# SECTION: Another Group
# ============================================================================

class ComponentB(nn.Module):
    """Docstring for component."""
    pass
```

### PyTorch-Specific Guidelines

1. **Device handling**: Use `to(device)` consistently, not `.cuda()` directly
2. **No in-place ops that break autograd**: Avoid `.data` mutation when possible
3. **Module registration**: Use `nn.Module` subclasses, not bare `nn.functional`
4. **State management**: Use `register_buffer` for non-learnable state
5. **Memory efficiency**: Consider `torch.no_grad()` for inference

### CUDA Compatibility

Standard PyTorch CUDA ops are fully supported.

### Testing Guidelines

1. **Smoke tests**: Verify forward pass works before claiming features work
2. **NaN checks**: Always check `torch.isnan()` and `torch.isinf()` on outputs
3. **Gradient checks**: Verify `loss.backward()` doesn't produce NaN gradients
4. **Shape verification**: Assert tensor shapes match expected dimensions
5. **Synthetic data**: Use `torch.randint()` for quick tests, real data for validation

```python
# Basic test template
def test_my_module():
    model = MyModule()
    x = torch.randn(2, 32, 128)
    
    # Forward
    out = model(x)
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.shape == (2, 32, 128), f"Expected (2, 32, 128), got {out.shape}"
    
    # Backward
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(model.weight.grad).any(), "NaN in gradients"
```

---

## File Organization

```
v6_core/
├── architecture/           # Neural network components
│   ├── __init__.py        # Exports all public APIs
│   ├── nexus_v6.py        # Main NEXUS V6 architecture (1117 lines)
│   ├── dml_ops.py         # (legacy) DirectML-compatible operations
│   ├── v6_assembly.py     # Model assembly
│   └── [other components]
├── training/              # Training scripts
│   ├── nexus_train.py     # Training loop
│   ├── validate_nexus.py  # Validation suite
│   └── v6_full_train.py   # Full training script
└── test_v6_smoke.py       # Quick smoke test
```

---

## Common Patterns

### Factory Functions (Model Builders)
```python
def build_nexus_small():
    """Small NEXUS model (~157M params)."""
    return NexusV6(
        vocab_size=32000,
        d_model=768,
        num_layers=16,
        num_experts=6,
        num_concepts=12,
        d_state=16
    )
```

### Forward Pass Returns
```python
# Standard pattern: return logits and info dict
def forward(self, x):
    # ... computation ...
    return logits, {'expert_weights': weights, 'pooled': pooled}
```

### Device-Agnostic Code
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
```
