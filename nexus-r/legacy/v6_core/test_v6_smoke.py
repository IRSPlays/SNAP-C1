"""
SNAP-C1 V6: Smoke Test
=======================
Quick test to verify V6 model builds and runs on DirectML.
"""

import torch

# Test imports
print("Testing imports...")
from v6_core.architecture import (
    V6ResonanceModel, build_v6_local, build_v6_small,
    get_device, RMSNorm, stable_sigmoid
)
print("  All imports OK")

# Test device detection
device = get_device()
print(f"  Device: {device}")

# Build small model for testing
print("\nBuilding V6 small model...")
model = build_v6_small(use_skip=True, dropout=0.0)
model = model.to(device)
print(f"  Model built OK")

# Count parameters
params = model.count_parameters()
print(f"\nParameter counts:")
for k, v in params.items():
    if isinstance(v, int):
        print(f"  {k}: {v:,}")
    else:
        print(f"  {k}: {v}")

# Test forward pass (pretraining)
print("\nTesting forward_pretrain...")
B, T = 2, 128
token_ids = torch.randint(0, 10000, (B, T), device=device)
type_ids = torch.zeros(B, T, dtype=torch.long, device=device)
labels = torch.randint(0, 10000, (B, T), device=device)

with torch.no_grad():
    result = model.forward_pretrain(token_ids, type_ids, labels)
    print(f"  Logits shape: {result['logits'].shape}")
    print(f"  Loss: {result['loss'].item():.4f}")

# Test forward_agent
print("\nTesting forward_agent...")
with torch.no_grad():
    result = model.forward_agent(token_ids, type_ids)
    print(f"  Tool ID: {result['tool_id']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  P(success): {result['p_success']}")

# Test skip rate
skip_rate = model.get_skip_rate()
print(f"\n  Skip rate: {skip_rate:.2%}")

# Test with use_skip=False
print("\n\nTesting WITHOUT skip...")
model_no_skip = build_v6_small(use_skip=False)
model_no_skip = model_no_skip.to(device)

with torch.no_grad():
    result = model_no_skip.forward_pretrain(token_ids, type_ids, labels)
    print(f"  Loss: {result['loss'].item():.4f}")

print("\n✓ All tests passed!")
print("\nV6 is ready for training.")
