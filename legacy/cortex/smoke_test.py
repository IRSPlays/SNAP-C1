"""Quick smoke test for CortexV1."""
import torch
from cortex.model import CortexV1

model = CortexV1(vocab_size=8192, d_model=256, use_memory=False)
for k, v in model.count_parameters().items():
    print(f"  {k}: {v:,}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

x = torch.randint(0, 8192, (4, 64), device=device)
out = model(x)
print(f"\nlogits shape: {out['logits'].shape}")
print(f"pred_error shape: {out['prediction_error'].shape}")
print(f"h_final shape: {out['h_final'].shape}")

loss = out["logits"].sum()
loss.backward()
print("backward OK")

has_grad = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        mod = name.split(".")[0]
        has_grad[mod] = has_grad.get(mod, 0) + p.grad.norm().item()

print("\nGradient flow:")
for m, g in has_grad.items():
    print(f"  {m}: {g:.4f}")

# Memory test (Phase 2 path)
print("\n--- Phase 2 (with memory) ---")
model2 = CortexV1(vocab_size=8192, d_model=256, use_memory=True).to(device)
out2 = model2(x)
print(f"logits shape: {out2['logits'].shape}")
print(f"memories stored: {model2.memory.size}")

print("\nSMOKE TEST PASSED")
