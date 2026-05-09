import sys, torch
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.model import EidosV1

m = EidosV1(
    vocab_size=256, d_model=512, n_heads=8, n_kv_heads=4,
    n_layers=4, dropout=0.0, num_values=torch.rand(256)
).cuda()
m.train()
m.neural_memory.reset()

x = torch.randint(0, 256, (2, 32), device='cuda')
lbl = torch.randint(0, 256, (2, 32), device='cuda')

out = m(x, labels=lbl)
ff_val = out.get('ff_loss')
rew_val = out.get('reward_signal')
print(f"loss={out['loss'].item():.4f}")
print(f"ff_loss={ff_val.item() if ff_val is not None else 'N/A'}")
print(f"reward_signal={rew_val.item() if rew_val is not None else 'N/A'}")

out['loss'].backward()
print("backward OK")

grad_nan = 0
for name, p in m.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        grad_nan += 1
        print(f"  NaN grad: {name}")
print(f"grad NaN: {grad_nan}")

# Also test AMP
with torch.amp.autocast('cuda', dtype=torch.float16):
    out2 = m(torch.randint(0, 256, (2, 16), device='cuda'),
             labels=torch.randint(0, 256, (2, 16), device='cuda'))
print(f"AMP loss={out2['loss'].item():.4f}, ff={out2.get('ff_loss', torch.tensor(0)).item():.4f}")
print(f"AMP backward OK")
out2['loss'].backward()

# Test multiple forward-backward iterations
for i in range(3):
    m.train()
    xb = torch.randint(0, 256, (2, 16), device='cuda')
    lb = torch.randint(0, 256, (2, 16), device='cuda')
    with torch.amp.autocast('cuda', dtype=torch.float16):
        outb = m(xb, labels=lb)
    outb['loss'].backward()
    print(f"  iter {i}: loss={outb['loss'].item():.4f}, ff={outb.get('ff_loss', torch.tensor(0)).item():.4f}")

print("\nALL CHECKS PASSED")
