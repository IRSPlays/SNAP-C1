import sys, torch
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.model import EidosV1

m = EidosV1(
    vocab_size=10000, d_model=512, n_heads=8, n_kv_heads=4,
    n_layers=4, dropout=0.0, num_values=torch.rand(10000)
)
counts = m.count_parameters()
print(f"Total: {counts['total']:,}")
print(f"Encoder: {counts['encoder']:,}")
print(f"MTP heads: {counts['mtp_heads']:,}")
print(f"LTC: {counts['ltc_cortex']:,}")
print(f"Pred coder: {counts['predictive_coder']:,}")
print(f"Neural mem: {counts['neural_memory']:,}")

mtp = m.mtp
main_params = sum(p.numel() for p in mtp.main_head.parameters())
shared_params = sum(p.numel() for p in mtp.shared_head.parameters())
norm_params = sum(p.numel() for n in mtp.extra_norms for p in n.parameters())
mtp_extra = shared_params + norm_params
print(f"\nMTP breakdown:")
print(f"  main_head: {main_params:,} (tied to embed)")
print(f"  shared_head: {shared_params:,}")
print(f"  extra_norms (3x): {norm_params:,}")
print(f"  extra total: {mtp_extra:,}")
old_extra = 3 * (512 * 10000)
print(f"  OLD extra (3 separate): {old_extra:,}")
print(f"  SAVED: {old_extra - mtp_extra:,} ({100*(old_extra - mtp_extra)/old_extra:.0f}%)")

# NeuralMemory savings
print(f"\nNeuralMemory:")
print(f"  dim_mem={m.neural_memory.dim_mem}")
print(f"  M buffer: {m.neural_memory.dim_mem}x{m.neural_memory.dim_mem} = {m.neural_memory.dim_mem**2:,} elts")
print(f"  OLD M buffer: 512x512 = 262,144 elts")
print(f"  Reduction: {262144 / m.neural_memory.dim_mem**2:.1f}x")
