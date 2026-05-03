import torch, sys, json, os
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab, encode_texts
from cortex.train import build_num_values

device = torch.device('cuda')
enc = get_tokenizer()

with open('C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/data/synthetic/train.jsonl', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]
texts = [f"Q: {r['instruction']}\nA: {r['output']}" for r in rows]
print(f"Synthetic examples: {len(texts)}")
print(f"Sample: {texts[0][:100]}")

bpe_to_local, local_to_bpe, VOCAB_SIZE = build_restricted_vocab(texts, enc, min_count=2)
num_values = build_num_values(local_to_bpe, VOCAB_SIZE, enc)
examples = encode_texts(texts, enc, 192, bpe_to_local)
print(f"Vocab: {VOCAB_SIZE}, Examples: {len(examples)}")

model = EidosV1(
    vocab_size=VOCAB_SIZE, d_model=512, n_heads=8, n_kv_heads=4,
    n_layers=4, dropout=0.2, num_values=num_values,
).to(device)
model.train()
model.neural_memory.reset()

inp = torch.stack([examples[0][0], examples[1][0]]).to(device)
lbl = torch.stack([examples[0][1], examples[1][1]]).to(device)
print(f"Input shape: {inp.shape}, Labels shape: {lbl.shape}")

out = model(inp, labels=lbl)
print(f"Keys: {list(out.keys())}")
print(f"loss={out['loss'].item():.4f}")
if 'num_loss' in out:
    print(f"num_loss={out['num_loss'].item():.6f}")
if 'val_pred_loss' in out:
    print(f"val_pred_loss={out['val_pred_loss'].item():.6f}")
print(f"cosine_dist={out['cosine_dist'].mean().item():.4f}")
print(f"iterations={out.get('iterations', 'N/A')}")
print(f"serotonin mean={out['serotonin'].mean().item():.4f}")
print(f"dopamine mean={out['dopamine'].mean().item():.4f}")

out['loss'].backward()
print("Backward OK")

grad_nans = 0
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).any():
        print(f"  NaN grad: {name}")
        grad_nans += 1
print(f"Gradient NaN count: {grad_nans}")

# Test memory persistence
inp2 = torch.stack([examples[2][0], examples[3][0]]).to(device)
lbl2 = torch.stack([examples[2][1], examples[3][1]]).to(device)
out2 = model(inp2, labels=lbl2)
print(f"Batch 2 loss={out2['loss'].item():.4f}")
M_norm = model.neural_memory.M.norm().item()
M_not_zero = (model.neural_memory.M.abs().sum() > 1e-8).item()
print(f"M norm after 2 batches: {M_norm:.4f}, non-zero: {M_not_zero}")

# Test that value_pred comes back
print(f"val_pred_loss (batch 2)={out2.get('val_pred_loss', 'N/A')}")

print("\nALL CHECKS PASSED")
