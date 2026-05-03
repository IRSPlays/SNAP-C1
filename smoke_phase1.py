"""Quick 2-epoch speed test with full Phase 1 config (AMP + compile + bmm)."""
import torch, sys, os, json, math, random, time
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab, encode_texts
from cortex.train import build_num_values, MathDataset
from torch.utils.data import DataLoader

device = torch.device('cuda')
enc = get_tokenizer()
base_dir = 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/data'

with open(os.path.join(base_dir, 'synthetic', 'train.jsonl'), encoding='utf-8') as f:
    rows = [json.loads(line) for line in f][:2000]
texts = [f"Q: {r['instruction']}\nA: {r['output']}" for r in rows]
random.shuffle(texts)
bpe_to_local, local_to_bpe, VOCAB_SIZE = build_restricted_vocab(texts, enc, min_count=2)
num_values = build_num_values(local_to_bpe, VOCAB_SIZE, enc)
examples = encode_texts(texts, enc, 96, bpe_to_local)  # Phase 1 seq_len=96
print(f"Vocab: {VOCAB_SIZE}, Examples: {len(examples)}")

model = EidosV1(
    vocab_size=VOCAB_SIZE, d_model=512, n_heads=8, n_kv_heads=4,
    n_layers=4, dropout=0.2, num_values=num_values,
).to(device)
model.train()
model.neural_memory.reset()

# torch.compile (skip on Windows — needs Triton)
print("Compiling... ", end="", flush=True)
try:
    import triton
    model = torch.compile(model)
    print("OK (inductor)")
except ImportError:
    print("skipped (no Triton on Windows)")
except Exception as e:
    print(f"skipped: {e}")

dataset = MathDataset(examples, [0.0] * len(examples))
loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)  # Phase 1 batch=4

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda', enabled=True)  # Phase 1 AMP
ACCUM = 2  # Phase 1 accum=2

optimizer.zero_grad(set_to_none=True)
total_batches = 0
start = time.time()

for epoch in range(2):
    for batch_idx, (inp, lbl, _) in enumerate(loader):
        inp, lbl = inp.to(device), lbl.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(inp, labels=lbl)
            loss = out['loss'] / ACCUM
        
        if not torch.isfinite(loss):
            print(f"  NaN at batch {batch_idx}!")
            break
        
        scaler.scale(loss).backward()
        total_batches += 1
        
        if (batch_idx + 1) % ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if total_batches % 20 == 0:
                elapsed = time.time() - start
                bps = total_batches / max(elapsed, 1e-5)
                print(f"  step {total_batches:4d}: loss={out['loss'].item():.4f} "
                      f"num={out.get('num_loss', torch.tensor(0)).item():.5f} "
                      f"| {bps:.1f} steps/s")
    
    print(f"  Epoch {epoch+1} done")

total_time = time.time() - start
print(f"\nTotal: {total_batches} batches in {total_time:.1f}s = {total_batches/total_time:.1f} steps/s")
print(f"Per-epoch estimate: {total_time/2:.1f}s for {len(loader)//ACCUM} steps")
