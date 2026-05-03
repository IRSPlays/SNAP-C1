import torch, sys, os, json, time
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab, encode_texts
from cortex.train import build_num_values, MathDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda')
enc = get_tokenizer()
base_dir = 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/data'

with open(os.path.join(base_dir, 'synthetic', 'train.jsonl'), encoding='utf-8') as f:
    rows = [json.loads(line) for line in f][:2000]
texts = [f"Q: {r['instruction']}\nA: {r['output']}" for r in rows]
bpe_to_local, local_to_bpe, VOCAB_SIZE = build_restricted_vocab(texts, enc, min_count=2)
num_values = build_num_values(local_to_bpe, VOCAB_SIZE, enc)
examples = encode_texts(texts, enc, 96, bpe_to_local)
print(f"Vocab: {VOCAB_SIZE}, Examples: {len(examples)}")

configs = [
    ("OLD: batch=2, seq=192, fp32", 2, 192, False),
    ("NEW: batch=4, seq=96,  fp16", 4, 96,  True),
]

for label, bs, seq_len, use_amp in configs:
    # Override seq_len for this test (we encoded at 96, need to handle older seq too)
    if seq_len > 96:
        examples_v2 = encode_texts(texts, enc, seq_len, bpe_to_local)
    else:
        examples_v2 = examples
    
    model = EidosV1(
        vocab_size=VOCAB_SIZE, d_model=512, n_heads=8, n_kv_heads=4,
        n_layers=4, dropout=0.2, num_values=num_values,
    ).to(device)
    model.train()
    model.neural_memory.reset()
    
    dataset = MathDataset(examples_v2, [0.0] * len(examples_v2))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Warmup 2 batches
    for i, (inp, lbl, _) in enumerate(loader):
        if i >= 2:
            break
        inp, lbl = inp.to(device), lbl.to(device)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(inp, labels=lbl)
            out['loss'].backward()
        else:
            out = model(inp, labels=lbl)
            out['loss'].backward()
        opt.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    start = time.time()
    
    n = 0
    for i, (inp, lbl, _) in enumerate(loader):
        if i >= 40:
            break
        inp, lbl = inp.to(device), lbl.to(device)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(inp, labels=lbl)
        else:
            out = model(inp, labels=lbl)
        out['loss'].backward()
        opt.zero_grad(set_to_none=True)
        n += 1
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_batch = bs * seq_len
    tokens_per_sec = n * tokens_per_batch / elapsed
    steps_per_sec = n / elapsed
    print(f"  {label}")
    print(f"    {steps_per_sec:.1f} steps/s, {tokens_per_sec:.0f} tokens/s, {elapsed:.1f}s")
    
    del model, opt, loader, dataset
    torch.cuda.empty_cache()
