import torch, sys, os, json, time
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab, encode_texts
from cortex.train import build_num_values, MathDataset
from torch.utils.data import DataLoader

device = torch.device('cuda')
enc = get_tokenizer()
base_dir = 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/data'

with open(os.path.join(base_dir, 'synthetic', 'train.jsonl'), encoding='utf-8') as f:
    rows = [json.loads(line) for line in f][:1000]
texts = [f"Q: {r['instruction']}\nA: {r['output']}" for r in rows]
bpe_to_local, local_to_bpe, VOCAB_SIZE = build_restricted_vocab(texts, enc, min_count=2)
num_values = build_num_values(local_to_bpe, VOCAB_SIZE, enc)
examples = encode_texts(texts, enc, 96, bpe_to_local)
print(f"Vocab: {VOCAB_SIZE}")

for bs in [2, 4]:
    model = EidosV1(
        vocab_size=VOCAB_SIZE, d_model=512, n_heads=8, n_kv_heads=4,
        n_layers=4, dropout=0.2, num_values=num_values,
    ).to(device)
    model.train()
    model.neural_memory.reset()
    dataset = MathDataset(examples, [0.0] * len(examples))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    arc = torch.amp.autocast('cuda', dtype=torch.float16)

    try:
        start = time.time()
        for i, (inp, lbl, _) in enumerate(loader):
            if i >= 10:
                break
            inp, lbl = inp.to(device), lbl.to(device)
            with arc:
                out = model(inp, labels=lbl)
                loss = out['loss'] / 2
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        elapsed = time.time() - start
        print(f"batch={bs}: 10 steps in {elapsed:.1f}s | loss={out['loss'].item():.4f} | OK")
    except Exception as e:
        print(f"batch={bs}: FAILED - {type(e).__name__}: {e}")
    del model, opt, scaler, loader, dataset
    torch.cuda.empty_cache()
