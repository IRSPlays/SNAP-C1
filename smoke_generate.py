import torch, sys, os, json
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')
from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab, encode_texts
from cortex.train import build_num_values

device = torch.device('cuda')
enc = get_tokenizer()

with open('C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1/data/synthetic/train.jsonl', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f][:50]
texts = [f"Q: {r['instruction']}\nA: {r['output']}" for r in rows]

bpe_to_local, local_to_bpe, VOCAB_SIZE = build_restricted_vocab(texts, enc, min_count=2)
num_values = build_num_values(local_to_bpe, VOCAB_SIZE, enc)
print(f"Vocab: {VOCAB_SIZE}")

model = EidosV1(
    vocab_size=VOCAB_SIZE, d_model=512, n_heads=8, n_kv_heads=4,
    n_layers=4, dropout=0.0, num_values=num_values,
).to(device)
model.eval()

eot = bpe_to_local.get(enc.eot_token, 0)

# Test generation with all features
prompt = "Q: What is 5 + 3?\nA:"
pids = enc.encode(prompt, allowed_special={'<|endoftext|>'})
pids = [bpe_to_local.get(t, 0) for t in pids]
p = torch.tensor([pids], device=device)

print("Testing generate (skip LTC + self-consistency):")
out = model.generate(p, max_new_tokens=30, temperature=0.0, top_k=1, eos_token_id=eot)
gen_ids = out[0].tolist()[len(pids):]
if eot in gen_ids:
    gen_ids = gen_ids[:gen_ids.index(eot)]
gen_bpe = [local_to_bpe.get(g, -1) for g in gen_ids if g > 0]
gen_text = enc.decode([x for x in gen_bpe if x >= 0])
print(f"  Prompt: {prompt}")
print(f"  Generated: {gen_text}")

# Test self-verification (force high threshold so it triggers)
print("\nTesting self-verification trigger:")
out2 = model.generate(p, max_new_tokens=30, temperature=0.0, top_k=1,
                      eos_token_id=eot,
                      enable_self_verify=True, verify_threshold=0.01, verify_retries=1)
gen_ids2 = out2[0].tolist()[len(pids):]
if eot in gen_ids2:
    gen_ids2 = gen_ids2[:gen_ids2.index(eot)]
gen_bpe2 = [local_to_bpe.get(g, -1) for g in gen_ids2 if g > 0]
gen_text2 = enc.decode([x for x in gen_bpe2 if x >= 0])
print(f"  With verify: {gen_text2}")

print("\nAll generate features OK")
