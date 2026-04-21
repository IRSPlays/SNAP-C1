"""Temporary analysis script — delete after use."""
import json, tiktoken
from collections import Counter

enc = tiktoken.get_encoding('gpt2')
files = [
    ('data/team_thinking/train.jsonl', 'team_think_train'),
    ('data/team_thinking/train_v1_backup.jsonl', 'team_think_v1'),
    ('data/team_thinking/train_v3_agentic.jsonl', 'team_think_v3'),
    ('data/self_correction/train.jsonl', 'self_corr'),
    ('data/tool_use/train.jsonl', 'tool_use'),
]
total_tokens = 0
all_counter = Counter()
for path, name in files:
    lens = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            text = f"Q: {row['instruction']}\nA: {row['output']}"
            toks = enc.encode(text)
            lens.append(len(toks))
            all_counter.update(toks)
    total_tokens += sum(lens)
    print(f'{name}: {len(lens)} examples, avg={sum(lens)/max(len(lens),1):.0f} toks, min={min(lens)}, max={max(lens)}, total={sum(lens)}')

print(f'\nTOTAL: {total_tokens} tokens across all train data')
print(f'With seq_len=256: ~{total_tokens // 256} full chunks')
print(f'Unique BPE tokens: {len(all_counter)}')

# Token frequency distribution
counts = sorted(all_counter.values(), reverse=True)
print(f'Top-10 token counts: {counts[:10]}')
print(f'Tokens appearing 1x: {sum(1 for c in counts if c == 1)}')
print(f'Tokens appearing <=3x: {sum(1 for c in counts if c <= 3)}')
print(f'Tokens appearing <=5x: {sum(1 for c in counts if c <= 5)}')

# Entropy calculation
import math
total = sum(counts)
entropy = -sum((c/total) * math.log(c/total) for c in counts)
print(f'\nUnigram entropy: {entropy:.3f} nats (ppl={math.exp(entropy):.1f})')
print(f'This is the THEORETICAL MINIMUM loss for a unigram model')
