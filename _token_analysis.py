import json, tiktoken, os

enc = tiktoken.get_encoding('gpt2')

total_template_tokens = 0
total_simple_tokens = 0

for source in ['team_thinking/train.jsonl', 'team_thinking/train_v1_backup.jsonl',
               'team_thinking/train_v3_agentic.jsonl',
               'self_correction/train.jsonl', 'tool_use/train.jsonl', 'simple_qa/train.jsonl']:
    path = os.path.join('data', source)
    if not os.path.exists(path):
        continue
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f]
    total_toks = 0
    for r in rows:
        text = f"Q: {r['instruction']}\nA: {r['output']}"
        total_toks += len(enc.encode(text))
    avg_toks = total_toks / len(rows) if rows else 0
    is_simple = 'simple_qa' in source
    if is_simple:
        total_simple_tokens += total_toks
    else:
        total_template_tokens += total_toks
    print(f'{source}: {len(rows)} rows, {total_toks} tokens, avg={avg_toks:.0f} tok/example')

print(f'\nTemplate sources: {total_template_tokens} tokens')
print(f'Simple QA source: {total_simple_tokens} tokens')
print(f'Token ratio simple/(simple+template): {total_simple_tokens/(total_simple_tokens+total_template_tokens)*100:.1f}%')
print(f'Template dominance: {total_template_tokens/max(total_simple_tokens,1):.1f}x more template tokens')

# Also check: how many chunks will be pure-template vs pure-simple
# given seq_len=256 and no-overlap stride
print(f'\n--- Estimated chunk composition ---')
print(f'If tokens were separate: template would fill ~{total_template_tokens//256} chunks, simple ~{total_simple_tokens//256} chunks')
