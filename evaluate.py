"""Eidos V1 evaluation — load checkpoint, evaluate GSM8K, show full Q/A pairs."""
import torch, sys, os, re, json
sys.path.insert(0, '.')
from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab
from cortex.train import build_num_values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

# Load checkpoint
checkpoint_path = os.path.join(os.path.dirname(__file__), 'cortex', 'checkpoints', 'eidos_v1_best.pt')
if not os.path.exists(checkpoint_path):
    print(f'Checkpoint not found: {checkpoint_path}')
    sys.exit(1)

ckpt = torch.load(checkpoint_path, map_location=device)
config = ckpt['config']
bpe_to_local = ckpt['bpe_to_local']
local_to_bpe = ckpt['local_to_bpe']
epoch = ckpt['epoch']
best_eval = ckpt['eval_loss']
print(f'Checkpoint: epoch {epoch+1}, best eval loss = {best_eval:.4f}')

enc = get_tokenizer()

# Load eval data
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

base_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data'))
eval_path = os.path.join(base_data_dir, 'gsm8k', 'eval.jsonl')

if not os.path.exists(eval_path):
    print(f'Eval file not found: {eval_path}')
    sys.exit(1)

rows = load_jsonl(eval_path)
print(f'GSM8K eval questions: {len(rows)}')

# Rebuild number value map (needed for model init)
num_values = build_num_values(local_to_bpe, len(local_to_bpe), enc)

# Build model
model = EidosV1(
    vocab_size=len(local_to_bpe),
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    n_kv_heads=config['n_kv_heads'],
    n_layers=config['n_layers'],
    dropout=0.0,
    num_values=num_values,
).to(device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

total_p = model.count_parameters()['total']
print(f'Model params: {total_p:,}')
print(f'Vocab: {len(local_to_bpe)} tokens')

eot_local = bpe_to_local.get(enc.eot_token, 0)

def extract_number(text):
    """Extract the final numeric answer."""
    patterns = [
        r'Answer:\s*(\d+(?:\.\d+)?)',
        r'answer is\s*(\d+(?:\.\d+)?)',
        r'####\s*(\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    return nums[-1] if nums else None

def extract_gt_answer(gt_text):
    m = re.search(r'####\s*(\d+(?:\.\d+)?)', gt_text)
    return m.group(1) if m else None

# Evaluate
num_samples = min(30, len(rows))
print(f'\n{"="*80}')
print(f'EVALUATING {num_samples} QUESTIONS')
print(f'{"="*80}\n')

correct = 0
total_tested = 0

for idx, row in enumerate(rows[:num_samples]):
    # Try different possible field names
    if 'instruction' in row:
        question = row['instruction']
    elif 'question' in row:
        question = row['question']
    else:
        question = list(row.values())[0] if row else str(row)

    if 'output' in row:
        output_raw = row['output']
    elif 'answer' in row:
        output_raw = row['answer']
    else:
        output_raw = list(row.values())[-1] if row else ''

    gt_answer = extract_gt_answer(output_raw)
    if gt_answer is None:
        # Try to find any number in the output
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', output_raw)
        gt_answer = nums[-1] if nums else None

    prompt = f"Q: {question}\nA:"

    prompt_ids = enc.encode(prompt, allowed_special={'<|endoftext|>'})
    prompt_ids = [bpe_to_local.get(t, 0) for t in prompt_ids]
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
        out = model.generate(
            prompt_tensor,
            max_new_tokens=200,
            temperature=0.0,
            top_k=1,
            eos_token_id=eot_local
        )

    gen_ids = out[0].tolist()[len(prompt_ids):]
    if eot_local in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eot_local)]

    gen_bpe = [local_to_bpe.get(g, -1) for g in gen_ids]
    gen_bpe = [g for g in gen_bpe if g >= 0]
    gen_text = enc.decode(gen_bpe)

    pred_answer = extract_number(gen_text)
    is_correct = (pred_answer is not None and gt_answer is not None and
                  abs(float(pred_answer) - float(gt_answer)) < 0.01)

    if is_correct:
        correct += 1
    total_tested += 1

    marker = '[CORRECT]' if is_correct else '[WRONG]'
    print(f'--- Question {idx+1}/{num_samples} {marker} ---')
    print(f'Q: {question}')
    print(f'A: {gen_text}')
    print(f'   Pred: {pred_answer or "N/A"} | GT: {gt_answer or "N/A"}')
    print()

print(f'{"="*80}')
print(f'ACCURACY: {correct}/{total_tested} = {100*correct/max(total_tested,1):.1f}%')
print(f'{"="*80}')
