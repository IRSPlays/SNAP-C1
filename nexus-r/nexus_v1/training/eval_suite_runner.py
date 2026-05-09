"""
Evaluate Nexus-R V1 checkpoint against the held-out eval suite.
Runs greedy (temp=0.01) and sampled (temp=0.7) generation on all examples.
Reports accuracy per category and overall.
"""
import json
import os
import sys
import torch
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nexus_v1.architecture import NexusR, NexusConfig


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = NexusR(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    bpe_to_local = ckpt['bpe_to_local']
    local_to_bpe = ckpt['local_to_bpe']
    return model, cfg, bpe_to_local, local_to_bpe


@torch.no_grad()
def generate_answer(model, enc, device, prompt_text, max_tokens=100, temperature=0.01,
                    top_k=1, bpe_to_local=None, local_to_bpe=None):
    """Generate an answer for a prompt."""
    eot_local = bpe_to_local.get(enc.eot_token, 0) if bpe_to_local else enc.eot_token
    prompt_ids = enc.encode(prompt_text, allowed_special={'<|endoftext|>'})
    if bpe_to_local:
        prompt_ids = [bpe_to_local.get(t, 0) for t in prompt_ids]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(prompt_tensor, max_new_tokens=max_tokens,
                               temperature=temperature, top_k=top_k)
    gen_local_ids = generated[0].tolist()[len(prompt_ids):]

    if eot_local in gen_local_ids:
        gen_local_ids = gen_local_ids[:gen_local_ids.index(eot_local)]

    if local_to_bpe:
        gen_bpe_ids = [local_to_bpe.get(lid, 0) for lid in gen_local_ids if lid > 0]
    else:
        gen_bpe_ids = gen_local_ids

    return enc.decode(gen_bpe_ids).strip() if gen_bpe_ids else ""


def score_answer(generated, expected):
    """Simple scoring: check if the key content of expected appears in generated."""
    gen_lower = generated.lower().strip().rstrip('.')
    exp_lower = expected.lower().strip().rstrip('.')

    # Exact match (ignoring trailing period and case)
    if gen_lower == exp_lower:
        return 1.0

    # For short numeric answers, check if the number appears
    exp_stripped = exp_lower.replace(',', '').strip()
    gen_stripped = gen_lower.replace(',', '').strip()
    if exp_stripped.replace('.', '').replace('-', '').isdigit():
        if exp_stripped == gen_stripped:
            return 1.0
        if exp_stripped in gen_stripped.split():
            return 0.8

    # Check if key words from expected appear in generated
    exp_words = set(exp_lower.split())
    gen_words = set(gen_lower.split())
    # Remove common stop words
    stop = {'a', 'an', 'the', 'is', 'are', 'of', 'in', 'to', 'it', 'and', 'or', 'for', 'at', 'by', 'on'}
    exp_key = exp_words - stop
    gen_key = gen_words - stop
    if exp_key and len(exp_key & gen_key) / len(exp_key) > 0.6:
        return 0.7

    return 0.0


def evaluate_suite(checkpoint_path, eval_path, temperature=0.01, top_k=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    enc = tiktoken.get_encoding('gpt2')
    model, cfg, bpe_to_local, local_to_bpe = load_model(checkpoint_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params, vocab={cfg.vocab_size}")

    with open(eval_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    print(f"Evaluating {len(examples)} examples (temp={temperature}, top_k={top_k})")
    print("-" * 70)

    results = {}
    all_scores = []

    for i, ex in enumerate(examples):
        q = ex['instruction']
        expected = ex['output']
        category = ex.get('category', 'unknown')

        prompt = f"Q: {q}\nA:"
        generated = generate_answer(model, enc, device, prompt,
                                     temperature=temperature, top_k=top_k,
                                     bpe_to_local=bpe_to_local, local_to_bpe=local_to_bpe)
        score = score_answer(generated, expected)

        if category not in results:
            results[category] = {'correct': 0, 'partial': 0, 'wrong': 0, 'total': 0, 'scores': []}
        results[category]['total'] += 1
        results[category]['scores'].append(score)
        if score >= 0.9:
            results[category]['correct'] += 1
        elif score >= 0.5:
            results[category]['partial'] += 1
        else:
            results[category]['wrong'] += 1

        all_scores.append(score)

        # Print wrong/partial answers for debugging
        if score < 0.9:
            status = "PARTIAL" if score >= 0.5 else "WRONG"
            print(f"  [{status}] [{category}] Q: {q[:60]}")
            print(f"    Expected: {expected[:80]}")
            print(f"    Got:      {generated[:80]}")
            print()

    # Summary
    print("=" * 70)
    print(f"EVALUATION SUMMARY (temp={temperature})")
    print("=" * 70)

    total_correct = 0
    total_partial = 0
    total_wrong = 0
    total_count = 0

    for cat in sorted(results.keys()):
        r = results[cat]
        avg_score = sum(r['scores']) / len(r['scores'])
        acc = r['correct'] / r['total'] * 100
        print(f"  {cat:20s}: {r['correct']:3d}/{r['total']:3d} correct "
              f"({acc:5.1f}%) | partial={r['partial']} wrong={r['wrong']} | avg_score={avg_score:.3f}")
        total_correct += r['correct']
        total_partial += r['partial']
        total_wrong += r['wrong']
        total_count += r['total']

    overall_acc = total_correct / total_count * 100
    overall_avg = sum(all_scores) / len(all_scores)
    print(f"\n  {'OVERALL':20s}: {total_correct:3d}/{total_count:3d} correct "
          f"({overall_acc:5.1f}%) | partial={total_partial} wrong={total_wrong} | avg_score={overall_avg:.3f}")
    print()

    return results


if __name__ == '__main__':
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'nexus_r_v1_best.pt')
    eval_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'diverse_qa', 'eval_suite.jsonl')

    print("\n=== GREEDY EVALUATION (temp=0.01) ===\n")
    greedy_results = evaluate_suite(ckpt_path, eval_path, temperature=0.01, top_k=1)

    print("\n=== SAMPLED EVALUATION (temp=0.7) ===\n")
    sampled_results = evaluate_suite(ckpt_path, eval_path, temperature=0.7, top_k=40)
