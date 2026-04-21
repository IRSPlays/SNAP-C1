"""Quick greedy-only eval suite runner."""
import argparse
import json
import os
import sys
import torch
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nexus_v1.training.eval_suite_runner import load_model, generate_answer, score_answer


def parse_args():
    default_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'nexus_r_v1_best.pt')
    default_eval_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'diverse_qa', 'eval_suite.jsonl')

    parser = argparse.ArgumentParser(description="Run the greedy-only Nexus-R eval suite")
    parser.add_argument('--checkpoint', default=default_checkpoint,
                        help='Path to the checkpoint to evaluate')
    parser.add_argument('--eval-path', default=default_eval_path,
                        help='Path to the JSONL evaluation suite')
    parser.add_argument('--relaxed-load', action='store_true',
                        help='Allow non-strict checkpoint loading')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = tiktoken.get_encoding('gpt2')

    model, cfg, bpe_to_local, local_to_bpe = load_model(
        args.checkpoint,
        device,
        strict=not args.relaxed_load,
    )

    if not os.path.exists(args.eval_path):
        raise FileNotFoundError(f"Evaluation suite not found: {args.eval_path}")

    with open(args.eval_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    results = {}
    all_scores = []

    for ex in examples:
        q = ex['instruction']
        expected = ex['output']
        category = ex.get('category', 'unknown')
        prompt = "Q: " + q + "\nA:"
        generated = generate_answer(model, enc, device, prompt, temperature=0.01, top_k=1,
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

    print("=" * 70)
    print("GREEDY EVALUATION SUMMARY (temp=0.01)")
    print("=" * 70)
    total_correct = total_partial = total_wrong = total_count = 0
    for cat in sorted(results.keys()):
        r = results[cat]
        avg_score = sum(r['scores']) / len(r['scores'])
        acc = r['correct'] / r['total'] * 100
        print("  %20s: %3d/%3d correct (%5.1f%%) | partial=%d wrong=%d | avg=%.3f" %
              (cat, r['correct'], r['total'], acc, r['partial'], r['wrong'], avg_score))
        total_correct += r['correct']
        total_partial += r['partial']
        total_wrong += r['wrong']
        total_count += r['total']
    overall_acc = total_correct / total_count * 100
    overall_avg = sum(all_scores) / len(all_scores)
    print()
    print("  %20s: %3d/%3d correct (%5.1f%%) | partial=%d wrong=%d | avg=%.3f" %
          ("OVERALL", total_correct, total_count, overall_acc, total_partial, total_wrong, overall_avg))

if __name__ == '__main__':
    main()
