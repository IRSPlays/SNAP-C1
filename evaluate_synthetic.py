"""Eidos Synthetic Arithmetic Evaluation — test accuracy on generated math problems.

Evaluates a checkpoint on fresh synthetic arithmetic problems across 4 levels:
  Level 1: Pure operations (+, -, *, /)
  Level 2: Multi-step chains
  Level 3: Word-form problems
  Level 4: GSM8K-style word problems

Usage:
    python evaluate_synthetic.py                    # evaluate best checkpoint
    python evaluate_synthetic.py --num 200          # test on 200 problems
    python evaluate_synthetic.py --checkpoint path  # evaluate specific checkpoint
"""

import torch, sys, os, re, json, argparse, random, math
sys.path.insert(0, '.')
from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab, encode_texts
from cortex.train import build_num_values
from cortex.synthetic_data import (
    gen_pure, gen_chain, gen_word_form, gen_mixed, gen_gsm8k_style,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint: epoch {ckpt.get('epoch', -1)+1}, eval loss = {ckpt.get('eval_loss', float('nan')):.4f}")
    return ckpt


def build_model(ckpt, local_to_bpe, device):
    config = ckpt['config']
    enc = get_tokenizer()
    num_values = build_num_values(local_to_bpe, len(local_to_bpe), enc)
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
    return model


def extract_answer_number(text):
    """Extract the final numeric answer after #### marker."""
    m = re.search(r'####\s*(\d+(?:\.\d+)?)', text)
    if m:
        return int(float(m.group(1)))
    nums = re.findall(r'\b(\d+)\b', text)
    return int(nums[-1]) if nums else None


def evaluate_one(model, question, enc, bpe_to_local, local_to_bpe, eot_local, device,
                 max_tokens=200):
    """Generate answer for one question, return (generated_text, predicted_number)."""
    prompt_ids = enc.encode(question, allowed_special={'<|endoftext|>'})
    prompt_ids = [bpe_to_local.get(t, 0) for t in prompt_ids]
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
        out = model.generate(
            prompt_tensor, max_new_tokens=max_tokens,
            temperature=0.0, top_k=1, eos_token_id=eot_local
        )
    gen_ids = out[0].tolist()[len(prompt_ids):]
    if eot_local in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eot_local)]
    gen_bpe = [local_to_bpe.get(g, -1) for g in gen_ids]
    gen_bpe = [g for g in gen_bpe if g >= 0]
    gen_text = enc.decode(gen_bpe)
    pred = extract_answer_number(gen_text)
    return gen_text, pred


def evaluate_level(model, label, gen_fn, count, enc, bpe_to_local, local_to_bpe,
                   eot_local, device):
    """Evaluate a batch of problems from one generator."""
    correct = 0
    tested = 0
    results = []
    for _ in range(count):
        try:
            question, chain, answer = gen_fn()
            gen_text, pred = evaluate_one(
                model, f"Q: {question}\nA:", enc, bpe_to_local, local_to_bpe, eot_local, device
            )
            is_correct = (pred is not None and abs(pred - answer) < 0.5)
            if is_correct:
                correct += 1
            tested += 1
            results.append((question, answer, pred, gen_text, is_correct))
        except Exception as e:
            continue
    acc = 100 * correct / max(tested, 1)
    return acc, correct, tested, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Eidos on synthetic arithmetic")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint.pt (default: cortex/checkpoints/eidos_v1_best.pt)')
    parser.add_argument('--num', type=int, default=100,
                        help='Number of test problems per level')
    parser.add_argument('--level', type=str, default='all',
                        choices=['all', 'pure', 'chain', 'word', 'mixed', 'gsm8k_style'],
                        help='Which difficulty level to evaluate')
    parser.add_argument('--show-errors', type=int, default=5,
                        help='Number of wrong answers to display per level')
    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try pretrain checkpoint first, fallback to finetune
        pretrain_path = os.path.join(
            os.path.dirname(__file__), 'cortex', 'checkpoints', 'eidos_pretrain_best.pt'
        )
        finetune_path = os.path.join(
            os.path.dirname(__file__), 'cortex', 'checkpoints', 'eidos_v1_best.pt'
        )
        if os.path.exists(pretrain_path):
            checkpoint_path = pretrain_path
        elif os.path.exists(finetune_path):
            checkpoint_path = finetune_path
        else:
            print("No checkpoint found. Train first with:")
            print("  python -m cortex.train --pretrain")
            print("  python -m cortex.train")
            sys.exit(1)
    ckpt = load_checkpoint(checkpoint_path, device)
    bpe_to_local = ckpt['bpe_to_local']
    local_to_bpe = ckpt['local_to_bpe']

    enc = get_tokenizer()
    model = build_model(ckpt, local_to_bpe, device)
    eot_local = bpe_to_local.get(enc.eot_token, 0)
    print(f"Model: {model.count_parameters()['total']:,} params, vocab={len(local_to_bpe)} tokens\n")

    # Define levels
    levels = [
        ("PURE (+, -, *, /)", gen_pure, args.num),
        ("CHAINS (2-4 steps)", gen_chain, args.num),
        ("WORD FORM (text numbers)", gen_word_form, args.num),
        ("MIXED (multi-digit)", gen_mixed, args.num),
        ("GSM8K-STYLE (word problems)", gen_gsm8k_style, args.num),
    ]

    if args.level != 'all':
        level_map = {
            'pure': levels[0],
            'chain': levels[1],
            'word': levels[2],
            'mixed': levels[3],
            'gsm8k_style': levels[4],
        }
        levels = [level_map[args.level]]

    print("=" * 80)
    total_correct = 0
    total_tested = 0
    all_wrong = []

    for label, gen_fn, count in levels:
        print(f"  Evaluating: {label} ({count} problems)...", end=' ', flush=True)
        acc, correct, tested, results = evaluate_level(
            model, label, gen_fn, count, enc, bpe_to_local, local_to_bpe, eot_local, device
        )
        print(f"{acc:.1f}% ({correct}/{tested})")
        total_correct += correct
        total_tested += tested
        all_wrong.extend([(label, r) for r in results if not r[4]])

    overall = 100 * total_correct / max(total_tested, 1)
    print(f"\n{'=' * 80}")
    print(f"OVERALL: {total_correct}/{total_tested} = {overall:.1f}%")
    print(f"{'=' * 80}")

    # Show sample errors
    if all_wrong and args.show_errors > 0:
        print(f"\n  --- Sample Errors (showing {min(args.show_errors, len(all_wrong))}) ---")
        shown = 0
        for label, (question, answer, pred, gen_text, _) in all_wrong:
            if shown >= args.show_errors:
                break
            print(f"\n  [{label}] Q: {question}")
            print(f"  GT: {answer}  |  Pred: {pred if pred is not None else 'N/A'}")
            print(f"  Gen: {gen_text[:150]}")
            shown += 1


if __name__ == '__main__':
    main()
