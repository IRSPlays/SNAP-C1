"""Eidos V1 Training Script — Phase 1 (single-phase, end-to-end).

Everything is differentiable:
- Diff Attention: differentiable subtraction
- Neural Memory: momentum Hebbian update (differentiable)
- LTC: differentiable Euler integration
- MTP: separate CE losses summed

Loss = CE_main + 0.3·(CE_mtp1 + CE_mtp2 + CE_mtp3)

Usage:
    cd SNAP-C1
    python -m cortex.train
"""

import json
import os
import sys
import time
import math
import random
from contextlib import nullcontext
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Allow torch.compile to capture Tensor.item() calls in the graph
try:
    torch._dynamo.config.capture_scalar_outputs = True
except Exception:
    pass

from cortex.model import EidosV1
from cortex.tokenizer import get_tokenizer, build_restricted_vocab, encode_texts

import tiktoken


def format_qa(row: dict) -> str:
    return f"Q: {row['instruction']}\nA: {row['output']}"


def load_jsonl(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def build_num_values(local_to_bpe, vocab_size, enc):
    """Build a [vocab_size] tensor mapping token IDs to their numeric values.

    Uses log1p scaling so values stay bounded (~0–1.5) regardless of magnitude.
    Raw number "10000" → log1p(10000)/10 ≈ 0.92 instead of 10000.
    This prevents num_proj from injecting massive signals into embeddings.
    """
    num_vals = torch.zeros(vocab_size)
    for local_id in range(vocab_size):
        bpe_id = local_to_bpe.get(local_id, -1)
        if bpe_id < 0:
            continue
        try:
            text = enc.decode([bpe_id]).strip()
            val = float(text)
            # Log-scale: log1p(5)≈1.8, log1p(175)≈5.2, log1p(100000)≈11.5
            # Divided by 10 → range ~0 to ~1.15, clamped at 1.5
            scaled = min(math.log1p(val) / 10.0, 1.5)
            num_vals[local_id] = scaled
        except (ValueError, OverflowError):
            pass
    count_nonzero = (num_vals != 0).sum().item()
    print(f"  Number tokens: {count_nonzero}/{vocab_size}")
    return num_vals


def parse_answer_value(text: str) -> float:
    """Extract and normalize the numeric answer from a GSM8K output field.
    
    Uses log1p(x)/10 scaling (same as number value embedding) so regression
    targets stay in the same ~0-1.5 range as the embedding values.
    """
    import re
    m = re.search(r'####\s*(\d+(?:\.\d+)?)', text)
    if m:
        raw = float(m.group(1))
        return min(math.log1p(raw) / 10.0, 1.5)
    return 0.0


class MathDataset(torch.utils.data.Dataset):
    def __init__(self, examples, answer_values):
        self.examples = examples
        self.answer_values = answer_values

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inp, lbl = self.examples[idx]
        return inp, lbl, self.answer_values[idx]


def load_data(base_dir: str, use_synthetic: bool = False, synthetic_count: int = 50000):
    train_path = os.path.join(base_dir, 'gsm8k', 'train.jsonl')
    eval_path = os.path.join(base_dir, 'gsm8k', 'eval.jsonl')
    synthetic_path = os.path.join(base_dir, 'synthetic', 'train.jsonl')

    train_texts, eval_texts, train_answers = [], [], []
    if use_synthetic and os.path.exists(synthetic_path):
        rows = load_jsonl(synthetic_path)
        if len(rows) > synthetic_count:
            rows = rows[:synthetic_count]
        # 90/10 train/eval split for synthetic — measures genuine generalization
        random.shuffle(rows)
        split = int(len(rows) * 0.9)
        train_rows = rows[:split]
        eval_rows = rows[split:]
        train_texts = [format_qa(r) for r in train_rows]
        eval_texts = [format_qa(r) for r in eval_rows]
        train_answers = torch.tensor([parse_answer_value(r['output']) for r in train_rows],
                                       dtype=torch.float32)
        print(f"  Synthetic: {len(train_texts)} train, {len(eval_texts)} eval")
    elif os.path.exists(train_path):
        rows = load_jsonl(train_path)
        train_texts = [format_qa(r) for r in rows]
        train_answers = torch.tensor([parse_answer_value(r['output']) for r in rows],
                                       dtype=torch.float32)
        print(f"  GSM8K train: {len(train_texts)}")
    if not use_synthetic and os.path.exists(eval_path):
        rows = load_jsonl(eval_path)
        eval_texts = [format_qa(r) for r in rows]
        print(f"  GSM8K eval: {len(eval_texts)}")

    return train_texts, eval_texts, train_answers


def maybe_autocast(device, enabled, dtype=torch.float16):
    if enabled and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=dtype)
    return nullcontext()


def build_test_prompts(eval_texts, train_texts=None, limit=8):
    prompts = []
    seen = set()
    marker = "\nA:"

    # During pretrain, show training data first so user sees what model is learning
    if train_texts is not None:
        for text in train_texts:
            if marker in text:
                prompt = text.split(marker, 1)[0] + marker
            else:
                prompt = text.rstrip()
                if not prompt.endswith("A:"):
                    prompt = f"{prompt}\nA:"
            if prompt in seen:
                continue
            seen.add(prompt)
            full_qa = text.replace(marker, "  >>> ANSWER:")
            prompts.append((prompt, full_qa))
            if len(prompts) >= limit // 2:
                break

    for text in eval_texts:
        if marker in text:
            prompt = text.split(marker, 1)[0] + marker
        else:
            prompt = text.rstrip()
            if not prompt.endswith("A:"):
                prompt = f"{prompt}\nA:"
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append((prompt, None))
        if len(prompts) >= limit:
            break
    if not prompts:
        prompts = [("Q: What is 2 + 3?\nA:", None), ("Q: Capital of France?\nA:", None)]
    return prompts


@torch.no_grad()
def generate_samples(model, enc, device, prompts, bpe_to_local, local_to_bpe,
                     max_tokens=300):
    model.eval()
    results = []
    eot_local = bpe_to_local.get(enc.eot_token, 0) if bpe_to_local else enc.eot_token
    for prompt_data in prompts:
        if isinstance(prompt_data, tuple):
            prompt_text, known_answer = prompt_data
        else:
            prompt_text = prompt_data
            known_answer = None
        prompt_ids = enc.encode(prompt_text, allowed_special={'<|endoftext|>'})
        if bpe_to_local:
            prompt_ids = [bpe_to_local.get(t, 0) for t in prompt_ids]
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(prompt_tensor, max_new_tokens=max_tokens,
                                   temperature=0.0, top_k=1, eos_token_id=eot_local)
        gen_ids = generated[0].tolist()[len(prompt_ids):]
        if eot_local in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eot_local)]
        if local_to_bpe:
            gen_bpe = [local_to_bpe.get(g, 0) for g in gen_ids if g > 0]
        else:
            gen_bpe = gen_ids
        gen_text = enc.decode(gen_bpe) if gen_bpe else ""
        results.append((prompt_text, gen_text, known_answer))
    model.train()
    return results


def train(pretrain: bool = False):
    print("=" * 70)
    phase = "PRETRAIN (synthetic arithmetic)" if pretrain else "FINETUNE (GSM8K)"
    print(f"Eidos — Differential Predictive Memory Transformer — {phase}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    vram_gb = 0
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {vram_gb:.1f} GB")

    # Scale config based on available VRAM
    if device.type == 'cuda' and vram_gb >= 16:
        scale = 'large'
        config_model = dict(d_model=768, n_heads=12, n_kv_heads=6, n_layers=8,
                            batch_size=32, accum_steps=1)
    elif device.type == 'cuda' and vram_gb >= 8:
        scale = 'medium'
        config_model = dict(d_model=640, n_heads=10, n_kv_heads=5, n_layers=6,
                            batch_size=16, accum_steps=2)
    else:
        scale = 'small'
        config_model = dict(d_model=512, n_heads=8, n_kv_heads=4, n_layers=4,
                            batch_size=4, accum_steps=2)

    CONFIG = {
        'd_model': config_model['d_model'],
        'n_heads': config_model['n_heads'],
        'n_kv_heads': config_model['n_kv_heads'],
        'n_layers': config_model['n_layers'],
        'seq_len': 96 if pretrain else 192,
        'batch_size': config_model['batch_size'],
        'accum_steps': config_model['accum_steps'],
        'epochs': 30 if pretrain else 15,
        'lr': 5e-5,
        'dropout': 0.2,
        'vocab_mode': 'restricted',
        'amp': True,
        'max_eval': 256,
        'gen_interval': 10 if pretrain else 3,
        'gen_prompts': 3 if pretrain else 8,
    }
    print(f"  Scale: {scale} ({config_model['d_model']}d, {config_model['n_layers']}L, batch={config_model['batch_size']})")

    enc = tiktoken.get_encoding('gpt2')

    base_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    train_texts, eval_texts, train_answers = load_data(base_data_dir, use_synthetic=pretrain)
    combined = list(zip(train_texts, train_answers.tolist()))
    random.shuffle(combined)
    train_texts = [t for t, _ in combined]
    train_answers = torch.tensor([a for _, a in combined], dtype=torch.float32)
    random.shuffle(eval_texts)
    if len(eval_texts) > CONFIG['max_eval']:
        eval_texts = eval_texts[:CONFIG['max_eval']]

    bpe_to_local, local_to_bpe, VOCAB_SIZE = build_restricted_vocab(
        train_texts + eval_texts, enc, min_count=2
    )
    print(f"  Vocab: {VOCAB_SIZE} tokens (restricted from {enc.n_vocab})")

    # Build number magnitude map for value-aware embeddings
    num_values = build_num_values(local_to_bpe, VOCAB_SIZE, enc)

    train_examples = encode_texts(train_texts, enc, CONFIG['seq_len'], bpe_to_local)
    eval_examples = encode_texts(eval_texts, enc, CONFIG['seq_len'], bpe_to_local)
    print(f"  Examples: {len(train_examples)} train, {len(eval_examples)} eval")

    test_prompts = build_test_prompts(eval_texts, train_texts if pretrain else None,
                                       limit=CONFIG['gen_prompts'])
    print(f"  Gen prompts: {len(test_prompts)} dataset-aligned")

    train_dataset = MathDataset(train_examples, train_answers)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_examples, batch_size=CONFIG['batch_size'],
                             shuffle=False, drop_last=False)

    ckpt_name = 'eidos_pretrain_best.pt' if pretrain else 'eidos_v1_best.pt'

    model = EidosV1(
        vocab_size=VOCAB_SIZE, d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'], n_kv_heads=CONFIG['n_kv_heads'],
        n_layers=CONFIG['n_layers'], dropout=CONFIG['dropout'],
        num_values=num_values,
    ).to(device)

    counts = model.count_parameters()
    print(f"  Params: {counts['total']:,} total")
    print(f"    encoder={counts['encoder']:,}, pred_coder={counts['predictive_coder']:,}, "
          f"ltc={counts['ltc_cortex']:,}, mtp={counts['mtp_heads']:,}")

    # torch.compile for graph-level optimization (20-30% faster, Linux-only with Triton)
    compiled = False
    if device.type == 'cuda':
        try:
            import triton
            model = torch.compile(model)
            compiled = True
            print("  torch.compile enabled (inductor+Triton)")
        except ImportError:
            print("  torch.compile skipped (Triton not installed — Windows needs WSL)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    N_EPOCHS = CONFIG['epochs']
    BATCH_SIZE = CONFIG['batch_size']
    ACCUM_STEPS = CONFIG['accum_steps']
    SEQ_LEN = CONFIG['seq_len']

    # Load pretrained weights when finetuning
    if not pretrain:
        pretrain_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'eidos_pretrain_best.pt')
        if os.path.exists(pretrain_path):
            print(f"  Loading pretrained weights from {pretrain_path}")
            pretrain_ckpt = torch.load(pretrain_path, map_location=device)
            pretrain_state = pretrain_ckpt['model_state_dict']
            # Filter out params with size mismatch (different vocab size)
            model_state = model.state_dict()
            compatible = {}
            skipped = []
            for key, val in pretrain_state.items():
                if key in model_state and val.shape == model_state[key].shape:
                    compatible[key] = val
                else:
                    skipped.append(key)
            if skipped:
                print(f"  Skipped {len(skipped)} incompatible params (different vocab): {skipped[:5]}...")
            model.load_state_dict(compatible, strict=False)
            del pretrain_ckpt
        else:
            print(f"  WARNING: No pretrain checkpoint found at {pretrain_path} — training from scratch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                  betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = N_EPOCHS * len(train_loader) // ACCUM_STEPS
    warmup = min(200, total_steps // 10)

    def lr_schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    amp_enabled = CONFIG['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    print(f"\n  Training: {N_EPOCHS} epochs, {total_steps} steps")
    print(f"  LR: {CONFIG['lr']}, warmup={warmup}, accum={ACCUM_STEPS}")
    print(f"  AMP: {'enabled' if amp_enabled else 'disabled'}")
    print("-" * 70)

    best_eval = float('inf')
    global_step = 0

    model.neural_memory.reset()

    # ── 7c: Curiosity curriculum — track per-example prediction error ──
    example_weights = torch.ones(len(train_dataset), device=device)
    use_curiosity = pretrain  # only during pretrain

    if use_curiosity:
        print("  Curiosity curriculum enabled (hard examples upweighted)")

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for i, (inp, lbl, _ans_vals) in enumerate(train_loader):
            inp, lbl = inp.to(device), lbl.to(device)

            with maybe_autocast(device, amp_enabled):
                out = model(inp, labels=lbl)
                loss = out['loss']

            if not torch.isfinite(loss):
                print(f"\n  WARNING: Non-finite loss at epoch {epoch+1}, batch {i} — resetting memory, skipping")
                model.neural_memory.reset()
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / ACCUM_STEPS).backward()

            epoch_loss += loss.item()
            n_batches += 1

            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    avg = epoch_loss / max(n_batches, 1)
                    num_loss_val = out.get('num_loss')
                    num_str = f"num={num_loss_val.item():.5f}" if num_loss_val is not None else ""
                    print(f"    step {global_step:5d}/{total_steps} | "
                          f"epoch {epoch+1}/{N_EPOCHS} | loss={avg:.4f} | "
                          f"{num_str} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e} | "
                          f"{time.time() - t0:.0f}s")

        avg_train = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        @torch.no_grad()
        def evaluate():
            model.eval()
            total = 0.0
            n = 0
            for inp_e, lbl_e in eval_loader:
                inp_e, lbl_e = inp_e.to(device), lbl_e.to(device)
                with maybe_autocast(device, amp_enabled):
                    out_e = model(inp_e, labels=lbl_e)
                total += out_e['loss'].item()
                n += 1
            model.train()
            return total / max(n, 1)

        avg_eval = evaluate()
        ppl_train = math.exp(min(avg_train, 20))
        ppl_eval = math.exp(min(avg_eval, 20))
        gap = avg_eval - avg_train

        # ── Arithmetic accuracy test (every 4 epochs on held-out synthetic) ──
        acc_str = ""
        if pretrain and epoch % 4 == 0:
            @torch.no_grad()
            def eval_accuracy():
                import re
                model.eval()
                correct = 0
                total = 0
                # Match all v2 answer formats: ####, Answer:, Result:, The answer is, etc.
                answer_patterns = [
                    r'####\s*(\d+(?:\.\d+)?)',
                    r'Answer:\s*(\d+(?:\.\d+)?)',
                    r'Result:\s*(\d+(?:\.\d+)?)',
                    r'answer is\s*(\d+(?:\.\d+)?)',
                    r'solution is\s*(\d+(?:\.\d+)?)',
                    r'Therefore.*?result is\s*(\d+(?:\.\d+)?)',
                    r'Final answer:\s*(\d+(?:\.\d+)?)',
                ]
                # Test on held-out synthetic eval examples
                for idx in range(0, min(40, len(eval_examples))):
                    ex = eval_examples[idx]
                    inp_test = ex[0].unsqueeze(0).to(device)
                    full_text = eval_texts[idx] if idx < len(eval_texts) else ""
                    gt = None
                    for pat in answer_patterns:
                        m = re.search(pat, full_text, re.IGNORECASE)
                        if m:
                            gt = float(m.group(1))
                            break
                    if gt is None:
                        continue
                    # Generate
                    gen_out = model.generate(inp_test, max_new_tokens=40,
                                             temperature=0.0, top_k=1,
                                             eos_token_id=bpe_to_local.get(enc.eot_token, 0))
                    gen_ids = gen_out[0].tolist()[inp_test.size(1):]
                    eot_l = bpe_to_local.get(enc.eot_token, 0)
                    if eot_l in gen_ids:
                        gen_ids = gen_ids[:gen_ids.index(eot_l)]
                    gen_bpe = [local_to_bpe.get(g, -1) for g in gen_ids if g > 0]
                    gen_text = enc.decode([g for g in gen_bpe if g >= 0])
                    # Extract predicted number (last number in output)
                    pred_match = re.findall(r'\b(\d+)\b', gen_text)
                    if pred_match:
                        pred = float(pred_match[-1])
                        if abs(pred - gt) < 0.5:
                            correct += 1
                    total += 1
                model.train()
                return correct, total
            acc_correct, acc_total = eval_accuracy()
            acc_str = f"acc={100*acc_correct/max(acc_total,1):.1f}% ({acc_correct}/{acc_total}) | "
        best_marker = ""

        if avg_eval < best_eval:
            best_eval = avg_eval
            best_marker = "  *BEST*"
            save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': CONFIG,
                'epoch': epoch,
                'eval_loss': avg_eval,
                'bpe_to_local': bpe_to_local,
                'local_to_bpe': local_to_bpe,
            }, os.path.join(save_dir, ckpt_name))

        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}  "
              f"{acc_str}"
              f"train={avg_train:.4f} (ppl={ppl_train:.1f})  "
              f"eval={avg_eval:.4f} (ppl={ppl_eval:.1f})  "
              f"gap={gap:+.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{elapsed:.0f}s{best_marker}")

        if epoch % CONFIG['gen_interval'] == 0 or epoch == 0:
            samples = generate_samples(model, enc, device, test_prompts,
                                       bpe_to_local, local_to_bpe, max_tokens=300)
            print(f"  --- Generation (epoch {epoch+1}, greedy) ---")
            for i, (prompt, gen, known) in enumerate(samples):
                if known is not None:
                    print(f"    [{i+1}] TRAINING EXAMPLE:")
                    print(f"    {known}")
                    print()
                print(f"    [{i+1}] PROMPT: {prompt}")
                print(f"    [{i+1}] MODEL:  {gen}")
                print()
            print()

        # ── 8a: Sleep/Replay — replay hard examples through LTC for consolidation ──
        if use_curiosity and epoch > 0 and epoch % 5 == 0 and pretrain:
            print("  --- Sleep/Replay consolidation ---")
            _, hard_indices = torch.topk(example_weights, min(500, len(train_dataset)))
            model.eval()
            replay_loss = 0.0
            replay_batches = 0
            for idx in hard_indices.tolist():
                ex_inp = train_dataset[idx][0].unsqueeze(0).to(device)
                ex_lbl = train_dataset[idx][1].unsqueeze(0).to(device)
                with maybe_autocast(device, amp_enabled):
                    # Full pipeline replay — LTC at max depth consolidates memory into cortex
                    out_replay = model(ex_inp, labels=ex_lbl)
                    replay_loss += out_replay['loss'].item()
                    replay_batches += 1
            if replay_batches > 0:
                print(f"    Replayed {replay_batches} hard examples, avg loss={replay_loss/replay_batches:.4f}")
                model.train()
            print()

    print("\n=== COMPLETE ===")
    print(f"  Best eval loss: {best_eval:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='Pretrain on synthetic arithmetic data')
    parser.add_argument('--generate-synthetic', type=int, default=0, metavar='N',
                        help='Generate N synthetic examples before training')
    args = parser.parse_args()

    if args.generate_synthetic > 0:
        from cortex.synthetic_data import generate_dataset
        data = generate_dataset(args.generate_synthetic)
        out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'train.jsonl')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Generated {len(data)} synthetic examples -> {out_path}")
        if not args.pretrain:
            print("Done. Run with --pretrain to train on synthetic.")
            sys.exit(0)

    train(pretrain=args.pretrain)
