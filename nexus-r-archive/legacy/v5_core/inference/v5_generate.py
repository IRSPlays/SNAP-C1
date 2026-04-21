"""
SNAP-C1 V5: Text Generation / Inference
=========================================
Interactive generation from a trained V5 checkpoint.

Training format (must match):
  [PROMPT]\n---\n[RESPONSE]
  Labels: -100 for prompt, actual tokens for response

So to generate, we feed the prompt + "\n---\n" and sample autoregressively.

Usage:
  # Interactive REPL:
  python v5_core/inference/v5_generate.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt

  # Single prompt:
  python v5_core/inference/v5_generate.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --prompt "Fix the bug in this code: def add(a, b): return a - b"

  # More creative / diverse output:
  python v5_core/inference/v5_generate.py --checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --temperature 0.8 --top_p 0.95
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENC = None

SEP = "\n---\n"


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device):
    """Load V5 model from checkpoint. Returns (model, config, phase)."""
    print(f"Loading {checkpoint_path} ...", end=" ", flush=True)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config')
    if config is None:
        raise ValueError("Checkpoint has no 'config' key. Is this a V5 checkpoint?")

    config['max_think_steps'] = 0  # No THINK loop for inference
    config['dropout'] = 0.0        # Disable dropout for inference

    model = V5ResonanceModel(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    phase = ckpt.get('phase', 'pretrain')
    step = ckpt.get('global_step', '?')
    best_loss = ckpt.get('best_loss', float('inf'))
    print(f"OK  (phase={phase}, step={step}, best_loss={best_loss:.4f})")
    print(f"  d_model={config['d_model']}, n_blocks={config['n_blocks']}, "
          f"max_seq_len={config['max_seq_len']}")
    return model, config, phase


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers
# ──────────────────────────────────────────────────────────────────────────────
def encode(text: str):
    return _ENC.encode(text, disallowed_special=())

def decode(ids) -> str:
    return _ENC.decode(ids)

EOS_TOKEN = _ENC.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0] if _ENC else 50256


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    device=None,
    config: dict = None,
    phase: str = 'instruct',
) -> str:
    """
    Autoregressive generation from a text prompt.
    For instruct models: PROMPT + SEP + generate
    For pretrain models: just continue from prompt
    """
    if _ENC is None:
        raise ImportError("pip install tiktoken")

    device = device or next(model.parameters()).device
    max_seq_len = config.get('max_seq_len', 512) if config else 512

    # Build context: add separator only for instruction-tuned models
    if phase == 'instruct':
        context_text = prompt + SEP
    else:
        context_text = prompt
    context_ids = encode(context_text)

    # Truncate prompt to leave room for generation (keep at least 32 tokens of prompt)
    max_context = max(32, max_seq_len - max_new_tokens)
    if len(context_ids) > max_context:
        context_ids = context_ids[-max_context:]
        print(f"  [prompt truncated to {max_context} tokens]")

    token_ids = list(context_ids)
    generated = []

    # Move sampling to CPU to avoid DirectML scatter_ crash
    cpu = torch.device('cpu')

    for _ in range(max_new_tokens):
        # Prepare input (take last max_seq_len tokens)
        input_ids = token_ids[-max_seq_len:]
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        t = torch.zeros_like(x)  # type_ids = 0

        # Forward pass - get logits for all positions
        result = model.forward_pretrain(x, t)
        logits = result['logits']  # [1, T, vocab]

        # Take logits at last position — move to CPU for sampling
        # (DirectML doesn't support scatter_ needed for top-k/top-p)
        next_logits = logits[0, -1, :].to(cpu).float()  # [vocab]

        # Apply temperature
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy
            next_token = next_logits.argmax().item()
            token_ids.append(next_token)
            generated.append(next_token)
            if next_token == EOS_TOKEN:
                break
            continue

        # Convert to probabilities
        probs = F.softmax(next_logits, dim=-1)

        # Top-k filtering
        if top_k > 0:
            k = min(top_k, probs.size(-1))
            top_k_probs, top_k_ids = torch.topk(probs, k)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, top_k_ids, True)
            probs[~mask] = 0.0

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative prob above top_p
            remove_mask = cumsum - sorted_probs > top_p
            sorted_probs[remove_mask] = 0.0
            # Unsort back
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, sorted_ids, sorted_probs)

        # Renormalize
        total = probs.sum()
        if total <= 0:
            probs = F.softmax(next_logits, dim=-1)  # fallback
        else:
            probs = probs / total

        # Sample
        next_token = torch.multinomial(probs, num_samples=1).item()
        token_ids.append(next_token)
        generated.append(next_token)

        # Stop at EOS
        if next_token == EOS_TOKEN:
            break

    return decode(generated)


# ──────────────────────────────────────────────────────────────────────────────
# Built-in test prompts
# ──────────────────────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    # Bug fixes (V4 training format)
    "Fix the bug in this Python function:\n\n```python\ndef divide(a, b):\n    return a / b\n```",
    "Fix the bug in this Python function:\n\n```python\ndef get_first(items):\n    return items[0]\n```",
    "Fix missing None check before attribute access:\n\n```python\ndef get_name(user):\n    return user.name.strip()\n```",
    # Code generation
    "Write a Python function that reverses a string.",
    "Write a Python function that checks if a number is prime.",
]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Text Generation")
    parser.add_argument('--checkpoint', required=True,
                        help='Path to V5 checkpoint (.pt)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to generate from (omit for REPL mode)')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (0=greedy, 1=creative)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling cutoff')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--test', action='store_true',
                        help='Run all built-in test prompts')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (slower but avoids DML issues)')
    args = parser.parse_args()

    # Device
    if args.cpu:
        device = torch.device('cpu')
        print("Device: cpu (forced)")
    else:
        device = get_device()
        print(f"Device: {device}")

    # Load model
    model, config, phase = load_model(args.checkpoint, device)
    print(f"  Generation mode: {phase}")

    gen_kwargs = dict(
        model=model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=device,
        config=config,
        phase=phase,
    )

    # === Single prompt mode ===
    if args.prompt:
        print(f"\n{'─' * 60}")
        print(f"PROMPT:\n{args.prompt}")
        print(f"{'─' * 60}")
        print("RESPONSE:")
        response = generate(prompt=args.prompt, **gen_kwargs)
        print(response)
        print(f"{'─' * 60}")
        return

    # === Test mode ===
    if args.test:
        print(f"\n{'═' * 60}")
        print("RUNNING BUILT-IN TEST PROMPTS")
        print(f"{'═' * 60}")
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n[Test {i+1}/{len(TEST_PROMPTS)}]")
            print(f"PROMPT: {prompt[:80]}{'...' if len(prompt)>80 else ''}")
            print("RESPONSE:")
            response = generate(prompt=prompt, **gen_kwargs)
            print(response)
            print(f"{'─' * 60}")
        return

    # === Interactive REPL mode ===
    print(f"\n{'═' * 60}")
    print("SNAP-C1 V5 — Interactive Mode")
    print(f"{'═' * 60}")
    print("Type your prompt. Special commands:")
    print("  /test    — run built-in test prompts")
    print("  /quit    — exit")
    print(f"{'─' * 60}\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt == '/quit':
            print("Goodbye!")
            break
        if prompt == '/test':
            for i, tp in enumerate(TEST_PROMPTS):
                print(f"\n[Test {i+1}] {tp[:60]}...")
                print("SNAP:", generate(prompt=tp, **gen_kwargs))
            continue

        print("\nSNAP:", end=" ", flush=True)
        response = generate(prompt=prompt, **gen_kwargs)
        print(response)
        print()


if __name__ == "__main__":
    main()
