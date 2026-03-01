"""
SNAP-C1 V5: Auto-DPO Self-Learning Engine (Pillar C)
======================================================
Online DPO from execution results — NO human labels.

How it works:
  1. terminal_loop.py / self_play_coder.py produce DPO pairs:
     (prompt, chosen_code rejected_code, chosen_reward, rejected_reward)
  2. This module loads them, trains LoRA adapters on the frozen base model.
  3. KL divergence guard prevents mode collapse.,
  4. Regression check against held-out tasks detects silent degradation.

LoRA implementation:
  We inject rank-16 LoRA matrices into every Linear layer in the
  resonance blocks. The base model weights stay frozen. Only LoRA
  params (~0.3% of total) are updated. This means:
  - Training is fast (~1 min per 64 pairs on H200)
  - Base capabilities are preserved
  - Multiple LoRA adapters can be saved/loaded (per-user profiles)

The DPO loss (from Rafailov et al. 2023):
  L = -log σ(β * (log π(chosen) - log π_ref(chosen)
                  - log π(rejected) + log π_ref(rejected)))

Where π is the LoRA-adapted model and π_ref is the frozen base.

Usage:
  # Train from DPO buffer:
  python auto_dpo_v5.py --base_checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --dpo_buffer v5_core/data/dpo_buffer_v5.jsonl --steps 100

  # Daemon mode (watches buffer, trains when 64+ new pairs arrive):
  python auto_dpo_v5.py --base_checkpoint v5_core/checkpoints/v5_instruct_best.pt \\
      --dpo_buffer v5_core/data/dpo_buffer_v5.jsonl --daemon
"""

import os
import sys
import gc
import json
import copy
import time
import math
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v5_core.utils.dml_ops import get_device
from v5_core.architecture.v5_assembly import V5ResonanceModel

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENC = None


# ──────────────────────────────────────────────────────────────────────────────
# LoRA Injection
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.
    Freezes the original weight and adds trainable A, B matrices.

    out = W_frozen @ x + (B @ A) @ x * scaling
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        # LoRA matrices (A: down-project, B: up-project)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen forward
        base = self.original(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + lora_delta * self.scaling

    def merge_and_unload(self) -> nn.Linear:
        """Merge LoRA weights back into the original Linear for inference."""
        with torch.no_grad():
            self.original.weight.add_(
                (self.lora_B @ self.lora_A) * self.scaling
            )
        return self.original


def inject_lora(model: V5ResonanceModel, rank: int = 16, alpha: float = 32.0,
                target_modules: list = None) -> dict:
    """
    Inject LoRA adapters into the model's resonance blocks.

    Args:
        model: The V5 model (base weights will be frozen)
        rank: LoRA rank (16 = ~0.3% of params for 7B model)
        alpha: LoRA scaling factor
        target_modules: Which submodule names to target (default: all Linear in resonance)

    Returns:
        dict mapping parameter names to LoRA modules (for saving/loading)
    """
    target_modules = target_modules or [
        'local_attn.in_proj', 'local_attn.out_proj',
        'global_attn', 'gate_proj', 'ffn',
        'lm_down', 'lm_up',
    ]

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    lora_modules = {}
    replaced = 0

    # Inject LoRA into resonance blocks
    for block_idx, block in enumerate(model.resonance.blocks):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this Linear is in a target module path
                should_target = any(t in name for t in target_modules) or not target_modules
                if should_target and module.in_features >= 64:  # Skip tiny Linears
                    lora = LoRALinear(module, rank=rank, alpha=alpha)
                    # Replace in the parent
                    parts = name.rsplit('.', 1)
                    if len(parts) == 2:
                        parent_name, child_name = parts
                        parent = dict(block.named_modules())[parent_name]
                        setattr(parent, child_name, lora)
                    else:
                        setattr(block, name, lora)
                    lora_modules[f"block_{block_idx}.{name}"] = lora
                    replaced += 1

    # Also inject into LM head
    for name in ['lm_down', 'lm_up']:
        module = getattr(model, name, None)
        if module is not None and isinstance(module, nn.Linear):
            lora = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(model, name, lora)
            lora_modules[f"lm_head.{name}"] = lora
            replaced += 1

    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Injected {replaced} adapters, rank={rank}, alpha={alpha}")
    print(f"[LoRA] Trainable: {lora_params:,} / {total_params:,} "
          f"({lora_params / total_params * 100:.2f}%)")

    return lora_modules


def save_lora(lora_modules: dict, path: str, metadata: dict = None):
    """Save only LoRA weights (tiny file, ~50MB for 7B model)."""
    state = {}
    for name, module in lora_modules.items():
        state[f"{name}.lora_A"] = module.lora_A.data.cpu()
        state[f"{name}.lora_B"] = module.lora_B.data.cpu()
    payload = {'lora_state': state, 'metadata': metadata or {}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[LoRA] Saved {len(state)} tensors to {path} ({size_mb:.1f} MB)")


def load_lora(lora_modules: dict, path: str) -> dict:
    """Load LoRA weights from a saved file."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['lora_state']
    loaded = 0
    for name, module in lora_modules.items():
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key in state and b_key in state:
            module.lora_A.data.copy_(state[a_key])
            module.lora_B.data.copy_(state[b_key])
            loaded += 1
    print(f"[LoRA] Loaded {loaded}/{len(lora_modules)} adapters from {path}")
    return payload.get('metadata', {})


# ──────────────────────────────────────────────────────────────────────────────
# DPO Training
# ──────────────────────────────────────────────────────────────────────────────

def tokenize(text: str, max_len: int = 512) -> list[int]:
    """Tokenize text using tiktoken cl100k_base."""
    if _ENC is None:
        raise ImportError("pip install tiktoken")
    tokens = _ENC.encode(text, disallowed_special=())
    return tokens[:max_len]


def compute_logprobs(model: V5ResonanceModel, token_ids: torch.Tensor,
                     type_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token log probabilities for a sequence.
    Returns sum of log probs over non-masked positions.

    Labels follow HuggingFace convention: labels[i] = tokens[i].
    Shift is applied here: logits[i] predicts token[i+1], so we compare
    logits[:-1] with labels[1:].
    """
    result = model.forward_pretrain(token_ids, type_ids, labels=None)
    logits = result['logits']  # [B, S, V]

    # Standard causal LM shift: logits[i] predicts token[i+1]
    S = logits.shape[1]
    shift_logits = logits[:, :-1, :]  # [B, S-1, V]
    shift_labels = labels[:, 1:S]     # [B, S-1] (align with logits length)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)  # [B, S-1, V]

    # Clamp indices to avoid gather at -100 (masked positions)
    gather_ids = shift_labels.clamp(min=0)
    gathered = log_probs.gather(2, gather_ids.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

    # Mask: only count non-padding positions
    mask = (shift_labels != -100).float()
    return (gathered * mask).sum(dim=-1)  # [B]


def dpo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor,
             ref_rejected_logps: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """
    DPO loss (Rafailov et al. 2023).

    L = -log σ(β * (log π(y_w|x) - log π_ref(y_w|x)
                    - log π(y_l|x) + log π_ref(y_l|x)))

    Returns (loss, metrics_dict)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards  # [B]
    loss = -F.logsigmoid(logits).mean()

    # Metrics
    with torch.no_grad():
        reward_acc = (logits > 0).float().mean().item()
        chosen_reward_mean = chosen_rewards.mean().item()
        rejected_reward_mean = rejected_rewards.mean().item()
        margin = (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        'loss': loss.item(),
        'reward_accuracy': reward_acc,
        'chosen_reward': chosen_reward_mean,
        'rejected_reward': rejected_reward_mean,
        'margin': margin,
    }
    return loss, metrics


def compute_kl_divergence(model: V5ResonanceModel,
                          ref_model: V5ResonanceModel,
                          token_ids: torch.Tensor,
                          type_ids: torch.Tensor) -> float:
    """
    Compute KL(π_policy || π_ref) on a sample to monitor divergence.
    If KL > threshold, we should skip the gradient step.
    """
    with torch.no_grad():
        result_policy = model.forward_pretrain(token_ids, type_ids, labels=None)
        result_ref = ref_model.forward_pretrain(token_ids, type_ids, labels=None)

        S = min(result_policy['logits'].shape[1], result_ref['logits'].shape[1])

        p_logprobs = F.log_softmax(result_policy['logits'][:, :S, :].float(), dim=-1)
        q_logprobs = F.log_softmax(result_ref['logits'][:, :S, :].float(), dim=-1)

        # KL(policy || ref) = sum policy * (log_policy - log_ref)
        kl = F.kl_div(q_logprobs, p_logprobs, reduction='batchmean', log_target=True)
        return kl.item()


# ──────────────────────────────────────────────────────────────────────────────
# DPO Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DPODataset:
    """Loads DPO pairs from a JSONL buffer file."""

    def __init__(self, buffer_path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.pairs = []

        if not os.path.exists(buffer_path):
            print(f"[DPODataset] Buffer not found: {buffer_path}")
            return

        with open(buffer_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get('chosen', '').strip() and entry.get('rejected', '').strip():
                            self.pairs.append(entry)
                    except json.JSONDecodeError:
                        print(f"[DPODataset] Skipping malformed line: {line[:80]}...")

        print(f"[DPODataset] Loaded {len(self.pairs)} pairs from {buffer_path}")

    def __len__(self):
        return len(self.pairs)

    def get_batch(self, indices: list, device: torch.device):
        """
        Prepare a batch of (chosen, rejected) token sequences.

        Each pair is formatted as:
          [PROMPT]\n---\n[CODE]

        Returns dict with:
          chosen_ids, chosen_labels, rejected_ids, rejected_labels, type_ids
        """
        sep = "\n---\n"
        chosen_ids_list = []
        rejected_ids_list = []
        chosen_labels_list = []
        rejected_labels_list = []

        for idx in indices:
            pair = self.pairs[idx]
            prompt = pair['prompt']
            chosen = pair['chosen']
            rejected = pair['rejected']

            # Tokenize
            prompt_tokens = tokenize(prompt)
            sep_tokens = tokenize(sep)
            chosen_tokens = tokenize(chosen)
            rejected_tokens = tokenize(rejected)

            # Build sequences
            chosen_seq = prompt_tokens + sep_tokens + chosen_tokens
            rejected_seq = prompt_tokens + sep_tokens + rejected_tokens

            # Labels: -100 for prompt, actual tokens for response
            prompt_len = len(prompt_tokens) + len(sep_tokens)

            chosen_labels = [-100] * min(prompt_len, len(chosen_seq))
            chosen_labels += chosen_seq[prompt_len:]

            rejected_labels = [-100] * min(prompt_len, len(rejected_seq))
            rejected_labels += rejected_seq[prompt_len:]

            # Truncate / pad to seq_len
            for seq, lab, seq_list, lab_list in [
                (chosen_seq, chosen_labels, chosen_ids_list, chosen_labels_list),
                (rejected_seq, rejected_labels, rejected_ids_list, rejected_labels_list),
            ]:
                seq = seq[:self.seq_len]
                lab = lab[:self.seq_len]
                pad = self.seq_len - len(seq)
                seq += [0] * pad
                lab += [-100] * pad
                seq_list.append(seq)
                lab_list.append(lab)

        # Stack into tensors
        chosen_ids = torch.tensor(chosen_ids_list, dtype=torch.long, device=device)
        rejected_ids = torch.tensor(rejected_ids_list, dtype=torch.long, device=device)
        chosen_labels = torch.tensor(chosen_labels_list, dtype=torch.long, device=device)
        rejected_labels = torch.tensor(rejected_labels_list, dtype=torch.long, device=device)
        type_ids = torch.zeros_like(chosen_ids)

        return {
            'chosen_ids': chosen_ids,
            'chosen_labels': chosen_labels,
            'rejected_ids': rejected_ids,
            'rejected_labels': rejected_labels,
            'type_ids': type_ids,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Held-out regression tests
# ──────────────────────────────────────────────────────────────────────────────

HELD_OUT_TASKS = [
    {
        'prompt': 'Write a function that returns the factorial of n.',
        'test': 'from module import factorial\nassert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1\nprint("PASS")',
        'func_name': 'factorial',
    },
    {
        'prompt': 'Write a function that flattens a nested list.',
        'test': 'from module import flatten\nassert flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nprint("PASS")',
        'func_name': 'flatten',
    },
    {
        'prompt': 'Write a function that checks if a string is a palindrome.',
        'test': 'from module import is_palindrome\nassert is_palindrome("racecar") == True\nassert is_palindrome("hello") == False\nassert is_palindrome("") == True\nprint("PASS")',
        'func_name': 'is_palindrome',
    },
]


def run_regression_check(model: V5ResonanceModel, config: dict,
                         device: torch.device) -> dict:
    """
    Run held-out tasks and return pass/fail metrics.
    This detects if DPO training is degrading base capabilities.
    """
    # Lazy import to avoid circular dependency
    from v5_core.inference.v5_generate import generate
    from v5_core.training.terminal_loop import TerminalRunner

    runner = TerminalRunner(timeout_sec=10)
    results = {}

    for task in HELD_OUT_TASKS:
        try:
            # Generate code
            code = generate(
                model=model, prompt=task['prompt'],
                max_new_tokens=256, temperature=0.2, top_p=0.9,
                device=device, config=config, phase='instruct',
            )

            # Execute generated code + test
            exec_result = runner.run_pytest(
                test_code=task['test'],
                source_code=code,
            )
            results[task['func_name']] = {
                'passed': exec_result.is_success() and 'PASS' in exec_result.stdout,
                'reward': exec_result.reward,
            }
        except Exception as e:
            results[task['func_name']] = {'passed': False, 'error': str(e)}

    passed = sum(1 for r in results.values() if r.get('passed', False))
    total = len(results)
    print(f"[Regression] {passed}/{total} held-out tasks passed")
    return {'results': results, 'pass_rate': passed / total if total > 0 else 0}


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_dpo(args):
    """Main DPO training loop."""
    print("=" * 60)
    print("SNAP-C1 V5 AUTO-DPO SELF-LEARNING")
    print("=" * 60)

    device = get_device()
    is_cuda = device.type == 'cuda'
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    ref_on_cpu = getattr(args, 'ref_on_cpu', False)
    print(f"Device: {device}, Precision: {'bf16' if use_bf16 else 'fp32'}")
    if ref_on_cpu:
        print(f"[Memory] Reference model on CPU (saves ~50% GPU VRAM)")

    # ── Load base model ──────────────────────────────────────────────────
    print(f"\nLoading base checkpoint: {args.base_checkpoint}")
    ckpt = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']
    config['max_think_steps'] = 0
    config['dropout'] = 0.0  # No dropout for DPO

    # Policy model (will have LoRA injected)
    policy = V5ResonanceModel(**config)
    policy.load_state_dict(ckpt['model_state_dict'], strict=True)
    if use_bf16:
        policy = policy.to(dtype=torch.bfloat16, device=device)
    else:
        policy = policy.to(device)

    # Enable gradient checkpointing for low-VRAM GPUs
    if getattr(args, 'grad_checkpoint', False):
        if hasattr(policy, 'gradient_checkpointing_enable'):
            policy.gradient_checkpointing_enable()
        else:
            # Manual: wrap resonance blocks
            from torch.utils.checkpoint import checkpoint as ckpt_fn
            policy._use_gradient_checkpointing = True
        print(f"[Memory] Gradient checkpointing enabled")

    # Reference model (frozen, no LoRA — used for KL computation)
    ref = V5ResonanceModel(**config)
    ref.load_state_dict(ckpt['model_state_dict'], strict=True)
    if ref_on_cpu:
        # Keep ref on CPU to save ~50% GPU VRAM
        # Slower KL checks, but frees huge VRAM on 8GB GPUs
        ref = ref.to(dtype=torch.float32, device=torch.device('cpu'))
    elif use_bf16:
        ref = ref.to(dtype=torch.bfloat16, device=device)
    else:
        ref = ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # ── Inject LoRA ──────────────────────────────────────────────────────
    lora_modules = inject_lora(policy, rank=args.lora_rank, alpha=args.lora_alpha)

    # Load existing LoRA if resuming
    if args.resume_lora and os.path.exists(args.resume_lora):
        load_lora(lora_modules, args.resume_lora)

    # ── Load DPO data ────────────────────────────────────────────────────
    dataset = DPODataset(args.dpo_buffer, seq_len=args.seq_len)
    if len(dataset) == 0:
        print("ERROR: No DPO pairs found. Run self_play_coder.py or terminal_loop.py first.")
        return

    # ── Optimizer ────────────────────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"Optimizer: AdamW, lr={args.lr}, {len(trainable_params)} param groups")

    # ── Training ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"DPO Training: {args.steps} steps, batch_size={args.batch_size}")
    print(f"Beta: {args.beta}, KL threshold: {args.kl_threshold}")
    print(f"{'=' * 60}\n")

    policy.train()
    total_pairs = len(dataset)
    step = 0
    skipped = 0
    consecutive_skips = 0
    max_consecutive_skips = args.steps * 3  # Safety: abort if stuck
    history = []

    import random
    indices = list(range(total_pairs))

    while step < args.steps:
        # Sample a random batch
        random.shuffle(indices)
        batch_idx = indices[:args.batch_size]
        batch = dataset.get_batch(batch_idx, device)

        # ── Compute log probs ───────────────────────────────────────────
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Policy log probs
                policy_chosen_lp = compute_logprobs(
                    policy, batch['chosen_ids'], batch['type_ids'], batch['chosen_labels']
                )
                policy_rejected_lp = compute_logprobs(
                    policy, batch['rejected_ids'], batch['type_ids'], batch['rejected_labels']
                )
        else:
            policy_chosen_lp = compute_logprobs(
                policy, batch['chosen_ids'], batch['type_ids'], batch['chosen_labels']
            )
            policy_rejected_lp = compute_logprobs(
                policy, batch['rejected_ids'], batch['type_ids'], batch['rejected_labels']
            )

        # Reference log probs (no grad)
        with torch.no_grad():
            if ref_on_cpu:
                # Move batch to CPU for ref, results back to GPU
                cpu_chosen = batch['chosen_ids'].cpu()
                cpu_rejected = batch['rejected_ids'].cpu()
                cpu_type = batch['type_ids'].cpu()
                cpu_chosen_labels = batch['chosen_labels'].cpu()
                cpu_rejected_labels = batch['rejected_labels'].cpu()
                ref_chosen_lp = compute_logprobs(
                    ref, cpu_chosen, cpu_type, cpu_chosen_labels
                ).to(device)
                ref_rejected_lp = compute_logprobs(
                    ref, cpu_rejected, cpu_type, cpu_rejected_labels
                ).to(device)
                del cpu_chosen, cpu_rejected, cpu_type, cpu_chosen_labels, cpu_rejected_labels
            elif use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ref_chosen_lp = compute_logprobs(
                        ref, batch['chosen_ids'], batch['type_ids'], batch['chosen_labels']
                    )
                    ref_rejected_lp = compute_logprobs(
                        ref, batch['rejected_ids'], batch['type_ids'], batch['rejected_labels']
                    )
            else:
                ref_chosen_lp = compute_logprobs(
                    ref, batch['chosen_ids'], batch['type_ids'], batch['chosen_labels']
                )
                ref_rejected_lp = compute_logprobs(
                    ref, batch['rejected_ids'], batch['type_ids'], batch['rejected_labels']
                )

        # ── DPO loss ────────────────────────────────────────────────────
        loss, metrics = dpo_loss(
            policy_chosen_lp, policy_rejected_lp,
            ref_chosen_lp, ref_rejected_lp,
            beta=args.beta,
        )

        # ── KL divergence check ─────────────────────────────────────────
        if step % args.kl_check_every == 0:
            if ref_on_cpu:
                # Move policy sample to CPU for KL check
                kl_ids = batch['chosen_ids'][:1].cpu()
                kl_type = batch['type_ids'][:1].cpu()
                # Temporarily move policy to CPU for KL (or use cached logits)
                with torch.no_grad():
                    pol_result = policy.forward_pretrain(
                        batch['chosen_ids'][:1], batch['type_ids'][:1], labels=None
                    )
                    ref_result = ref.forward_pretrain(kl_ids, kl_type, labels=None)
                    S = min(pol_result['logits'].shape[1], ref_result['logits'].shape[1])
                    p = F.log_softmax(pol_result['logits'][:, :S, :].float().cpu(), dim=-1)
                    q = F.log_softmax(ref_result['logits'][:, :S, :].float(), dim=-1)
                    kl = F.kl_div(q, p, reduction='batchmean', log_target=True).item()
                del kl_ids, kl_type
            else:
                kl = compute_kl_divergence(
                    policy, ref, batch['chosen_ids'][:1], batch['type_ids'][:1]
                )
            metrics['kl'] = kl
            if kl > args.kl_threshold:
                print(f"  [Step {step}] KL={kl:.4f} > {args.kl_threshold} — SKIPPING gradient")
                skipped += 1
                consecutive_skips += 1
                if consecutive_skips > max_consecutive_skips:
                    print(f"  [ABORT] {consecutive_skips} consecutive KL violations. Stopping.")
                    break
                del loss
                gc.collect()
                if is_cuda:
                    torch.cuda.empty_cache()
                continue
            else:
                consecutive_skips = 0  # Reset on successful step

        # ── Backward + update ───────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        step += 1
        history.append(metrics)

        # ── Logging ─────────────────────────────────────────────────────
        if step % 5 == 0 or step == 1:
            kl_str = f" kl={metrics.get('kl', '?'):.4f}" if 'kl' in metrics else ""
            print(f"  Step {step}/{args.steps} | loss={metrics['loss']:.4f} "
                  f"| acc={metrics['reward_accuracy']:.2f} "
                  f"| margin={metrics['margin']:.4f}{kl_str}")

        # ── Save ─────────────────────────────────────────────────────────
        if step % args.save_every == 0:
            save_path = f"v5_core/checkpoints/v5_lora_step{step}.pt"
            save_lora(lora_modules, save_path, metadata={
                'step': step, 'history': history[-10:],
                'skipped': skipped, 'config': config,
            })

        # ── Memory cleanup ──────────────────────────────────────────────
        del loss, batch
        if is_cuda and step % 10 == 0:
            torch.cuda.empty_cache()
        gc.collect()

    # ── Final save ───────────────────────────────────────────────────────
    save_path = args.output or "v5_core/checkpoints/v5_lora_latest.pt"
    save_lora(lora_modules, save_path, metadata={
        'total_steps': step, 'skipped': skipped,
        'final_metrics': history[-1] if history else {},
        'config': config,
    })

    # ── Regression check ─────────────────────────────────────────────────
    if args.regression_check:
        print("\n[Regression check] Running held-out tasks...")
        policy.eval()
        regression = run_regression_check(policy, config, device)
        if regression['pass_rate'] < 0.5:
            print("  WARNING: Regression detected! LoRA may have degraded base skills.")
            print("  Consider reverting to previous LoRA checkpoint.")

    print(f"\n{'=' * 60}")
    print(f"AUTO-DPO COMPLETE")
    print(f"  Steps: {step}, Skipped: {skipped}")
    print(f"  LoRA saved to: {save_path}")
    if history:
        print(f"  Final loss: {history[-1]['loss']:.4f}")
        print(f"  Final accuracy: {history[-1]['reward_accuracy']:.2f}")
    print(f"{'=' * 60}")


def daemon_mode(args):
    """
    Watch the DPO buffer file. When 64+ new pairs arrive,
    run a training step automatically. Runs forever.
    """
    print(f"[Daemon] Watching {args.dpo_buffer} for new pairs...")
    print(f"[Daemon] Will train every {args.daemon_batch} new pairs")
    last_count = 0

    while True:
        try:
            if os.path.exists(args.dpo_buffer):
                with open(args.dpo_buffer, 'r') as f:
                    count = sum(1 for _ in f)
            else:
                count = 0

            new_pairs = count - last_count
            if new_pairs >= args.daemon_batch:
                print(f"\n[Daemon] {new_pairs} new pairs detected. Training...")
                args.steps = min(new_pairs // args.batch_size, 50)  # Cap at 50 steps
                train_dpo(args)
                last_count = count
                # Carry forward LoRA state for next round
                args.resume_lora = args.output or "v5_core/checkpoints/v5_lora_latest.pt"
            else:
                time.sleep(args.daemon_interval)

        except KeyboardInterrupt:
            print("\n[Daemon] Stopped.")
            break
        except Exception as e:
            print(f"[Daemon] Error: {e}")
            time.sleep(60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SNAP-C1 V5 Auto-DPO Self-Learning")

    parser.add_argument('--base_checkpoint', required=True,
                        help='Path to base (instruct-tuned) checkpoint')
    parser.add_argument('--dpo_buffer', default='v5_core/data/dpo_buffer_v5.jsonl',
                        help='Path to DPO pairs JSONL file')
    parser.add_argument('--output', default=None,
                        help='Output path for LoRA checkpoint')
    parser.add_argument('--resume_lora', default=None,
                        help='Resume from existing LoRA checkpoint')

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                        help='LoRA alpha scaling (default: 32.0)')

    # Training
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of DPO training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (pairs per step)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for LoRA params')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta (reward scaling)')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Max sequence length for DPO pairs')

    # Safety
    parser.add_argument('--kl_threshold', type=float, default=0.3,
                        help='Max KL divergence before skipping step')
    parser.add_argument('--kl_check_every', type=int, default=10,
                        help='Check KL every N steps')
    parser.add_argument('--regression_check', action='store_true',
                        help='Run held-out regression tests after training')

    # Memory optimization (for 8GB GPUs like RX 7600)
    parser.add_argument('--ref_on_cpu', action='store_true',
                        help='Keep reference model on CPU (saves ~50%% GPU VRAM)')
    parser.add_argument('--grad_checkpoint', action='store_true',
                        help='Enable gradient checkpointing (saves VRAM, slower)')

    # Checkpoints
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save LoRA every N steps')

    # Daemon mode
    parser.add_argument('--daemon', action='store_true',
                        help='Run in daemon mode (watch buffer, auto-train)')
    parser.add_argument('--daemon_batch', type=int, default=64,
                        help='Min new pairs before daemon triggers training')
    parser.add_argument('--daemon_interval', type=float, default=60,
                        help='Seconds between buffer checks in daemon mode')

    args = parser.parse_args()

    if args.daemon:
        daemon_mode(args)
    else:
        train_dpo(args)


if __name__ == "__main__":
    main()
