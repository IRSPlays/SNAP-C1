"""
Nexus-R V1 Smoke Test
======================
Validates: import, forward pass, backward pass, gradient flow,
           parameter counts, recursion info, and generation.

Run: python -m nexus_v1.tests.test_smoke
"""

import sys
import os

# Add parent dirs to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F


def test_tiny_forward():
    """Test forward pass on tiny config."""
    from nexus_v1.architecture import build_nexus_tiny

    model = build_nexus_tiny()
    print(f"[1/6] Model built: {type(model).__name__}")

    # Count params
    info = model.count_params()
    total = info['total']
    print(f"  Total params: {total:,}")
    for k, v in info.items():
        if k != 'total':
            print(f"  {k}: {v:,}")

    assert total > 0, "Model has no parameters"
    assert total < 50_000_000, f"Tiny model too big: {total:,}"
    print("  PASS\n")
    return model


def test_forward_backward(model):
    """Test forward + backward on random tokens."""
    B, T = 2, 64
    vocab = model.cfg.vocab_size

    input_ids = torch.randint(0, vocab, (B, T))
    labels = torch.randint(0, vocab, (B, T))

    # Forward
    out = model(input_ids, labels=labels)
    logits = out['logits']
    loss = out['loss']
    info = out['recursion_info']

    print(f"[2/6] Forward pass:")
    print(f"  logits shape: {logits.shape}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  recursion steps: {info['total_recursive_steps']}")
    print(f"  h_cycles used: {info['h_cycles_used']}")
    print(f"  converged early: {info['converged_early']}")

    assert logits.shape == (B, T, vocab), f"Bad logits shape: {logits.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"

    # Backward
    loss.backward()
    print("  Backward pass: OK")

    # Check gradients exist
    grad_count = 0
    none_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_count += 1
            else:
                none_count += 1

    print(f"  Params with gradients: {grad_count}")
    print(f"  Params with zero/None grad: {none_count}")
    assert grad_count > 0, "No gradients flowing!"
    print("  PASS\n")


def test_gradient_flow(model):
    """Verify gradients flow through key components."""
    model.zero_grad()
    B, T = 1, 32
    input_ids = torch.randint(0, model.cfg.vocab_size, (B, T))
    labels = torch.randint(0, model.cfg.vocab_size, (B, T))

    out = model(input_ids, labels=labels)
    out['loss'].backward()

    # Check specific components have gradients
    # NOTE: thought_init may not get gradients when H_cycles > 1 because
    # it only participates in no_grad H-steps (by TRM design). This is expected.
    checks = {
        'embed': model.embed.weight.grad,
        'anchor_encoder': None,
        'final_norm': model.final_norm.weight.grad,
    }

    known_no_grad = set()

    # Find a grad from anchor encoder
    for name, p in model.anchor_encoder.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            checks['anchor_encoder'] = p.grad
            break

    print("[3/6] Gradient flow check:")
    all_ok = True
    for name, grad in checks.items():
        if grad is not None and grad.abs().sum() > 0:
            print(f"  {name}: OK (grad norm={grad.norm():.6f})")
        else:
            print(f"  {name}: MISSING GRAD")
            all_ok = False

    # Reasoner layer grads (recursive blocks)
    for i, layer in enumerate(model.reasoner.layers):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.parameters()
        )
        status = "OK" if has_grad else "MISSING"
        print(f"  reasoner.layer[{i}]: {status}")
        if not has_grad:
            all_ok = False

    assert all_ok, "Some components missing gradients!"
    print("  PASS\n")


def test_recursion_info():
    """Test that recursion metadata is reasonable."""
    from nexus_v1.architecture import build_nexus_tiny

    model = build_nexus_tiny()
    B, T = 1, 16
    input_ids = torch.randint(0, model.cfg.vocab_size, (B, T))

    with torch.no_grad():
        out = model(input_ids)

    info = out['recursion_info']
    print("[4/6] Recursion info:")
    print(f"  total steps: {info['total_recursive_steps']}")
    print(f"  h_cycles: {info['h_cycles_used']}")
    print(f"  halt sims: {info['halt_similarities']}")
    print(f"  converged: {info['converged_early']}")

    assert info['total_recursive_steps'] > 0
    assert info['h_cycles_used'] > 0
    print("  PASS\n")


def test_generation():
    """Test autoregressive generation."""
    from nexus_v1.architecture import build_nexus_tiny

    model = build_nexus_tiny()
    model.eval()

    prompt = torch.randint(0, model.cfg.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=16)

    print("[5/6] Generation:")
    print(f"  prompt length: {prompt.shape[1]}")
    print(f"  generated length: {generated.shape[1]}")
    print(f"  new tokens: {generated.shape[1] - prompt.shape[1]}")
    assert generated.shape[1] == prompt.shape[1] + 16
    print("  PASS\n")


def test_loss_decreases():
    """Train for a few steps and verify loss decreases (no memorization, just learning signal)."""
    from nexus_v1.architecture import build_nexus_tiny

    model = build_nexus_tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, T = 4, 32
    input_ids = torch.randint(0, model.cfg.vocab_size, (B, T))
    labels = input_ids.clone()

    losses = []
    for step in range(10):
        out = model(input_ids, labels=labels)
        loss = out['loss']
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("[6/6] Loss decrease test (10 steps):")
    print(f"  Step 0 loss: {losses[0]:.4f}")
    print(f"  Step 9 loss: {losses[-1]:.4f}")
    print(f"  Decreased: {losses[-1] < losses[0]}")

    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("  PASS\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Nexus-R V1 Smoke Test")
    print("=" * 60 + "\n")

    model = test_tiny_forward()
    test_forward_backward(model)
    test_gradient_flow(model)
    test_recursion_info()
    test_generation()
    test_loss_decreases()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
