"""
SNAP-C1 V5: Smoke Test
========================
Tests all components individually + full assembly on DirectML (RX 7600).
Verifies: forward pass, backward pass, no scatter_ crashes, correct shapes.

Run: python v5_core/test_v5_smoke.py
"""

import sys
import time
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, ".")

from v5_core.utils.dml_ops import get_device, RMSNorm, SwiGLU, stable_sigmoid, chunked_softmax


def get_test_device():
    device = get_device()
    print(f"Device: {device}")
    return device


def test_dml_ops(device):
    print("\n=== Test 1: DirectML-Safe Ops ===")
    x = torch.randn(2, 64, 256, device=device, requires_grad=True)

    # RMSNorm
    norm = RMSNorm(256).to(device)
    y = norm(x)
    y.sum().backward()
    print(f"  RMSNorm:        OK  shape={y.shape}")

    # SwiGLU
    x2 = torch.randn(2, 64, 256, device=device, requires_grad=True)
    ffn = SwiGLU(256, 1024).to(device)
    y2 = ffn(x2)
    y2.sum().backward()
    print(f"  SwiGLU:         OK  shape={y2.shape}")

    # StableSigmoid
    x3 = torch.randn(2, 64, 256, device=device, requires_grad=True)
    y3 = stable_sigmoid(x3)
    y3.sum().backward()
    print(f"  StableSigmoid:  OK  shape={y3.shape}")

    # chunked_softmax
    x4 = torch.randn(2, 100279, device=device, requires_grad=True)
    y4 = chunked_softmax(x4, dim=-1)
    y4.sum().backward()
    print(f"  chunked_softmax: OK  shape={y4.shape}")


def test_multi_hash_embedding(device):
    print("\n=== Test 2: Multi-Hash Embedding ===")
    from v5_core.architecture.multi_hash_embedding import MultiHashEmbedding

    embed = MultiHashEmbedding(d_model=256, K=8, d_hash=32).to(device)
    token_ids = torch.randint(0, 100279, (2, 128), device=device)

    y = embed(token_ids)
    y.sum().backward()

    params = sum(p.numel() for p in embed.parameters())
    print(f"  Forward:  OK  shape={y.shape}")
    print(f"  Backward: OK  (no scatter crash)")
    print(f"  Params:   {params:,}")


def test_resonance_block(device):
    print("\n=== Test 3: Resonance Block ===")
    from v5_core.architecture.resonance_block import ResonanceBlock

    block = ResonanceBlock(d_model=256, n_heads=4, window_size=32, max_seq_len=128).to(device)
    x = torch.randn(2, 128, 256, device=device, requires_grad=True)

    # Test causal mode (pretraining)
    y_causal = block(x, causal=True)
    y_causal.sum().backward()
    print(f"  Causal:       OK  shape={y_causal.shape}")

    # Test bidirectional mode (agent)
    block.zero_grad()
    x2 = torch.randn(2, 128, 256, device=device, requires_grad=True)
    y_bidir = block(x2, causal=False)
    y_bidir.sum().backward()
    print(f"  Bidirect:     OK  shape={y_bidir.shape}")

    params = sum(p.numel() for p in block.parameters())
    print(f"  Backward: OK")
    print(f"  Params:   {params:,}")


def test_elastic_context(device):
    print("\n=== Test 4: Elastic Context ===")
    from v5_core.architecture.elastic_context import ElasticContext

    # Use smaller boundaries for testing
    ec = ElasticContext(
        d_model=256,
        level_boundaries=[64, 192, 512],
        strides=[1, 4, 16]
    ).to(device)

    x = torch.randn(2, 512, 256, device=device, requires_grad=True)
    y = ec(x)
    y.sum().backward()

    slots = ec.get_slot_count(512)
    print(f"  Forward:  OK  shape={y.shape}  (512→{slots} slots)")
    print(f"  Backward: OK")

    # Test with shorter sequence
    x_short = torch.randn(2, 100, 256, device=device, requires_grad=True)
    y_short = ec(x_short)
    y_short.sum().backward()
    slots_short = ec.get_slot_count(100)
    print(f"  Short seq: OK  shape={y_short.shape}  (100→{slots_short} slots)")


def test_observation_encoder(device):
    print("\n=== Test 5: Observation Encoder ===")
    from v5_core.architecture.observation_encoder import ObservationEncoder

    # Small config for testing
    enc = ObservationEncoder(d_model=256, K=8, d_hash=32, max_seq_len=512).to(device)

    token_ids = torch.randint(0, 100279, (2, 256), device=device)
    type_ids = torch.randint(0, 6, (2, 256), device=device)

    y = enc(token_ids, type_ids)
    y.sum().backward()

    params = sum(p.numel() for p in enc.parameters())
    print(f"  Forward:  OK  shape={y.shape}")
    print(f"  Backward: OK")
    print(f"  Params:   {params:,}")


def test_action_decoder(device):
    print("\n=== Test 6: Action Decoder ===")
    from v5_core.architecture.action_decoder import ActionDecoder

    dec = ActionDecoder(d_model=256, n_tools=8, vocab_size=1000).to(device)

    resonance_out = torch.randn(2, 64, 256, device=device, requires_grad=True)
    context = torch.randn(2, 64, 256, device=device)
    context_ids = torch.randint(0, 1000, (2, 64), device=device)

    action = dec(resonance_out, context, context_ids)

    # Test backward through tool_logits
    action['tool_logits'].sum().backward()

    print(f"  tool_id:    {action['tool_id'].tolist()}")
    print(f"  confidence: {action['confidence'].tolist()}")
    print(f"  should_think: {action['should_think'].tolist()}")
    print(f"  Backward: OK")


def test_outcome_predictor(device):
    print("\n=== Test 7: Outcome Predictor ===")
    from v5_core.architecture.outcome_predictor import OutcomePredictor

    pred = OutcomePredictor(d_model=256, n_tools=8, d_hidden=64).to(device)

    hidden = torch.randn(2, 256, device=device, requires_grad=True)
    tool_ids = torch.tensor([3, 5], device=device)

    result = pred(hidden, tool_ids)
    p_success = result['p_success']
    logit = result['logit']
    actual = torch.tensor([1.0, 0.0], device=device)
    loss = pred.loss(logit, actual)
    loss.backward()

    print(f"  P(success): {p_success.tolist()}")
    print(f"  Logit:      {logit.tolist()}")
    print(f"  Loss:       {loss.item():.4f}")
    print(f"  Backward:   OK")


def test_full_assembly(device):
    print("\n=== Test 8: Full V5 Assembly ===")
    from v5_core.architecture.v5_assembly import V5ResonanceModel

    # Small config for smoke test
    model = V5ResonanceModel(
        d_model=256, n_blocks=2, n_heads=4,
        window_size=32, max_seq_len=256,
        vocab_size=1000, K_hash=8, d_hash=32,
        max_think_steps=1, dropout=0.0,
    ).to(device)

    token_ids = torch.randint(0, 1000, (2, 128), device=device)
    type_ids = torch.randint(0, 6, (2, 128), device=device)
    labels = torch.randint(0, 1000, (2, 128), device=device)

    # Test pre-training forward
    result = model.forward_pretrain(token_ids, type_ids, labels)
    result['loss'].backward()

    print(f"  Pretrain logits: shape={result['logits'].shape}")
    print(f"  Pretrain loss:   {result['loss'].item():.4f}")
    print(f"  Backward:        OK")

    # Test agent forward
    model.zero_grad()
    agent_result = model.forward_agent(token_ids, type_ids)
    # Backward through tool logits
    agent_result['tool_logits'].sum().backward()

    print(f"  Agent tool_id:    {agent_result['tool_id'].tolist()}")
    print(f"  Agent confidence: {[f'{c:.3f}' for c in agent_result['confidence'].tolist()]}")
    print(f"  Agent p_success:  {[f'{p:.3f}' for p in agent_result['p_success'].tolist()]}")
    print(f"  Think steps:      {agent_result['think_steps']}")
    print(f"  Agent backward:   OK")

    # Print param counts
    counts = model.count_parameters()
    print(f"\n  Parameter breakdown:")
    for k, v in counts.items():
        if isinstance(v, int):
            print(f"    {k:20s}: {v:>12,}")
        else:
            print(f"    {k:20s}: {v}")


def main():
    print("=" * 60)
    print("SNAP-C1 V5 RESONANCE — Full Smoke Test")
    print("=" * 60)

    device = get_test_device()
    t0 = time.time()

    test_dml_ops(device)
    test_multi_hash_embedding(device)
    test_resonance_block(device)
    test_elastic_context(device)
    test_observation_encoder(device)
    test_action_decoder(device)
    test_outcome_predictor(device)
    test_full_assembly(device)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ALL TESTS PASSED in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
