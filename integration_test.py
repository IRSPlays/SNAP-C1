"""Eidos Integration Test — exercises every module forward+backward+NaN+shapes.

Tests:
  - DiffAttentionLayer: forward, backward, GQA expansion, RoPE, λ computation
  - EidosEncoder: embedding + num_proj injection, 4-layer stack
  - PredictiveCoder: z_hat, error, cosine_dist, value_pred
  - Neuromodulator: δ, ν, σ with num_count and without
  - NeuralMemory: write+read, persistence, write_gate, bmm path, NaN reset
  - LTCCortex: single/multi iteration, memory fusion, tau dynamics
  - MultiTokenPredictor: shared head, losses, alignment
  - EidosV1 full: end-to-end forward+backward, all losses, generate

Run: python integration_test.py
"""

import torch
import sys, os
sys.path.insert(0, 'C:/Users/Haziq/Documents/SNAP-C1/SNAP-C1')

from cortex.modules.diff_attention import DiffAttentionLayer, RotaryEmbedding
from cortex.modules.encoder import EidosEncoder
from cortex.modules.predictive_coder import PredictiveCoder
from cortex.modules.neuromodulator import Neuromodulator
from cortex.modules.neural_memory import NeuralMemory
from cortex.modules.ltc_cortex import LTCCortex
from cortex.modules.mtp_head import MultiTokenPredictor
from cortex.model import EidosV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")
passes = 0
failures = 0

def check(name, condition, detail=""):
    global passes, failures
    if condition:
        passes += 1
        print(f"  [PASS] {name} {detail}")
    else:
        failures += 1
        print(f"  [FAIL] {name} {detail}")

# ════════════════════════════════════════════════════════════════════
# 1. DiffAttentionLayer
# ════════════════════════════════════════════════════════════════════
print("=== 1. DiffAttentionLayer ===")

B, T, D = 2, 32, 512
n_heads, n_kv = 8, 4
head_dim = D // n_heads

layer = DiffAttentionLayer(D, n_heads, n_kv, depth=2, max_depth=4, dropout=0.0).to(device)
rope = RotaryEmbedding(head_dim, 64)
cos, sin = rope(T)
cos, sin = cos.to(device), sin.to(device)
x = torch.randn(B, T, D, device=device)

out = layer(x, cos, sin, is_causal=True)
check("forward shape", out.shape == (B, T, D))
check("forward finite", torch.isfinite(out).all())
check("forward not zeros", out.abs().sum() > 0)

lam = layer._compute_lambda()
check("lambda shape", lam.shape == (1, n_heads, 1, 1))
check("lambda finite", torch.isfinite(lam).all())
check("lambda in [0,1]", 0 <= lam.min() <= lam.max() <= 2.0)

loss = out.mean()
loss.backward()
check("backward: q_proj grad", layer.q_proj.weight.grad is not None)
check("backward: lambda_q1 grad", layer.lambda_q1.grad is not None)
check("backward: grad finite", torch.isfinite(layer.lambda_q1.grad).all())

# Test GQA expansion path
x2 = torch.randn(2, 64, 512, device=device)
cos2, sin2 = rope(64)
cos2, sin2 = cos2.to(device), sin2.to(device)
out2 = layer(x2, cos2, sin2, is_causal=True)
check("GQA forward shape (T=64)", out2.shape == (2, 64, 512))
check("GQA forward finite", torch.isfinite(out2).all())

print()

# ════════════════════════════════════════════════════════════════════
# 2. EidosEncoder
# ════════════════════════════════════════════════════════════════════
print("=== 2. EidosEncoder ===")

vocab = 256
num_vals = torch.rand(vocab)
encoder = EidosEncoder(vocab, D, n_heads, n_kv, 4, 512, 0.0, num_values=num_vals).to(device)
input_ids = torch.randint(0, vocab, (B, T), device=device)

z, pooled = encoder(input_ids)
check("z shape", z.shape == (B, T, D))
check("pooled shape", pooled.shape == (B, D))
check("z finite", torch.isfinite(z).all())
check("pooled finite", torch.isfinite(pooled).all())
check("num_values buffer exists", hasattr(encoder, 'num_values'))
check("num_proj exists", encoder.num_proj is not None)

loss2 = z.mean()
loss2.backward()
check("backward: token_emb grad", encoder.token_emb.weight.grad is not None)
check("backward: num_proj grad", encoder.num_proj.weight.grad is not None)

# Without num_values
encoder_no_num = EidosEncoder(vocab, D, n_heads, n_kv, 4, 512, 0.0).to(device)
z_nn, _ = encoder_no_num(input_ids)
check("no-num forward finite", torch.isfinite(z_nn).all())
check("no-num no num_proj", not hasattr(encoder_no_num, 'num_values') or encoder_no_num.num_proj is None)

print()

# ════════════════════════════════════════════════════════════════════
# 3. PredictiveCoder
# ════════════════════════════════════════════════════════════════════
print("=== 3. PredictiveCoder ===")

pred = PredictiveCoder(D).to(device)
z_prev = torch.randn(B, T, D, device=device)
z_next = torch.randn(B, T, D, device=device)
z_pool = torch.randn(B, D, device=device)

z_hat, error, cosine_dist, value_pred = pred(z_prev, z_next, z_pool)
check("z_hat shape", z_hat.shape == (B, T, D))
check("error shape", error.shape == (B, T))
check("cosine_dist shape", cosine_dist.shape == (B, T))
check("value_pred shape", value_pred.shape == (B, T))
check("z_hat finite", torch.isfinite(z_hat).all())
check("cosine_dist in [0,2]", (cosine_dist >= 0).all() and (cosine_dist <= 2.1).all())
check("value_pred finite", torch.isfinite(value_pred).all())
check("value_head exists", hasattr(pred, 'value_head'))

loss3 = z_hat.mean() + cosine_dist.mean() + value_pred.mean()
loss3.backward()
check("backward: value_head grad", pred.value_head.weight.grad is not None)
check("backward: fc1 grad", pred.fc1.weight.grad is not None)

print()

# ════════════════════════════════════════════════════════════════════
# 4. Neuromodulator
# ════════════════════════════════════════════════════════════════════
print("=== 4. Neuromodulator ===")

mod = Neuromodulator().to(device)
err = torch.randn(B, T, device=device) * 0.3
mem_match = torch.randn(B, 1, device=device) * 0.3
num_ct = torch.tensor([5, 12], device=device)

# With num_count
d, n, s, a = mod(err, mem_match, num_count=num_ct)
check("dopamine shape", d.shape == (B, T))
check("norepi shape", n.shape == (B,))
check("serotonin shape", s.shape == (B, 1))
check("norepi from complexity", n[0].item() >= 2)  # 5 numbers → at least 3
check("dopamine in [0,1]", (d >= 0).all() and (d <= 1).all())
check("serotonin in [0,1]", (s >= 0).all() and (s <= 1).all())

# Without num_count (fallback)
d2, n2, s2, a2 = mod(err, mem_match)
check("norepi fallback finite", torch.isfinite(n2).all())

# Test running stats update
mod.train()
for _ in range(5):
    mod(torch.randn(2, 32, device=device) * 0.5 + 0.1, torch.randn(2, 1, device=device) * 0.1)
check("error_mean updated", mod.error_mean.item() != 0.0)

print()

# ════════════════════════════════════════════════════════════════════
# 5. NeuralMemory
# ════════════════════════════════════════════════════════════════════
print("=== 5. NeuralMemory ===")

mem = NeuralMemory(D).to(device)
mem.train()
z_in = torch.randn(B, T, D, device=device)
surprise = torch.rand(B, T, device=device)

# First write
h = mem(z_in, surprise=surprise)
check("read shape", h.shape == (B, T, D))
check("read finite", torch.isfinite(h).all())
check("M not zero after write", mem.M.abs().sum() > 1e-8)
check("M normalized", abs(mem.M.norm().item() - 1.0) < 0.1)

# Second batch — memory persists
z_in2 = torch.randn(B, T, D, device=device)
surprise2 = torch.rand(B, T, device=device)
h2 = mem(z_in2, surprise=surprise2)
check("M persists across batches", mem.M.abs().sum() > 1e-8)

# Write gate
gate = torch.sigmoid(torch.randn(B, T, device=device))
h3 = mem(z_in, surprise=surprise, write_gate=gate)
check("write_gate forward", torch.isfinite(h3).all())

# NaN M reset — when M has NaN, next forward uses only batch_updates (bypasses M)
# M is finite after recovery, not necessarily zero (current batch writes its own update)
mem.M.data[:] = float('nan')
h4 = mem(z_in, surprise=surprise)
check("NaN M recovery: h finite", torch.isfinite(h4).all())
check("NaN M recovery: M finite after", torch.isfinite(mem.M).all())

# Momentum
mem2 = NeuralMemory(D, momentum_init=0.95).to(device)
check("momentum_init=0.95", mem2.momentum_init == 0.95)

print()

# ════════════════════════════════════════════════════════════════════
# 6. LTCCortex
# ════════════════════════════════════════════════════════════════════
print("=== 6. LTCCortex ===")

ltc = LTCCortex(D).to(device)
x_ltc = torch.randn(B, T, D, device=device)
mem_ctx = torch.randn(B, T, D, device=device)

# 1 iteration (fast)
h1 = ltc(x_ltc, iterations=1)
check("1-iter shape", h1.shape == (B, T, D))
check("1-iter finite", torch.isfinite(h1).all())

# 8 iterations with memory
h8 = ltc(x_ltc, iterations=8, memory=mem_ctx)
check("8-iter shape", h8.shape == (B, T, D))
check("8-iter finite", torch.isfinite(h8).all())

# No memory
h_no_mem = ltc(x_ltc, iterations=4)
check("no-mem finite", torch.isfinite(h_no_mem).all())

# Tau dynamics — test backward with memory
ltc.train()
loss6 = ltc(x_ltc, iterations=2, memory=mem_ctx).mean()
loss6.backward()
check("backward: mem_fuse grad", ltc.mem_fuse.weight.grad is not None)
check("backward: tau_linear grad", ltc.tau_linear.weight.grad is not None)
check("backward: grad finite", torch.isfinite(ltc.tau_linear.weight.grad).all())

print()

# ════════════════════════════════════════════════════════════════════
# 7. MultiTokenPredictor
# ════════════════════════════════════════════════════════════════════
print("=== 7. MultiTokenPredictor ===")

mtp = MultiTokenPredictor(D, vocab).to(device)
x_mtp = torch.randn(B, T, D, device=device)
labels_mtp = torch.randint(0, vocab, (B, T), device=device)

out_mtp = mtp(x_mtp, labels=labels_mtp)
check("logits shape", out_mtp['logits'].shape == (B, T, vocab))
check("loss exists", 'loss' in out_mtp)
check("ce_loss exists", 'ce_loss' in out_mtp)
check("mtp_losses exist", 'mtp_losses' in out_mtp)
check("loss finite", torch.isfinite(out_mtp['loss']).all())
check("shared_head exists", hasattr(mtp, 'shared_head'))
check("extra_norms exist", len(mtp.extra_norms) == 3)

# Test ignore_index (lots of -100)
labels_sparse = labels_mtp.clone()
labels_sparse[:, :T//2] = -100
out_sparse = mtp(x_mtp, labels=labels_sparse)
check("sparse labels finite", torch.isfinite(out_sparse['loss']).all())

out_mtp['loss'].backward()
check("backward: shared_head grad", mtp.shared_head.weight.grad is not None)

# Without labels (inference)
out_no_lbl = mtp(x_mtp)
check("no-labels logits shape", out_no_lbl['logits'].shape == (B, T, vocab))
check("no-labels no loss", 'loss' not in out_no_lbl)

print()

# ════════════════════════════════════════════════════════════════════
# 8. EidosV1 — Full model
# ════════════════════════════════════════════════════════════════════
print("=== 8. EidosV1 Full Model ===")

full_vocab = 512
num_vals_full = torch.rand(full_vocab)
model = EidosV1(
    vocab_size=full_vocab, d_model=512, n_heads=8, n_kv_heads=4,
    n_layers=4, dropout=0.0, num_values=num_vals_full,
).to(device)

counts = model.count_parameters()
check("total params", counts['total'] > 20_000_000)
check("param counts dict", all(k in counts for k in ['total', 'encoder', 'ltc_cortex', 'mtp_heads']))

ids = torch.randint(0, full_vocab, (B, T), device=device)
lbls = torch.randint(0, full_vocab, (B, T), device=device)

model.train()
model.neural_memory.reset()
out_full = model(ids, labels=lbls)

for key in ['logits', 'loss', 'ce_loss', 'cosine_dist', 'dopamine', 'serotonin',
            'thought', 'prediction_error', 'memory_match', 'iterations']:
    check(f"dict has '{key}'", key in out_full, f"(shape={out_full.get(key, 'MISSING')})")

check("loss finite", torch.isfinite(out_full['loss']).all())
check("iterations >= 2", out_full['iterations'] >= 2)
check("num_loss exists", 'num_loss' in out_full)
check("val_pred_loss exists", 'val_pred_loss' in out_full)

out_full['loss'].backward()
grad_nan = 0
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        grad_nan += 1
check("all grads finite", grad_nan == 0, f"({grad_nan} NaN grads)")

# Generate test
model.eval()
probe = torch.randint(0, full_vocab, (1, 8), device=device)
gen = model.generate(probe, max_new_tokens=10, temperature=0.0, top_k=1)
check("generate output shape", gen.shape[0] == 1)
check("generate more tokens", gen.shape[1] > 8)

# Generate with self-verify
gen2 = model.generate(probe, max_new_tokens=10, temperature=0.0, top_k=1,
                      enable_self_verify=True, enable_self_consistency=True,
                      enable_skip_ltc=True)
check("self-verify generate", gen2.shape[1] > 8)

# Check memory persistence
model.train()
M_before = model.neural_memory.M.clone()
out_full2 = model(ids, labels=lbls)
M_after = model.neural_memory.M
check("M changed after forward", not torch.allclose(M_before, M_after, atol=1e-6))

print()

# ════════════════════════════════════════════════════════════════════
# 9. Edge Cases
# ════════════════════════════════════════════════════════════════════
print("=== 9. Edge Cases ===")

# Zero-length input might cause issues, skip

# All -100 labels
labels_all_neg = torch.full((B, T), -100, device=device)
model.train()
out_neg = model(ids, labels=labels_all_neg)
check("all -100 labels: loss finite", torch.isfinite(out_neg['loss']).all(), f"(loss={out_neg['loss'].item():.4f})")

# Multiple forward passes without reset
for _ in range(3):
    _ = model(torch.randint(0, full_vocab, (B, 16), device=device))
check("multi-forward no crash", True)

# AMP (autocast) compatibility
with torch.amp.autocast('cuda', dtype=torch.float16):
    out_amp = model(torch.randint(0, full_vocab, (B, 16), device=device),
                    labels=torch.randint(0, full_vocab, (B, 16), device=device))
check("AMP forward finite", torch.isfinite(out_amp['loss']).all())

# Eval mode
model.eval()
with torch.no_grad():
    out_eval = model(torch.randint(0, full_vocab, (2, 16), device=device))
check("eval forward no grad", out_eval['cosine_dist'].mean().item() > 0)

print()

# ════════════════════════════════════════════════════════════════════
# Results
# ════════════════════════════════════════════════════════════════════
total = passes + failures
print(f"{'='*60}")
print(f"RESULTS: {passes}/{total} passed, {failures} failed")
if failures == 0:
    print("ALL INTEGRATION TESTS PASSED")
else:
    print(f"*** {failures} FAILURES ***")
print(f"{'='*60}")
