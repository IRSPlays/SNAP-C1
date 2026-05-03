# Eidos V1 — Complete System Upgrade Plan

## 1. Architecture Overview

```
tokens → Diff Attention (4 layers, GQA) → z, pooled
  │
  ├─ PredictiveCoder     →  ẑ, ε, cosine_dist, value_pred
  ├─ Neuromodulator      →  δ (write gate), ν (LTC budget), σ (memory/cortex blend)
  ├─ NeuralMemory        →  M [D×D] persistent, write-gated, Hebbian update
  ├─ LTCCortex           →  h_cortex (ν iterations, memory-augmented)
  ├─ Integrator          →  σ·h_mem + (1-σ)·h_cortex
  └─ MTP (4 heads)       →  logits + num_head(per-position)
```

**Papers**: Differential Transformer (ICLR 2025), Titans (Dec 2024), DeepSeek V3 (Dec 2024), LTC (2020)

**Current config**: d_model=512, n_heads=8, n_kv_heads=4, n_layers=4, seq_len=192, batch=2, accum=4, lr=5e-5, dropout=0.2

---

## 2. Current Problems

| Problem | Root Cause | Impact |
|---|---|---|
| Training too slow | No AMP, small batches, seq_len=192 wastes 80% on padding | 26 min/epoch on synthetic |
| Eval gap misleading | Pretrain eval uses GSM8K (different vocab), not held-out synthetic | +15.29 gap looks like overfit, is actually domain mismatch |
| 3.3% GSM8K accuracy | Architecture wired for generic LM, not math | Model learns format, not arithmetic |
| NeuralMemory reset per epoch | `reset()` called in epoch loop | Never accumulates knowledge |
| ν always ~1 | Neuromodulator gets error-based ν, not complexity-based | LTC wasted (4.7M dead params) |
| num_proj inaudible | std=0.005 vs token std=0.044 | Numbers indistinguishable from text |
| num_head on pooled output | Global guess, not per-position | No step-by-step value tracking |

---

## 3. What We Already Fixed (in this upgrade)

| Fix | File | Before | After |
|---|---|---|---|
| Diff attention NaN guard | `diff_attention.py` | `enable_gqa=True` (CUDA bug) | Manual GQA expansion |
| NeuralMemory NaN guard | `neural_memory.py` | Writes NaN M → poisons all batches | Reset M if NaN detected |
| Memory momentum | `neural_memory.py` | 0.90 | 0.95 (stronger retention) |
| num_proj amplification | `encoder.py` | std=0.005 | std=0.05 (10× louder) |
| PredictiveCoder value_head | `predictive_coder.py` | Embed prediction only | + scalar value prediction |
| Neuromodulator ν | `neuromodulator.py` | `1 + ε̄·4` (usually 1) | `2 + num_count//3` (2-16) |
| Neuromodulator σ | `neuromodulator.py` | temp=20 (binary) | `(match-0.5)×5` (soft blend) |
| Memory persistence | `train.py` | Reset every epoch | Reset once at training start |
| Per-position num_head | `model.py` | Pooled → scalar | Per-position → predict next value |
| Value prediction loss | `model.py` | None | MSE on number positions |
| Synthetic data pipeline | `synthetic_data.py` (NEW) | None | 50K examples, 4 difficulty levels |
| Pretrain/finetune split | `train.py` | Single checkpoint name | `eidos_pretrain_best.pt` / `eidos_v1_best.pt` |
| Auto-load pretrained | `train.py` | Always scratch | Finetune loads pretrain checkpoint |
| Synthetic eval during pretrain | `train.py` | GSM8K eval (wrong domain) | 90/10 synthetic train/eval split |
| Full Q/A display | `train.py` | Truncated, eval-only | Training examples + full generation |
| Evaluate script for synthetic | `evaluate_synthetic.py` (NEW) | None | 5 difficulty levels, per-op accuracy |
| NaN recovery | `train.py` | Crash on NaN | Reset memory + skip batch |

---

## 4. Phase 1 — Speed Optimization (30 lines, 3-5× faster)

```
CONFIG changes:
    'amp': True                      # fp16 mixed precision → 40% faster
    'batch_size': 4                  # 2× throughput (with AMP, VRAM freed)
    'seq_len': 96                    # Synthetic data is 15-40 tokens → 2× less compute
    'accum_steps': 2                 # batch=4, accum=2 = same effective batch
    
    # Reduce LTC cost during training
    ltc_training_iters: 2            # Override ν during training (not inference)
    
    # Reduce generation overhead
    'gen_interval': 10               # Generate every 10 epochs, not 3
    'gen_prompts': 3                 # 3 prompts, not 8

Training loop:
    # torch.compile for graph optimization
    import torch
    model = torch.compile(model)     # 20-30% faster (PyTorch 2.10+)
    
    # Warmup compile on first batch
    out = model(first_batch)         # Compile first, then train

NeuralMemory optimization:
    # Replace einsum('bt,btd,bte->bde', w, k, v) with batched bmm
    # O(B×T×D²) → same but faster on CUDA
    w_k = (w.unsqueeze(-1) * k).transpose(1, 2)  # [B, D, T]
    batch_updates = w_k @ v                       # [B, D, D]
```

**Expected**: 26 min/epoch → 5-7 min/epoch. 30 epochs in 3 hours instead of 13.

---

## 5. Phase 2 — Architecture Efficiency (preserves all components)

### 5a. MTP Head Sharing (17M → 6M params)

```
Current: 3 separate Linear(512, vocab) → 3 × 512 × vocab params
Fixed:   All extra heads share ONE projection matrix
         Each head has its own RMSNorm (tiny) + shared Linear
         → 512 × vocab params instead of 3 × 512 × vocab

Saves:  2 × 512 × vocab = 11.6M params on 11K vocab
        At 100M scale: saves ~50M params
```

### 5b. Low-Rank NeuralMemory (variable cost)

```
Current: M = [D, D] = [512, 512] = 262K elements
         Read: q·M / √D → O(D²)
         Write: einsum → O(B×T×D²)

Option A: Low-rank M = U·V^T where U,V ∈ [D, R], R=64
         Storage: 2×D×R = 65K (vs 262K)
         Write:  O(B×T×D×R) instead of O(B×T×D²) — 8× faster
         Read:   O(D×R) instead of O(D²) — 8× faster
         
Option B: Multiple small memory banks (4 banks of [256,256])  
         Each bank specializes. Parallel read/write.
         4× bandwidth, same total storage.
```

### 5c. Reversible Layers (for scaling)

```
Standard: Store activations for backward → O(L×B×T×D) memory
Reversible: Recompute activations during backward → O(1) memory
Trade: 2× forward passes = 30% slower, but 3× VRAM savings
Use for: Scaling to d_model=768, n_layers=8 on same GPU
```

### 5d. Flash Attention (for speed)

```
Current: Manual GQA expansion + scaled_dot_product_attention
Fix: Use torch's flash attention backend:
     F.scaled_dot_product_attention(q, k, v, ...) with SDPBackend.FLASH_ATTENTION
     → 2-3× faster attention, O(√n) memory instead of O(n²)
```

### Parameter Breakdown by Scale

| Config | d_model | n_layers | Params | VRAM (fp16+amp) |
|---|---|---|---|---|
| Current | 512 | 4 | 43M | 1.8 GB |
| Efficient | 512 | 4 | 35M | 1.5 GB |
| Scaled | 640 | 6 | 85M | 2.8 GB |
| Max Local | 768 | 8 | 150M | 3.8 GB |
| Cloud | 1024 | 12 | 350M | 8 GB |
| Cloud | 1536 | 24 | 1.2B | 24 GB |
| Cloud | 2048 | 32 | 3.1B | 48 GB |
| Cloud | 3072 | 40 | 8.2B | 80 GB |

---

## 6. Phase 3 — Alternative Training (Forward-Forward)

### The Core Idea

Replace backprop through encoder layers with Forward-Forward (Hinton, 2022):

```
POSITIVE pass: Real data → compute "goodness" per layer → maximize
NEGATIVE pass: Shuffled/corrupted data → compute "goodness" → minimize

Goodness = ‖layer_output‖² (sum of squared activations)
Loss per layer = -log(σ(goodness_pos)) - log(1-σ(goodness_neg))

Each encoder layer trains INDEPENDENTLY via local loss.
No gradient flows between layers.
Layers can train in PARALLEL (4 layers = 4× speed).
```

### What changes in Eidos

```
Encoder layers (DiffAttention × 4):
  POSITIVE:  Q: "5+3=?" A: "8"  → encoder → high activation norms
  NEGATIVE:  Q: "5+3=?" A: "12" → encoder → low activation norms
  Local loss per layer drives representation learning

NeuralMemory:
  Already Hebbian — NO change needed. Zero gradient cost.
  
PredictiveCoder:
  Already has local loss (MSE on z_hat and value_pred).
  Just increase weight from 0.01 → 0.5.

LTCCortex:
  Local stability loss: minimize ||h_{t+1} - h_t|| after convergence.
  Already stable by design (LTC paper proves boundedness).

MTP head:
  Still needs backprop — but through ONE linear layer.
  Fast (1ms per batch instead of 10ms).
```

### Speed Impact

| Component | Current (backprop through all) | Forward-Forward |
|---|---|---|
| Encoder (4 layers) | 40% of time | 25% (parallel) |
| NeuralMemory | 10% | 10% (unchanged) |
| LTC | 25% | 25% (local loss) |
| MTP | 10% | 10% (1-layer backprop) |
| PredictiveCoder | 5% | 5% (local loss) |
| **Total** | 100% | **~50%** (5-10× speedup potential) |

### Risk

Forward-Forward is experimental. May not match backprop accuracy for language tasks. Mitigation: keep backprop as fallback, test FF on synthetic arithmetic first (simpler domain).

---

## 7. Zero-Cost Advantages (already built, just need wiring)

These require NO new components. Only changes to model.py/train.py logic.

### 7a. Self-Verification at Inference

```python
# During generation:
z_hat, error, cosine_dist, value_pred = predictive_coder(...)
if cosine_dist.mean() > threshold:
    # Model is confused — re-process with higher LTC iterations
    h_cortex = ltc_cortex(z, iterations=16, memory=h_mem)  # think harder
    out = sigma * h_mem + (1 - sigma) * h_cortex
    result = mtp(out, labels=None)
```

### 7b. Self-Consistency from MTP Heads

```python
# 4 MTP heads predict at different horizons
head_0 = argmax(logits_main[:, -1])           # next token
head_1 = argmax(logits_extra[:, -1])          # 2 tokens ahead
# If all 4 agree on the answer number → high confidence
# If they diverge → flag as uncertain, re-generate
```

### 7c. Curiosity-Driven Curriculum

```python
# Track prediction error per example
if example_prediction_error > threshold:
    # This example surprised the model → revisit it more often
    train_dataset.weight[idx] *= 1.2  # upweight hard examples
```

### 7d. Skip Computation on Easy Cases

```python
# At inference:
if problem_has_fewer_than_3_numbers:
    skip LTC entirely (out = h_mem)  # memory recall only
    skip predictive coder check
    # 5× faster inference for simple problems
```

### 7e. Built-in Speculative Decoding

```python
# MTP predicts 4 tokens at once
candidates = [argmax(head_i) for head_i in [main, extra1, extra2, extra3]]
# Main head verifies: if candidates match → accept all 4
# If mismatch → fallback to single-token generation
# 2-4× faster inference
```

---

## 8. Small Additions (leveraging existing components)

### 8a. Sleep/Replay Consolidation

```python
# After each epoch:
top_k_surprising = select_k_examples_with_highest_prediction_error(train_data, k=1000)
for example in top_k_surprising:
    # Replay with HIGH LTC iterations → cortex learns from memory
    h_cortex = ltc_cortex(z, iterations=16, memory=h_mem)
    # Memory → cortex knowledge transfer (like sleep consolidation)
```

### 8b. Contrastive Learning

```python
# Positive pair: (question, correct_answer) → cosine_sim(pooled_q, pooled_a)
# Negative pair: (question, wrong_answer) → cosine_sim should be lower
contrastive_loss = -log(exp(sim_pos) / (exp(sim_pos) + Σ exp(sim_neg)))
# Teaches model to distinguish right from wrong without token-level labels
```

### 8c. Progressive Layer Training

```python
# Phase 1: Train only layer 0 (epochs 1-5)
# Phase 2: Train layers 0-1 (epochs 6-10)
# Phase 3: Train layers 0-2 (epochs 11-15)
# Phase 4: Full model (epochs 16-30)
# Each phase: freeze earlier layers, train new ones
# → faster convergence, less overfitting
```

### 8d. Lifelong Learning Sequence

```python
# Step 1: Train on arithmetic → save memory snapshot
# Step 2: Freeze memory, train on language → memory provides arithmetic base
# Step 3: Freeze memory, train on code → memory provides math + language
# Step 4: Unfreeze, joint train → full multi-capability model
# Memory accumulates knowledge across phases — no forgetting
```

### 8e. Number Stream (Dual Processing)

```python
# Instead of just adding num_proj to token embeddings:
# Run number values through a SEPARATE small MLP
# Cross-attend: text stream attends to number stream
# Result: numbers get REAL computation, not just embedding addition

num_stream = small_mlp(num_values[input_ids])  # [B, T, D]
text_stream = encoder(input_ids)                # [B, T, D]
combined = text_stream + cross_attention(text_stream, num_stream)
```

---

## 9. Architecture Upgrades (new components, significant impact)

### 9a. Mixture of Experts in Encoder

```python
# Replace standard FFN with MoE
# 4 experts: arithmetic, language, pattern, fallback
# Router: learned gating picks top-1 expert per token
# Same FLOPs as current FFN, 4× capacity

class MoE_FFN(nn.Module):
    def __init__(self, d_model, n_experts=4):
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([SwiGLU_FFN(d_model) for _ in range(n_experts)])
    
    def forward(self, x):
        gates = softmax(self.router(x))           # [B, T, E]
        expert_out = torch.stack([e(x) for e in self.experts])  # [E, B, T, D]
        return (gates.unsqueeze(-1) * expert_out).sum(0)  # weighted blend
```

### 9b. Fast Weights via Neuromodulator

```python
# Neuromodulator outputs weight MODIFICATIONS, not just modulation signals
# δ, ν, σ, α → also ΔW for encoder layers
# Each input gets a CUSTOMIZED model in one forward pass
# Hypernetwork-style: neuromodulator → weight shift → adapted encoder

delta_w = self.weight_modulator(combined_signal)  # [params_dim]
# Apply to encoder attention weights (soft gating)
adapted_q_proj = q_proj * (1 + Δw_q)
```

### 9c. Memory as Structured Graph

```python
# Instead of flat K-V matrix:
# Memory = (entity_key, relation, entity_value) triples
# Write: extract (subject, predicate, object) from input
# Read: graph traversal → "what relates to X via relation R?"
# Actual reasoning, not pattern matching

class GraphMemory(nn.Module):
    def write(self, s, p, o):  # subject, predicate, object
        self.triples.append((s, p, o))
    
    def read(self, query_entity, query_relation):
        # Find all triples matching (query, relation, ?)
        # Return matched objects
```

---

## 10. The AGI Features (Eidos-Only Advantages)

### 10a. Continuous Learning

NeuralMemory persists forever. Each new example updates M via Hebbian outer product. Old knowledge decays slowly (α=0.95). No fine-tuning boundary — just keep training.

All other models: weight update overwrites old knowledge. Fine-tuning = catastrophic forgetting.

### 10b. Test-Time Reasoning Depth

At inference, ν controls LTC depth. Simple question: ν=2, 50ms. Hard question: ν=64, 1.5s. The model THINKS LONGER without architectural change.

Standard models: fixed depth. Can't scale compute per problem.

### 10c. Explicit Knowledge vs Implicit Reasoning

σ (serotonin) blends between memory recall and fresh reasoning.
- σ→1: "I know this" — fast recall from NeuralMemory
- σ→0: "I need to think" — deep LTC recurrence
- The model KNOWS what it knows vs what it needs to compute

### 10d. Self-Improving Loop

```
1. Generate harder problems (from learned distribution)
2. Solve them (current model)
3. Compute solutions (external calculator for correctness)
4. Train on verified correct solutions
5. Generate even harder problems → repeat
```

No limit. The model gets smarter the longer it runs.

---

## 11. Implementation Priority Order

```
WEEK 1: Speed (Phase 1)
  □ Enable AMP (fp16)
  □ Increase batch_size to 4
  □ Reduce seq_len to 96 for synthetic
  □ torch.compile(model)
  □ NeuralMemory einsum → bmm
  □ Skip generation during most epochs
  
WEEK 2: Zero-Cost Advantages (Section 7)
  □ Self-verification at inference
  □ MTP self-consistency check
  □ Curiosity-driven curriculum
  □ Skip LTC on easy problems
  
WEEK 3: Architecture Efficiency (Section 5)
  □ MTP head sharing
  □ Low-rank NeuralMemory (R=64)
  □ Flash Attention integration
  □ Reversible layers (for scaling)
  
WEEK 4: Small Additions (Section 8)
  □ Sleep/replay consolidation
  □ Contrastive learning loss
  □ Progressive layer training
  □ Number stream (dual processing)
  
WEEK 5-6: Alternative Training (Section 6)
  □ Forward-Forward for encoder layers
  □ Benchmark FF vs backprop on arithmetic
  □ If FF beats backprop → switch training mode
  □ If FF loses → keep as research artifact

WEEK 7-8: Architecture Upgrades (Section 9)
  □ MoE in encoder FFN
  □ Memory as structured graph
  
MONTH 3: Scale to Cloud (Section 5d)
  □ Lambda Labs / RunPod setup
  □ DeepSpeed ZeRO-3 integration
  □ Scale to 1B params
  □ Train on 5-10B tokens (multi-domain)
  
MONTH 4-6: AGI Features (Section 10)
  □ Self-improving loop
  □ Continuous learning pipeline
  □ Test-time compute scaling
  □ Lifelong multi-domain training
```

---

## 12. Training Commands (Current State)

```bash
# Generate synthetic arithmetic data
python -m cortex.train --generate-synthetic 50000

# Phase 1: Pretrain on synthetic arithmetic (30 epochs)
python -m cortex.train --pretrain

# Evaluate synthetic arithmetic accuracy
python evaluate_synthetic.py

# Phase 2: Finetune on GSM8K (15 epochs)
python -m cortex.train

# Evaluate GSM8K accuracy
python evaluate.py
```

---

## 13. Key Files Map

| File | Purpose | Lines |
|---|---|---|
| `cortex/model.py` | EidosV1 assembly, forward pass, generate | 191 |
| `cortex/modules/encoder.py` | Token + number embedding, 4 diff attn layers | 60 |
| `cortex/modules/diff_attention.py` | Diff Attention + RoPE + SwiGLU FFN | 150 |
| `cortex/modules/predictive_coder.py` | Embed + value prediction, surprise | 55 |
| `cortex/modules/neural_memory.py` | Differentiable Hebbian write+read | 97 |
| `cortex/modules/neuromodulator.py` | δ, ν, σ from surprise + complexity | 55 |
| `cortex/modules/ltc_cortex.py` | LTC recurrence with memory fusion | 76 |
| `cortex/modules/mtp_head.py` | 4-head prediction with alignment fix | 79 |
| `cortex/train.py` | Training loop, data loading, synthetic support | 448 |
| `cortex/tokenizer.py` | Restricted BPE vocab, answer masking | 83 |
| `cortex/synthetic_data.py` | Arithmetic problem generator (50K examples) | 220 |
| `evaluate.py` | GSM8K evaluation | 153 |
| `evaluate_synthetic.py` | Synthetic arithmetic evaluation | 210 |
| `cortex/RESEARCH.md` | Paper summaries | 102 |
| `cortex/research/` | 6 PDF papers | - |

---

## 14. Changelog

```
Changed:
  cortex/model.py              # Full rewiring: δ-gated memory, per-position num_head, value_pred loss
  cortex/modules/encoder.py    # num_proj std 0.005 -> 0.05
  cortex/modules/neural_memory.py  # momentum 0.95, write_gate support, NaN guard
  cortex/modules/predictive_coder.py  # +value_head, 4-tuple return
  cortex/modules/neuromodulator.py    # ν from complexity, softer σ blend
  cortex/modules/diff_attention.py    # Manual GQA (no enable_gqa), NaN-safe
  cortex/train.py               # Memory persistence, synthetic data, 90/10 split, pretrain/finetune save

Added:
  cortex/synthetic_data.py      # 4-level arithmetic problem generator
  evaluate_synthetic.py         # Synthetic arithmetic evaluation script

Changed config:
  - memory.reset() moved to training start (not per-epoch)
  - answer_values no longer passed to model (per-position num_head uses num_values[input_ids])
  - pretrain saves to eidos_pretrain_best.pt
  - finetune auto-loads eidos_pretrain_best.pt if exists
  - pretrain eval uses 90/10 synthetic split (not GSM8K)
  - generation shows full Q/A + training examples
```
