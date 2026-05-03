# Cortex V1 — Research Foundations

## 1. Modern Hopfield Networks (Ramsauer et al., 2020)
**arXiv: 2008.02217** — "Hopfield Networks is All You Need"

### Core Math
Energy function: E(ξ) = -lse(β, X^Tξ) + ½||ξ||²

Update rule (one-step retrieval):
    ξ_new = X · softmax(β · X^T · ξ)

Where:
- ξ ∈ R^d = query state
- X = [x_1, ..., x_N] ∈ R^{d×N} = stored patterns
- β = inverse temperature (sharpness)

### Key Insight
This is IDENTICAL to transformer attention: softmax(QK^T)V
But with PERSISTENT storage: K,V live in memory, not recomputed each pass.

### Capacity
Exponential: C · d^{d/(d-2)} patterns for d-dimensional state.
At d=256: effectively unlimited.

### Write Operation (Hebbian)
M ← M + δ · k ⊗ v^T  (outer product)

### Read Operation
v = V · softmax(β · K^T · q)

---

## 2. Liquid Time-Constant Networks (Hasani et al., 2020)
**arXiv: 2006.04439** — "Liquid Time-constant Networks"

### Core Math
dh/dt = -h / τ(h, x) + f(h, x)

Where time constant τ is LEARNED:
    τ(h, x) = σ(W_τ·h + U_τ·x + b_τ)

### Discretized (Euler step, for GPU)
h_{t+1} = h_t + Δt · (-h_t/τ_t + f(h_t, x_t))

### Properties
- Novel input → τ small → fast reaction
- Familiar input → τ large → stable, energy-efficient
- Bounded states (proven in paper)
- Superior expressivity vs standard RNNs

---

## 3. Complementary Learning Systems (McClelland et al., 1995)
**Psychological Review, 102(3)** — "Why there are complementary learning systems..."

### Dual-System Theory
- Hippocampus: Fast, sparse, pattern-separated. One-shot learning.
- Neocortex: Slow, distributed, overlapping. Gradual generalization.

### Consolidation
Sleep replay: hippocampus → neocortex transfer.
Interleaved replay prevents catastrophic forgetting.

---

## 4. Predictive Coding (Rao & Ballard, 1999)
**Nature Neuroscience 2** — "Predictive coding in the visual cortex"

### Core Concept
Top-down predictions cancel bottom-up signals.
Only prediction ERRORS propagate.

### In Our System
z_hat_{t+1} = f_predict(z_t)       (top-down)
ε_t = z_{t+1} - z_hat_{t+1}       (error)
||ε|| = 1 - cos(z_true, z_pred)    (cosine distance)

---

## 5. Architecture Design

### CortexV1 Flow
```
Tokens → Encoder(attention+FFN) → z (embeddings)
  ├→ PredictiveCoder: predicts z_{t+1}, outputs ε
  ├→ Neuromodulator: [δ, ν, σ, α] = f(||ε||, memory_confidence)
  ├→ Hippocampus: Write if δ > θ, Read via query
  ├→ Cortex (LTC-RNN): dh/dt = -h/τ + f(h,x), iterations via ν
  └→ Integrator: out = σ·h_mem + (1-σ)·h_cortex
      → Decoder (weight-tied LM head) → logits
```

### Neuromodulator Signals (Deterministic V1)
- δ (dopamine): write gate — clamp((||ε||-μ)/σ, 0, 1)
- ν (norepinephrine): LTC iterations — 1 + round(||ε|| * 4)
- σ (serotonin): memory trust — sigmoid(memory_match_score)
- α (acetylcholine): Hebbian update rate — 0 (unused V1)

### Training Phases
1. Encoder + LTC + Decoder: Standard LM training
2. Activate Hippocampus + Neuromodulator: Error-driven writes
3. End-to-end fine-tune: Low LR, all components
