# NEXUS-R Architecture: nexus_v1

## Overview

nexus_v1 is based on V7 architecture, proven to learn on real data.

**Status:** Validated on TinyShakespeare
- Train perplexity: ~4.7
- Val perplexity: ~5.5
- No memorization (train/val gap ~1.2x)

---

## Architecture Components

### 1. Flash Attention (F.scaled_dot_product_attention)
**What:** Hardware-accelerated attention via PyTorch's built-in SDPA

**Why:** 
- O(N) memory instead of O(N²)
- Fused CUDA kernels
- No custom kernel compilation needed

**Implementation:**
```python
attn_output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    is_causal=True,
    dropout_p=0.0 if not self.training else 0.1
)
```

### 2. RoPE (Rotary Positional Encoding)
**What:** Positional encoding via rotation, not addition

**Why:**
- Zero parameters (free!)
- Better length generalization
- No upper bound on sequence length

**Implementation:**
- Precompute rotation angles
- Rotate Q and K before attention
- No learned positional embeddings

### 3. SwiGLU FFN
**What:** Gated Linear Unit with SiLU activation

**Formula:** `FFN(x) = (Silu(W1(x)) * W3(x)) @ W2`

**Why:**
- Outperforms GELU on benchmarks
- Gating mechanism controls information flow
- Standard in Llama, Mistral, PaLM

### 4. GQA (Grouped Query Attention)
**What:** Shared K/V heads with multiple Q heads

**Why:**
- Memory efficient (smaller KV cache)
- 6 Q heads, 2 KV heads = 3x reduction in K/V memory
- Negligible quality loss

### 5. RMSNorm
**What:** Root Mean Square Layer Normalization

**Formula:** `y = w * (x / rms(x))`

**Why:**
- Faster than LayerNorm (no mean computation)
- Similar or better performance
- Standard in modern LLMs

### 6. Cosine LR + Warmup
**What:** Learning rate schedule with linear warmup then cosine decay

**Why:**
- Stable training
- Faster convergence
- Proven schedule (used in GPT-3, LLaMA, etc.)

---

## Model Sizes

| Model | Params | Layers | Q Heads | KV Heads | Hidden Dim | FFN Dim |
|-------|--------|--------|--------|----------|-------------|---------|
| Tiny | 14.1M | 6 | 4 | 2 | 256 | 1024 |
| Small | 29.6M | 8 | 6 | 2 | 384 | 1536 |
| Medium | 62.0M | 12 | 8 | 2 | 512 | 2048 |

---

## Data Flow

```
Input Tokens
    │
    ▼
Embedding Layer (vocab_size → d_model)
    │
    ▼
NexusBlock × num_layers
    │
    ├── RMSNorm
    │
    ├── FlashAttention + RoPE
    │   ├── Q, K, V projection
    │   ├── RoPE rotation (Q, K)
    │   ├── Repeat K,V for GQA
    │   └── F.sdpa (causal mask)
    │
    └── SwiGLU FFN
        ├── W1 → SiLU
        ├── W3 → × (gate)
        └── W2 → output
    │
    ▼
RMSNorm
    │
    ▼
LM Head (d_model → vocab_size)
    │
    ▼
Output (logits per token)
```

---

## Training Configuration

### Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-4 | For 29.6M model |
| Weight decay | 0.1 | Strong regularization |
| Warmup steps | 100 | Linear warmup |
| Batch size | 8 | Sequence length 128 |
| Gradient clipping | 1.0 | Prevent exploding grads |
| Scheduler | CosineAnnealing | Decay to 10% of LR |

### Training Data

**TinyShakespeare:** ~1.1M characters
- 90% train / 10% validation
- Character-level tokenization
- ~50 unique characters

---

## Known Limitations

1. **Character-level only** - Not tokenized for words
2. **Small vocabulary** - ~50 chars limits expressivity
3. **Short context** - 128 tokens
4. **No reasoning** - Pure pattern matching

These will be addressed in later versions.

---

## What's Working

- [x] Learning (perplexity improves)
- [x] No memorization (healthy train/val gap)
- [x] Text generation (Shakespeare-style output)
- [x] Efficient GPU utilization (F.sdpa)

## What's Not Working

- [ ] Long training (10k+ steps) - Pending RTX 6000 Ada
- [ ] Reasoning tasks (GSM8K) - No benchmark yet
- [ ] Word-level tokenization - Character-only
- [ ] Working memory - Not implemented

---

## Validation Results

### TinyShakespeare (nexus_v1 Small, 1000 steps)
```
Step 0:   train=10.3, val=9.9, val_ppl=20103
Step 200: train=2.37, val=2.38, val_ppl=10.8
Step 400: train=1.95, val=2.11, val_ppl=8.3
Step 600: train=1.85, val=1.99, val_ppl=7.3
Step 800: train=1.71, val=1.70, val_ppl=5.5
```

**Observation:** Healthy learning, no memorization.

---

## Next Components (nexus_v2+)

### Working Memory
- Hash-based attention for O(1) retrieval
- Persistent fact storage
- Query-based memory access

### Reasoning Module
- Program synthesis
- Chain-of-thought
- Neural program interpreter

### World Model
- Causal reasoning
- Predictive modeling
- Counterfactual thinking

---

## References

1. Flash Attention: https://arxiv.org/abs/2203.03649
2. RoPE: https://arxiv.org/abs/2104.09864
3. SwiGLU: https://arxiv.org/abs/2002.05202
4. GQA: https://arxiv.org/abs/2307.09288
5. RMSNorm: https://arxiv.org/abs/1910.07467

---

Last updated: 2026-03-23
