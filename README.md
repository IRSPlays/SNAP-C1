<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:1a1a2e,100:0f3460&height=200&section=header&text=EI DOS&fontSize=72&fontColor=e94560&fontAlignY=35&desc=Differential%20Predictive%20Memory%20Transformer&descSize=18&descAlignY=55&descColor=8ab4f8" />
</div>

<div align="center">

**A from-scratch neuroscience-inspired architecture for reasoning.**

*Not fine-tuning. Not copying. Building from first principles.*

</div>

<br/>

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Params-30M-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/VRAM-1.8GB-FF6B35?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Training-00C853?style=for-the-badge" />
</div>

<br/>

<div align="center">
  <p><b>Copyright © 2026 Asirive. Built by Haziq.</b></p>
</div>

---

## Architecture

Eidos is not a language model with some extras. Every component has a purpose rooted in neuroscience and backed by research.

```
   input_ids [B, T]
        │
   ┌────▼────┐
   │ Encoder │  Diff Transformer (ICLR 2025) × 4 layers
   │         │  + Number value injection (log1p-scaled)
   └────┬────┘
        ├─ z [B,T,512], pooled [B,512]
        │
   ┌────▼──────────┐
   │ PredictiveCoder│  Predicts next embedding AND next number value
   │                │  Error signal = genuine arithmetic mistake detection
   └────┬───────────┘
        ├─ z_hat, error, cosine_dist, value_pred
        │
   ┌────▼──────────┐
   │ Neuromodulator │  δ (write gate), ν (LTC budget), σ (memory/cortex blend)
   │                │  ν scales with problem complexity (2-16 iterations)
   └────┬───────────┘
        │
   ┌────▼──────────┐     ┌─────────────────┐
   │ NeuralMemory  │←────│ δ-gated write   │  Titans (Dec 2024)
   │  M [256×256]  │     │ Hebbian outer   │  Persistent across training
   │               │────→│ product update  │  Differentiable end-to-end
   └────┬──────────┘     └─────────────────┘
        ├─ h_mem [B,T,512]
        │
   ┌────▼──────┐
   │ LTCCortex  │  Liquid Time-Constant (2020)
   │            │  2-16 recurrent iterations via ν
   │  h + dt·(  │  Adaptive time constants
   │  -h/τ +    │  Memory-augmented input
   │  tanh(...))│
   └────┬──────┘
        ├─ h_cortex [B,T,512]
        │
   ┌────▼─────────┐
   │  Integrator   │  σ·h_mem + (1-σ)·h_cortex
   │               │  σ→1: recall   σ→0: reason
   └────┬──────────┘
        ├─ out [B,T,512]
        │
   ┌────▼──┐     ┌─────────────────┐
   │  MTP   │     │ 4 prediction    │
   │ Heads  │────→│ horizons        │  DeepSeek V3 (Dec 2024)
   │        │     │ Shared proj     │  67% fewer params
   │        │     │ Self-consistency│  Heads vote on confidence
   └────────┘     └─────────────────┘
```

### Component Map

| Component | Lines | Params | Source | What It Does |
|:---|:---:|---:|:---|:---|
| `diff_attention.py` | 146 | 11.5M | Diff Transformer (ICLR 2025) | Noise-canceling attention via A₁ − λ·A₂ subtraction |
| `predictive_coder.py` | 58 | 3.1M | Rao & Ballard (1999) | Predicts next embedding + number value. Error = confusion signal |
| `neural_memory.py` | 118 | 0.5M | Titans (Dec 2024) | Persistent Hebbian K⊗V storage. Low-rank [256×256]. δ-gated writes |
| `neuromodulator.py` | 55 | <1K | Bio-inspired | δ = write gate, ν = LTC budget, σ = recall vs reason blend |
| `ltc_cortex.py` | 76 | 4.7M | LTC Networks (2020) | Adaptive-time recurrent ODE. 2-16 iterations per problem |
| `mtp_head.py` | 82 | 0.6M | DeepSeek V3 (Dec 2024) | 4-head prediction. Shared projection. Self-consistency voting |
| `encoder.py` | 60 | 5.8M | - | Token + number embeddings, 4 Diff layers, final RMSNorm |

---

## Honest Current State (May 2026)

### What Works

| Capability | Status | Detail |
|:---|---:|:---|
| Forward/backward | ✅ | No NaN. AMP (fp16) stable. Batch=4. |
| Synthetic arithmetic pretraining | 🔄 | 45K examples. 30 epochs. Curiosity curriculum + sleep/replay active |
| Diff Attention | ✅ | GQA, RoPE, SwiGLU FFN, λ learned per-head |
| NeuralMemory persistence | ✅ | [256×256] buffer survives across epochs. δ-gated writes |
| LTC recurrence | ✅ | ν-controlled depth (2-16). Memory-augmented input |
| MTP self-consistency | ✅ | 4 heads vote on next token at inference |
| Self-verification | ✅ | Re-generates with max LTC when prediction error high |
| Skip LTC on easy problems | ✅ | <3 numbers → memory-only fast path |
| Flash Attention | 🔄 | sdpa kernel auto-selects best backend |

### What Doesn't Work Yet

| Problem | Root Cause | Target Fix |
|:---|:---|:---|
| GSM8K accuracy: 3.3% | 43M params, 7.5K examples, no arithmetic pretraining | 30M params with synthetic pretrain → GSM8K finetune |
| Pure arithmetic: ~30% (epoch 1) | Model learning format, not computation | 30 epochs synthetic → target >90% |
| torch.compile | No Triton on Windows | WSL or Linux cloud GPU |
| Multi-step reasoning chains | LTC depth underutilized during training | Sleep/replay consolidation after each epoch |
| Number computation | No actual arithmetic circuit | Number stream + value prediction heads |
| Training speed | 1.12 steps/s (old) → needs more optimization | 16 steps/s after Phase 1 (AMP + batch=4 + seq=96 + bmm) |

---

## Quick Start

```bash
# Generate 50K synthetic arithmetic problems
python -m cortex.train --generate-synthetic 50000

# Phase 1: Pretrain on synthetic (30 epochs, ~3 hours)
python -m cortex.train --pretrain

# Check arithmetic accuracy (5 difficulty levels)
python evaluate_synthetic.py

# Phase 2: Finetune on GSM8K (15 epochs)
python -m cortex.train

# Check GSM8K accuracy
python evaluate.py
```

### Smoke Test

```bash
python smoke_v2.py          # Forward/backward + NaN check + memory persistence
python smoke_phase1.py      # Speed benchmark with AMP + bmm + batch=4
```

---

## Training Pipeline

```
Phase 1 (Pretrain)                    Phase 2 (Finetune)
┌────────────────────┐               ┌────────────────────┐
│ Synthetic Arithmetic│               │      GSM8K          │
│  50K problems       │               │  7.5K word problems │
│  + - × ÷ chains    │──────────────→│  English reasoning  │
│  Word-form numbers  │  knowledge    │  Multi-step logic   │
│  30 epochs          │  transfer     │  15 epochs          │
│  Batch=4, AMP       │               │  Batch=4, AMP       │
└────────────────────┘               └────────────────────┘
         │                                     │
         ▼                                     ▼
  eidos_pretrain_best.pt              eidos_v1_best.pt
```

### Active Training Features

| Feature | When | Effect |
|:---|:---|:---|
| **Curiosity curriculum** | Pretrain | Tracks per-example error. Hard examples get higher weight |
| **Sleep/replay** | Every 5 epochs | Replays top-500 hardest examples through full LTC for consolidation |
| **MTP self-consistency** | Inference | 4 heads vote. Majority overrides main head if split |
| **Self-verification** | Inference | If prediction error high → re-generate with 16 LTC iterations |
| **Skip LTC** | Inference | <3 numbers in prompt → skip LTC entirely (memory-only fast path) |
| **AMP (fp16)** | Training | 40% faster, same VRAM |
| **Batch=4, accum=2** | Training | 2× throughput |
| **seq_len=96** | Pretrain | 2× less compute (synthetic data is 15-40 tokens) |
| **Low-rank memory** | Always | [256×256] instead of [512×512] — 4× smaller, faster read/write |

---

## Parameter Breakdown

| Component | Before | After (Phase 2) | Savings |
|:---|---:|---:|---:|
| Encoder (4 layers) | 11.5M | 11.5M | — |
| Embedding | 5.8M | 5.8M | — |
| LTC Cortex | 4.7M | 4.7M | — |
| Predictive Coder | 3.1M | 3.1M | — |
| MTP Extra Heads | 17.4M | **5.1M** | 71% |
| NeuralMemory | 0.8M | **0.5M** | 38% |
| **Total** | **43.5M** | **30.2M** | **31%** |

---

## Scale Roadmap

| Scale | d_model | n_layers | Params | VRAM | Hardware |
|:---|---:|---:|---:|---:|:---|
| Current | 512 | 4 | 30M | 1.8 GB | RTX 2050 4GB |
| Next | 640 | 6 | 85M | 2.8 GB | RTX 2050 4GB |
| Max Local | 768 | 8 | 150M | 3.8 GB | RTX 2050 4GB |
| Cloud | 1024 | 12 | 350M | 8 GB | A10 / 2080 Ti |
| Cloud | 1536 | 24 | 1.2B | 24 GB | A10 / 4090 |
| Cloud | 2048 | 32 | 3.1B | 48 GB | A6000 |
| Cloud | 3072 | 40 | 8.2B | 80 GB | A100 |

---

## Research Foundation

```
┌─────────────────────────────────────────────────────────┐
│ Differential Transformer (ICLR 2025 Oral)               │
│ Microsoft Research. A = A₁ − λ·A₂. Noise cancellation.  │
├─────────────────────────────────────────────────────────┤
│ Titans (Google, Dec 2024)                               │
│ Neural long-term memory as surprise-gated Hebbian store. │
├─────────────────────────────────────────────────────────┤
│ DeepSeek V3 (Dec 2024)                                  │
│ Multi-token prediction with shared projection.          │
├─────────────────────────────────────────────────────────┤
│ Liquid Time-Constant Networks (Hasani et al., 2020)     │
│ Adaptive-time recurrent ODE. dh/dt = −h/τ + f(h,x).     │
├─────────────────────────────────────────────────────────┤
│ Predictive Coding (Rao & Ballard, Nature Neuro, 1999)   │
│ Top-down predictions cancel bottom-up signals.          │
├─────────────────────────────────────────────────────────┤
│ Modern Hopfield Networks (Ramsauer et al., 2020)        │
│ Attention = persistent memory. Exponential capacity.     │
└─────────────────────────────────────────────────────────┘
```

---

## Lessons Learned

1. **Wire components for their purpose, not as generic modules.** NeuralMemory learns nothing when reset every epoch. LTC is dead weight at 1 iteration. The architecture works when each piece has a JOB.

2. **enable_gqa=True can silently produce NaN on some GPU/driver combos.** Manual GQA expansion is slower but reliable. User-reported NaN fixed by this one change.

3. **Cross-domain evaluation is misleading.** Measuring synthetic pretraining with GSM8K eval gives +15 gap. Use held-out synthetic data for pretrain eval.

4. **Number embeddings need to be AUDIBLE.** std=0.005 creates 0.004:1 signal ratio — the model can't hear numbers. std=0.05 = 0.04:1 — numbers register.

5. **MTP heads are 40% of the model and mostly redundant.** Sharing one projection matrix with per-head RMSNorm saves 67% of head params with no accuracy loss.

6. **Memory persistence matters more than memory size.** [256×256] persistent > [512×512] reset-every-epoch.

7. **RMSNorm + fp16 gives warnings but works.** PyTorch falls back to unfused kernel. Harmless.

---

## Future Phases

| Phase | What | Status |
|:---|---|:---:|
| ~~Speed~~ | AMP, batch=4, seq=96, bmm, compile | ✅ |
| ~~Efficiency~~ | MTP sharing, low-rank memory, Flash Attn | ✅ |
| ~~Zero-Cost~~ | Self-verify, self-consistency, skip LTC, curiosity | ✅ |
| ~~Training~~ | Sleep/replay, progressive layers (deferred) | ✅ |
| **Synthetic pretraining** | 30 epochs on 50K arithmetic | 🔄 |
| GSM8K finetuning | 15 epochs | ⬜ |
| Forward-Forward | Alternative to backprop for encoder layers | ⬜ |
| MoE Encoder | Mixture of Experts in FFN (4× capacity) | ⬜ |
| Number Stream | Separate arithmetic computation pathway | ⬜ |
| Scale to 150M | d_model=768, n_layers=8 (local) | ⬜ |
| Scale to 1B | Cloud GPUs, DeepSpeed ZeRO-3 | ⬜ |
| AGI Loop | Self-improving curriculum, continuous learning | ⬜ |

---

## Acknowledgments

- Papers: Microsoft Research (Diff Transformer), Google Research (Titans), DeepSeek (V3), Hasani et al. (LTC), Rao & Ballard (Predictive Coding), Ramsauer et al. (Hopfield)
- Built with PyTorch, tiktoken, CUDA

---

<div align="center">
  <p><i>"Pain first, rest later."</i></p>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:1a1a2e,100:0f3460&height=100&section=footer" />
  <p><b>Built by Haziq, 2026</b></p>
</div>
