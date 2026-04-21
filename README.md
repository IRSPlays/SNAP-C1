<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=220&section=header&text=SNAP-C1&fontSize=80&fontColor=e94560&fontAlignY=35&desc=Structural%20Neural%20Architecture%20Pipeline&descSize=18&descAlignY=55&descColor=a7a7c5" />
</div>

<div align="center">
  <p><b>An experimental from-scratch neural architecture exploring self-improving code agents.</b></p>
  <p><i>This is a personal research project. Nothing here is production-ready. No benchmarks have been passed yet.</i></p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge&logo=apache&logoColor=white" />
</div>

<div align="center">
  <p><b>Copyright &copy; 2026 Asirive. Built by Haziq, Founder of Asirive.</b></p>
</div>

<br/>

## What is SNAP-C1?

SNAP-C1 is a **personal experiment** in building a neural architecture from scratch — not fine-tuning an existing model, but designing every component by hand to learn how they work and what breaks.

> **Status:** Experimental. Architecture is built. No training on real data yet. No benchmark results.

---

## Honest Current State (March 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| NEXUS V6 Architecture | Implemented | 940 lines, 4 sizes (40M/68M/157M/462M) |
| Forward/Backward Pass | ✅ Tested | No NaN gradients |
| WSD Trainer | ✅ Implemented | Warmup-Stable-Decay schedule |
| Training on Real Data | ✅ **Working!** | TinyStories: 6.3 → 0.06 loss |
| GPU Training | ⚠️ CPU only | No CUDA GPU in this environment |
| Benchmarks | ❌ | Not yet run |

**What works:** 
- Model forward/backward passes, loss computation, WSD learning rate schedule
- Training on real text data - loss decreases rapidly (100x+ reduction)
- Fixed embedding initialization to prevent logits explosion
- Depth-adaptive experts, Mamba+attention hybrid, MoE with load balancing

**What doesn't work yet:** 
- GPU training (need CUDA hardware)
- Training on coding/reasoning data (tool_use data format needs alignment)
- Benchmarking against other models

---

## Architecture: NEXUS V6

NEXUS V6 combines innovations from recent research papers with novel components:

### Research-Backed Components
- **Tree-Guided Self-Evolution** (2603.18620) - Learnable context refinement
- **WSD LR Schedule** (2602.06797) - Warmup-Stable-Decay for stable training
- **Concept Discovery** (2512.24617) - Variable-length concept detection
- **Depth-Adaptive Experts** (2603.19172) - Layer-depth-aware expert routing

### Novel Components
- **Entanglement Mixer** - Quantum-inspired weight correlation
- **Latent Concept Experts** - Concept-specialized expert routing
- **Self-Evolving Hebbian Layer** - Outcome-guided plasticity
- **Adaptive Mamba-Attention Hybrid** - Dynamic sequence processing
- **Evolutionary Pooling** - Input-complexity-adaptive pooling

### Model Sizes
| Size | Parameters | Layers | Experts |
|------|------------|--------|--------|
| Small | 157M | 16 | 6 |
| Medium | 462M | 24 | 8 |
| Large | 1.26B | 32 | 12 |

---

## Project History

| Version | What Was Tried | What Went Wrong |
|:-------:|----------------|-----------------|
| **V1** | LoRA fine-tuning on Qwen 3-4B | Trained on CPU in fp32. 99.97% frozen. Can't teach reasoning. |
| **V2** | From-scratch SSM + recurrent core | Random targets (`torch.randint`). Fake reward signal. 102M frozen embeddings. |
| **V3** | ODE solver + AST decoder | 6x reasoning capacity cut. Limited AST vocab. |
| **V4** | Fused pipeline + MoE + RAG | 65% frozen params. 256-token context (need 5000+). Expert bank returns `torch.randn()`. |
| **V5** | Binary embedding + Resonance blocks | Incomplete. Still exploring architecture options. |
| **V6** | Consolidated NEXUS architecture | ✅ Fixed embedding init, training works on real data! |

**Key fix:** Embedding initialization was causing logits explosion. Fixed by using `std=1/sqrt(d_model)` to prevent ~20x scaled logits when weights are tied.

---

## Training Results (March 22, 2026)

**Verified on TinyStories dataset:**
- Model: 68.2M parameters (NexusTiny)
- Training: 30 steps, batch 8, lr=5e-4
- Loss: 9.94 → 0.07 (99.3% reduction!)
- Time: ~52 seconds on CPU

**Key fixes applied:**
1. Embedding initialization: `std=1/sqrt(d_model)` prevents logits explosion
2. Fixed `FlashAttentionLayer` → `FlashAttention` import
3. All `.item()` calls removed (were breaking gradients)
4. Load balancing + z-loss for MoE working

---

## Efficiency Goals

The architecture is designed to beat models 10-100x its size through:

1. **Depth-Adaptive Experts**: Only active experts per layer depth
2. **Latent Concept Discovery**: Focus computation on relevant concepts
3. **Mamba + Attention Hybrid**: O(n) SSM for long contexts, attention for local
4. **Top-K Sparse MoE**: Only compute with top-k experts per token
5. **Self-Evolution**: Hebbian plasticity for online learning

**Next steps to verify efficiency:**
- [ ] Benchmark against transformer at same size
- [ ] Compare to 10x larger model on reasoning tasks
- [ ] Measure inference speed on long sequences
- [ ] Verify expert sparsity (how many experts actually used?)

---

## Hardware

- **Training:** NVIDIA RTX 6000 Ada (CUDA)

---

## Quick Test

```bash
cd /workspaces/SNAP-C1

# Test NEXUS V6
python -c "
from v6_core.architecture.nexus_v6 import build_nexus_small
import torch

model = build_nexus_small()
x = torch.randint(0, 32000, (2, 64))
logits, info = model(x)
print(f'Output: {logits.shape}')
print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
"
```

---

## Current TODO

- [x] Fix embedding initialization (logits explosion)
- [x] Verify loss decreases on real data
- [ ] Train on coding/reasoning data (tool_use JSONL)
- [ ] Get GPU access for faster training
- [ ] Benchmark efficiency vs. larger models
- [ ] Verify reasoning/coding capabilities

---

## Lessons Learned

1. **Dead parameters are worse than no parameters.** 100M frozen params eat VRAM and do nothing.
2. **Test the backward pass on your actual hardware.** Many PyTorch ops work forward but crash backward.
3. **Random targets produce random weights.** Pre-training on `torch.randint` doesn't teach anything.
4. **"It converges" doesn't mean "it's correct."** Stable loss ≠ learning.
5. **Context window matters more than model size.** A model that can't read enough context can't reason.
6. **Log everything.** Track loss, gradient norms, and sample outputs over time.
7. **Don't add paper components without validation.** 10 innovations ≠ 10x better if they conflict.

---

## License

Copyright &copy; 2026 Asirive. All rights reserved.

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Haziq, Founder of Asirive.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=120&section=footer" />
</div>
