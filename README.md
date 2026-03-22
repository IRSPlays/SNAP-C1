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
  <img src="https://img.shields.io/badge/DirectML-AMD_RX-ED1C24?style=for-the-badge&logo=amd&logoColor=white" />
</div>

<br/>

## What is SNAP-C1?

SNAP-C1 is a **personal experiment** in building a neural architecture from scratch — not fine-tuning an existing model, but designing every component by hand to learn how they work and what breaks.

> **Status:** Experimental. Architecture is built. No training on real data yet. No benchmark results.

---

## Honest Current State (March 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| NEXUS V6 Architecture | Implemented | 1117 lines, 3 sizes (157M/462M/1.26B) |
| Forward/Backward Pass | Tested ✓ | No NaN on synthetic data |
| WSD Trainer | Implemented | Warmup-Stable-Decay schedule |
| RTX 6000 Access | ❌ | SSH connection not established yet |
| Training on Real Data | ❌ | Not started |
| Benchmarks | ❌ | None run |

**What works:** Model forward/backward passes, loss computation, WSD learning rate schedule.

**What doesn't work yet:** Actually training on real coding/reasoning data, connecting to GPU server.

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
| **V3** | ODE solver + AST decoder | 6x reasoning capacity cut. Limited AST vocab. GRU crashes on DirectML. |
| **V4** | Fused pipeline + MoE + RAG | 65% frozen params. 256-token context (need 5000+). Expert bank returns `torch.randn()`. |
| **V5** | Binary embedding + Resonance blocks | Incomplete. Still exploring architecture options. |
| **V6** | Consolidated NEXUS architecture | Untested on real data. SSH to GPU not working. |

**Pattern:** Every version carried 40-83% dead weight due to DirectML limitations and design mistakes.

---

## Hardware

- **Local Development:** AMD RX 7600 8GB (DirectML)
- **Planned Training:** NVIDIA RTX 6000 Ada (access pending)

DirectML limitations that shaped architecture:
- `scatter_`, `scatter_add_` → banned (breaks nn.Embedding backward)
- `aten::_thnn_fused_gru_cell` → banned (custom GRU implemented)
- `torch.max(dim=).backward` → uses scatter (workaround implemented)

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

- [ ] Establish SSH to RTX 6000
- [ ] Set up training environment on GPU
- [ ] Find/download coding/reasoning dataset
- [ ] Run first training step on real data
- [ ] Verify loss decreases (not just NaN-free)
- [ ] Benchmark against baseline transformer

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

Personal research project. Use at your own risk. No warranty.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=120&section=footer" />
</div>
