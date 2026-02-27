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

SNAP-C1 is a **personal experiment** in building a neural architecture from scratch — not fine-tuning an existing model, but designing every component (embedding, attention, decoder, training loop) by hand to learn how they work and what breaks.

It has gone through 5 versions. V1-V4 were learning experiences with real problems (see [Honest History](#honest-history)). V5 is the current attempt, trying to build something that actually works.

> **Status:** Experimental. V4 can train on GPU. V5 architecture is designed, implementation in progress. No benchmark results yet. No claims of capability.

---

## What This Project Actually Is

- A from-scratch neural architecture (not a wrapper around GPT/Llama/Qwen)
- Built to run on an **AMD RX 7600 8GB** via DirectML (not CUDA)
- An experiment in whether a small model (~1.5B params) can learn to use tools and improve from its own experience
- A learning project — every version taught something about what doesn't work

## What This Project Is NOT

- Not a competitor to GPT-4, Claude, or any production AI
- Not a working coding assistant (yet)
- Not peer-reviewed research
- Not something you should rely on for anything

---

## Honest History <a name="honest-history"></a>

Every version had real problems. Documenting them honestly because hiding failures is how bad research happens.

| Version | What Was Tried | What Went Wrong |
|:-------:|----------------|-----------------|
| **V1** | LoRA fine-tuning on Qwen 3-4B | Trained on CPU in fp32. 99.97% of params frozen. LoRA can steer formatting but can't teach reasoning. The model ceiling was Qwen itself. |
| **V2** | From-scratch SSM compressor + 12-layer recurrent core + concept decoder | Pre-training used **random targets** (`torch.randint`). The RLFS reward signal was fake — `dummy_logits.mean() * reward` doesn't send real gradients. 69M-param HyperNetwork was defined but never called. Embedding (102M params) frozen on AMD. |
| **V3** | ODE solver (Euler method) + AST node decoder | Cut reasoning capacity 6x vs V2 (4 blocks vs 12). AST vocabulary limited to 50 variable names — can't spell real code. Adversarial verifier uses nn.GRU which crashes on DirectML. Weight transfer from V2 silently fails. |
| **V4** | Fused V2+V3 + pointer-generator + ChromaDB RAG + MoE routing | 65% of params are frozen embeddings. 256-token context window (SWE-bench needs 5,000+). Expert bank returns `torch.randn()` — no expert files exist. Beam search ignores copy distribution. Trained on 5,000 toy examples. |

**Pattern: Every version carried 40-83% dead weight — parameters that existed but never got gradients on AMD hardware, because `nn.Embedding` backward uses `scatter_add_` which DirectML rejects.**

Full technical autopsy in [docs/V5_ARCHITECTURE.md](docs/V5_ARCHITECTURE.md) Section 1.

---

## V5: Current Experiment — "Resonance"

V5 tries to fix everything that was broken in V1-V4. Whether it actually works is TBD.

### Core Ideas Being Tested

**1. Binary Embedding (replacing nn.Embedding)**
- Encode token IDs as 17-bit binary vectors, process through an MLP
- ~860K params instead of 102M, and **100% trainable** on DirectML (no scatter ops)
- Unproven: might lose information compared to a lookup table

**2. Resonance Blocks (replacing ODE solver + Transformer attention)**
- Dual-path: sliding window attention (local, O(n×128)) + FFT spectral mixing (global, O(n log n))
- Gated fusion decides which path matters for each token
- Unproven: spectral mixing for code is speculative. Might not capture syntax well.

**3. Elastic Hierarchical Context (replacing 256-token hard limit)**
- Multi-resolution: recent tokens full-res, older tokens compressed via avg_pool1d
- 8,192 effective tokens in ~1,856 slots
- Unproven: pooled representations might lose critical details

**4. Action Decoder (replacing text-completion paradigm)**
- Model outputs structured tool decisions (SEARCH, READ, EDIT, RUN) instead of generating text that gets parsed
- Unproven: whether 1.5B params can learn reliable tool selection is an open question

**5. Dual-Speed Learning (Fast Brain + Slow Brain)**
- Fast: ChromaDB stores every task trace instantly, retrieves similar past experiences
- Slow: Periodic LoRA fine-tuning on accumulated traces consolidates patterns into weights
- Inspired by hippocampus/neocortex complementary learning in neuroscience
- Unproven: the concept is sound in theory. Making it work in practice is the hard part.

**6. Self-Generated Curriculum**
- Model analyzes its own failure patterns and generates practice problems
- Unproven: might overfit to easy tasks or generate meaningless practice

**7. Federated Multi-User Learning**
- Multiple users share anonymized traces → everyone's model improves
- Unproven: not implemented yet. Privacy and data quality are unsolved.

### Architecture Diagram

```
User Request
    │
    ▼
Binary Embedding (token_id → bits → MLP → vector)
    │
    ▼
Elastic Context (8K tokens → 1856 multi-res slots)
    │
    ▼
Fast Brain retrieves similar past traces from ChromaDB
    │
    ▼
8× Resonance Blocks (windowed attention + FFT spectral mixing)
    │
    ├──► Action Decoder (which tool to call + arguments)
    │
    └──► Outcome Predictor (P(success) before executing)
    │
    ▼
Execute tool → Observe result → Loop or respond
    │
    ▼
Save trace → Periodically fine-tune (Slow Brain)
```

Full spec: [docs/V5_ARCHITECTURE.md](docs/V5_ARCHITECTURE.md)

---

## Project Structure

```
SNAP-C1/
├── v1_legacy/                  # V1: LoRA on Qwen (archived)
├── v2_core/                    # V2: SSM compressor + recurrent core (archived)
├── v3_core/                    # V3: ODE solver + AST decoder (archived)
├── v4_core/                    # V4: fused pipeline + pointer-gen (can train on GPU)
│   ├── architecture/
│   │   ├── v4_assembly.py
│   │   └── ast_decoder.py
│   ├── training/
│   │   └── v4_instruction_trainer.py
│   └── data/
│       └── v4_instruction_dataset.json
├── v5_core/                    # V5: current experiment (in progress)
│   ├── architecture/           # Binary embed, resonance blocks, elastic context, etc.
│   ├── agent/                  # Action loop, tool registry, memory, curriculum
│   └── training/               # Pre-training, self-evolution, federation
├── docs/
│   └── V5_ARCHITECTURE.md     # Full V5 design spec
└── evaluation/                 # Benchmarking (no results yet)
```

---

## Hardware

Developed and tested on:
- **GPU**: AMD Radeon RX 7600 8GB (via PyTorch DirectML)
- **RAM**: 16 GB
- **OS**: Windows 11

DirectML has limitations that shaped every design decision:
- `scatter_`, `scatter_add_` → banned (breaks nn.Embedding, F.one_hot, torch.gather backward)
- `aten::_thnn_fused_gru_cell` → banned (custom GRU cell implemented)
- `aten::sigmoid` → unstable on some tensor shapes (using tanh-based workaround)
- `torch.max(dim=).backward` → uses scatter (using .detach())

Cloud training planned via RunPod A100 for the 1.5B-param version.

---

## Setup

```bash
# Clone
git clone https://github.com/IRSPlays/SNAP-C1.git
cd SNAP-C1

# Create venv
python -m venv venv
venv\Scripts\activate  # Windows

# Install deps
pip install -r requirements.txt
```

---

## Current State (as of Feb 2026)

- [x] V4 trains on AMD RX 7600 via DirectML (all scatter_ issues resolved)
- [x] V4 ran ~1,170 batches of instruction tuning (loss ~1.7-3.0, normal for early training)
- [x] V5 architecture fully designed (22-section spec)
- [ ] V5 implementation (in progress)
- [ ] V5 pre-training on RunPod
- [ ] Any benchmark evaluation
- [ ] Any evidence that it actually works as an agent

---

## Lessons Learned

Things I learned the hard way building V1-V4:

1. **Dead parameters are worse than no parameters.** 100M frozen params eat VRAM and do nothing. V5 targets 100% trainable.
2. **Test the backward pass on your actual hardware.** Many PyTorch ops work forward on DirectML but crash on backward. Test every op before building on it.
3. **Random targets produce random weights.** Pre-training on `torch.randint` doesn't teach anything. Use real data.
4. **"It converges" doesn't mean "it's correct."** An ODE solver reaching equilibrium just means the state stopped changing. Doesn't mean it found the right answer.
5. **Context window matters more than model size.** A 237M-param model with 256-token context can't solve problems that require reading 5,000 tokens of code.
6. **Log everything.** The only way to know if training is actually learning is to track loss, gradient norms, and sample outputs over time.

---

## License

This is a personal research project. Use at your own risk. No warranty of any kind.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=120&section=footer" />
</div>
