<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=220&section=header&text=SNAP-C1&fontSize=80&fontColor=e94560&fontAlignY=35&desc=Structural%20Neural%20Architecture%20Pipeline&descSize=18&descAlignY=55&descColor=a7a7c5" />
</div>

<div align="center">
  <p><b>An experimental neural architecture for code understanding through continuous-time differential equations, SSD-streamed Mixture-of-Experts, and AST-based structural decoding.</b></p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/DirectML-AMD_RX-ED1C24?style=for-the-badge&logo=amd&logoColor=white" />
</div>

<br/>

## What is SNAP-C1?

SNAP-C1 is a **research project** exploring whether code reasoning can be approached structurally rather than sequentially. Instead of predicting the next token like traditional language models, SNAP-C1 processes code through a pipeline of specialized neural components that operate on mathematical representations of program structure.

> **Status:** Active research & development. Currently training on open-source Python codebases (Django, Flask, scikit-learn) and evaluating against SWE-Bench Verified.

---

## Architecture Overview

The system has evolved through 4 major versions, each building on the last:

```mermaid
graph LR
    classDef v2 fill:#1a1a2e,stroke:#e94560,stroke-width:2px,color:#eee
    classDef v3 fill:#16213e,stroke:#0f3460,stroke-width:2px,color:#eee
    classDef v4 fill:#0f3460,stroke:#53354a,stroke-width:2px,color:#eee
    classDef io fill:#533549,stroke:#e94560,stroke-width:1px,color:#eee

    A["📝 Code Prompt"]:::io --> B
    B["V2: Holographic<br/>Compressor"]:::v2 --> C
    C["V4: ChromaDB<br/>RAG Retrieval"]:::v4 --> D
    D["V4: SSD MoE<br/>Expert Router"]:::v4 --> E
    E["V3: ODE Liquid<br/>Time Core"]:::v3 --> F
    F["V4: AST + BPE<br/>Decoder"]:::v4 --> G["📄 Output"]:::io
```

### Component Breakdown

| Stage | Component | Version | What It Does |
|:-----:|-----------|:-------:|-------------|
| 1 | **Holographic Compressor** | V2 | Compresses input into a dense 1024-D state vector |
| 2 | **ChromaDB RAG Engine** | V4 | Queries a vector database of indexed Python repositories to find relevant context |
| 3 | **SSD Micro-Expert Router** | V4 | Softmax router that selects which expert weight shards to stream from disk into VRAM |
| 4 | **Continuous Recurrent Core** | V3 | Processes the context vector through an ODE (Ordinary Differential Equation) solver that iterates until the hidden state converges to equilibrium |
| 5 | **AST Pointer-Generator** | V4 | Decodes the equilibrium vector into Abstract Syntax Tree node predictions with a hybrid BPE copy mechanism |

---

## Key Design Decisions

### 🔄 ODE-Based Reasoning (V3 Core)
Instead of fixed-depth feedforward layers, the recurrent core uses **Liquid Time-Constant (LTC) neurons** governed by differential equations. The solver runs until the hidden state stabilizes — meaning harder problems naturally get more compute cycles. This is inspired by [Neural ODE](https://arxiv.org/abs/1806.07366) and [Liquid Time-Constant Networks](https://arxiv.org/abs/2006.04439).

### 💾 SSD-Streamed Experts (V4 Router)
To keep VRAM usage low, expert weight shards are stored on disk as `.safetensors` files and streamed into GPU memory on-demand via a softmax routing policy. Only the top-K experts needed for a given input are loaded.

### 🌳 Structural Output (V4 Decoder)
The model predicts AST node types and branching structure rather than raw text tokens. A pointer-generator mechanism allows copying BPE tokens from the retrieved context, giving the model access to an effectively unbounded vocabulary.

### 🔀 Cross-Hardware Portability
A centralized device resolver (`v4_core/utils/device.py`) automatically selects CUDA (NVIDIA), DirectML (AMD), or CPU — making the same codebase runnable on both cloud GPUs and consumer hardware.

---

## Project Structure

```
SNAP-C1/
├── v2_core/                    # Holographic compression (Mamba-style SSM)
│   └── architecture/
│       └── holographic_compressor.py
├── v3_core/                    # ODE recurrent core + training
│   ├── architecture/
│   │   └── recurrent_core.py   # Liquid Time-Constant ODE solver
│   ├── data/                   # AST graph parser + dataset generation
│   └── training/               # V3 standalone trainer
├── v4_core/                    # Full pipeline assembly
│   ├── architecture/
│   │   ├── v4_assembly.py      # Master pipeline (connects all stages)
│   │   └── ast_decoder.py      # Pointer-generator AST decoder
│   ├── memory/
│   │   ├── ssd_router.py       # Micro-expert SSD streaming router
│   │   └── chroma_indexer.py   # ChromaDB vector retrieval
│   ├── data/                   # BPE tokenizer + dataset builder
│   ├── training/
│   │   └── v4_ddp_trainer.py   # Parallelized trainer (AMP, DataLoader, compile)
│   └── evaluation/
│       ├── v4_inference.py     # Capability testing
│       └── v4_swe_bench.py     # SWE-Bench Verified benchmark
├── runpod_setup.sh             # Quick-start cloud training script
└── runpod_expanded_training.sh # Full pipeline: clone repos → train → benchmark
```

---

## Quick Start

### Prerequisites
```bash
pip install torch tiktoken safetensors loguru chromadb numpy tqdm
```

### Generate Training Data
Extract structural logic from any Python codebase:
```bash
python v4_core/data/v4_general_dataset_builder.py --target_dir /path/to/repo --output my_dataset.json
```

### Train
```bash
# Local (AMD RX 7600 / NVIDIA GPU)
python v4_core/training/v4_ddp_trainer.py --epochs 100 --batch_size 16

# With expanded dataset + deeper ODE convergence
python v4_core/training/v4_ddp_trainer.py \
  --epochs 15 --batch_size 16 --workers 4 \
  --dataset v4_core/data/v4_expanded_dataset.json \
  --max_loops 100
```

### Evaluate
```bash
# Capability test (held-out prompts)
python v4_core/evaluation/v4_inference.py --weights v4_core/snapshot_v4_hyper_router.pt

# SWE-Bench Verified solvability benchmark
python v4_core/evaluation/v4_swe_bench.py --weights v4_core/snapshot_v4_hyper_router.pt --max_instances 50
```

### Cloud Training (RunPod)
One-shot script that clones Django/Flask/scikit-learn/requests, generates 43k+ training chunks, trains, and benchmarks:
```bash
chmod +x runpod_expanded_training.sh
./runpod_expanded_training.sh
```

---

## Training Optimizations

The trainer implements several parallelization strategies for NVIDIA GPUs:

- **True Batched Inference** — all chunks processed as a single `[B, 1, 1024]` tensor (11.8x speedup over sequential)
- **AMP Mixed Precision** — FP16 matmuls with FP32 accumulation via `torch.amp`
- **`torch.compile`** — JIT kernel fusion on the ODE core and compressor submodules
- **Multi-worker DataLoader** — async prefetching with pinned memory

---

## Version History

| Version | Focus | Key Innovation |
|:-------:|-------|---------------|
| **V1** | Prototype | Basic RLHF with subprocess-based code execution |
| **V2** | Compression | Holographic state compression (SSM-based) |
| **V3** | Reasoning | Continuous-time ODE solver replacing fixed-depth layers |
| **V4** | Scale | RAG retrieval + SSD expert streaming + AST decoding + batched GPU training |

---

## References

- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) — Chen et al., 2018
- [Liquid Time-Constant Networks](https://arxiv.org/abs/2006.04439) — Hasani et al., 2020
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023
- [SWE-Bench](https://www.swebench.com/) — Jimenez et al., 2024

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=120&section=footer" />
</div>
