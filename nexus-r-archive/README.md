# NEXUS-R: Reasoning-Capable AGI Architecture

## Project Overview

NEXUS-R is a next-generation AI architecture designed to move beyond pattern matching toward genuine reasoning, planning, and self-improvement.

### The Core Problem with LLMs

LLMs only predict the next token. They don't:
- Reason through problems step-by-step
- Understand causality or physics
- Plan multi-step solutions
- Evaluate their own reasoning
- Learn continuously

### NEXUS-R Vision

Build systems that:
1. **Reason** - Not just predict, but think through problems
2. **Plan** - Decompose complex tasks into subgoals
3. **Self-Evaluate** - Recognize and correct their own mistakes
4. **Learn Continuously** - Never stop improving

---

## Version History

### Legacy (v1-v7)

These versions represent the evolution of NEXUS from scratch:

| Version | Description | Status |
|---------|-------------|--------|
| v1 | LoRA fine-tuning on Qwen | Failed - frozen params |
| v2 | From-scratch SSM | Failed - random targets |
| v3 | ODE solver + AST decoder | Failed - crashes |
| v4 | MoE + RAG | Failed - dead code |
| v5 | Binary embedding | Failed - incomplete |
| v6 | Consolidated NEXUS | Failed - learning collapse |
| v7 | Simplified transformer | **Working** |

### NEXUS-V1 (Current)

Based on V7's proven architecture with modern optimizations:

**Architecture:**
- Flash Attention (F.scaled_dot_product_attention)
- RoPE positional encoding
- SwiGLU activation
- Grouped Query Attention (GQA)
- RMSNorm
- Cosine LR scheduler with warmup

**Model Sizes:**
| Model | Params | KV Heads |
|-------|--------|----------|
| Tiny | 14.1M | 2 |
| Small | 29.6M | 2 |
| Medium | 62.0M | 2 |

**Status:** Validated on TinyShakespeare (real data)
- Train perplexity: ~4.7
- Val perplexity: ~5.5
- No memorization

---

## Roadmap

### Phase 1: Validation (nexus_v1)
- Validate V7 architecture on GPU
- Benchmark on standard tasks
- Confirm learning works

### Phase 2: Working Memory (nexus_v2)
- Add persistent working memory
- Hash-based O(1) attention
- Query-relevant facts

### Phase 3: Reasoning Engine (nexus_v3)
- Program synthesis module
- Chain-of-thought reasoning
- Differentiable neural programs

### Phase 4: World Model (nexus_v4)
- Causal reasoning
- Predictive modeling
- Physics intuition

### Phase 5: Self-Improvement (nexus_v5+)
- Meta-learning
- Self-modification
- Continuous learning

---

## Why This Could Beat 10-100x Larger Models

| Traditional LLM | NEXUS-R |
|------------------|---------|
| Predicts tokens | Reasons through problems |
| Statistical patterns | Causal understanding |
| Single forward pass | Multi-step planning |
| Fixed reasoning | Learns to reason |
| Passive learning | Active experimentation |
| No self-awareness | Self-evaluating |

---

## Getting Started

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (for training)

### Training
```bash
cd nexus-r/nexus_v1
python training/train.py --model small --steps 10000
```

### Evaluation
```bash
python evaluation/benchmark.py --tasks tiny_shakespeare,gsm8k
```

---

## Research Questions

1. Can we learn reasoning strategies from data?
2. Does world model improve causal reasoning?
3. Can systems self-diagnose failures?
4. Can architecture modify itself?

---

## Project Structure

```
nexus-r/
├── legacy/           # v1-v7 (old versions)
│   ├── v1_legacy/
│   ├── v2_core/
│   ├── v3_core/
│   ├── v4_core/
│   ├── v5_core/
│   ├── v6_core/
│   └── v7_core/
│
└── nexus_v1/        # Reasoning-capable version
    ├── architecture/ # Model code
    ├── training/     # Training scripts
    ├── evaluation/    # Benchmarks
    └── docs/         # Documentation
```

---

## Contributing

This is a research project. See docs/ for architecture specifications.

---

## License

TBD - Research use only.
