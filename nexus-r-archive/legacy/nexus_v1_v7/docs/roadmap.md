# NEXUS-R Roadmap

## Project Status: nexus_v1 (Validation Phase)

Current focus: Validate V7 architecture on powerful GPU, establish baselines.

---

## Version Roadmap

### nexus_v1: Foundation (Current)
**Status:** Validation pending

| Task | Priority | Status |
|------|----------|--------|
| Train on RTX 6000 Ada | HIGH | Pending |
| 10k+ steps training | HIGH | Pending |
| GSM8K benchmark | MEDIUM | Pending |
| MATH benchmark | MEDIUM | Pending |
| Establish perplexity baseline | HIGH | Done (5.5) |

**What's proven:**
- Architecture learns
- No memorization
- Perplexity improves

**What's pending:**
- Scale validation
- Reasoning task benchmarks

---

### nexus_v2: Working Memory
**Target:** Add persistent working memory

Components:
- [ ] Hash-based attention (O(1) retrieval)
- [ ] Working memory buffer
- [ ] Query-memory attention
- [ ] Fact storage/retrieval

**Why:** Current transformers have fixed context. Working memory lets the model hold and manipulate facts across long sequences.

---

### nexus_v3: Reasoning Engine
**Target:** Step-by-step reasoning, not just prediction

Components:
- [ ] Program synthesis module
- [ ] Chain-of-thought reasoning
- [ ] Differentiable neural programs
- [ ] Reasoning trace memory

**Why:** LLMs memorize solutions. Reasoning engines learn to solve novel problems.

---

### nexus_v4: World Model
**Target:** Internal physics/causality understanding

Components:
- [ ] Causal reasoning module
- [ ] Predictive world model
- [ ] Counterfactual reasoning
- [ ] Physics intuition

**Why:** Pattern matching ≠ understanding. World models learn how the world actually works.

---

### nexus_v5: Self-Improvement
**Target:** System that improves itself

Components:
- [ ] Meta-learning controller
- [ ] Self-diagnosis engine
- [ ] Architecture modification
- [ ] Continuous learning

**Why:** Fixed models are limited. Self-improving systems can exceed their initial capabilities.

---

## Benchmark Targets

| Task | Baseline (GPT-2 small) | Nexus_v1 Target | Nexus_v5 Target |
|------|------------------------|-----------------|-----------------|
| TinyShakespeare PPL | ~15 | ~5.5 (done) | ~3.0 |
| GSM8K (math) | ~15% | TBD | >50% |
| MATH | ~5% | TBD | >30% |
| ARC-C | ~30% | TBD | >60% |

---

## Research Milestones

### Milestone 1: Validate Learning
- [x] Architecture learns on real data
- [ ] 10k steps on GPU
- [ ] Perplexity < 4.0
- [ ] No memorization

**Evidence:** Train/val gap < 2x

### Milestone 2: Reasoning Basics
- [ ] GSM8K > 20%
- [ ] Chain-of-thought helps
- [ ] Can solve novel problems

**Evidence:** Benchmarks on unseen problems

### Milestone 3: Efficient Reasoning
- [ ] Beat 10x larger models on efficiency
- [ ] Better perplexity with fewer params
- [ ] Faster inference

**Evidence:** Speed + accuracy benchmarks

### Milestone 4: Self-Improvement
- [ ] System identifies own errors
- [ ] Modifies reasoning strategy
- [ ] Improves without human intervention

**Evidence:** Learning curves after self-modification

---

## Why Each Phase

### Why Working Memory First?
1. Transformers are fixed-context (O(N) attention)
2. Real reasoning requires holding facts
3. Sparse access = efficient retrieval
4. Foundation for all later reasoning

### Why Reasoning Before World Model?
1. Can test on formal tasks (math, logic)
2. Simpler to validate
3. Needed for world model feedback
4. Gradual complexity increase

### Why World Model Before Self-Improvement?
1. Need causal understanding for self-diagnosis
2. World model enables counterfactual reasoning
3. Can't improve what you don't understand
4. Builds on reasoning + memory foundation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU training fails | Debug on current hardware first |
| Reasoning doesn't improve | Fall back to perplexity optimization |
| Overfitting | Regularization + diverse data |
| Architecture too complex | Incremental validation |

---

## Success Criteria

### nexus_v1 Success
- [ ] Trained 10k+ steps on RTX 6000 Ada
- [ ] Perplexity < 4.0 on TinyShakespeare
- [ ] GSM8K baseline established

### nexus_v2 Success
- [ ] Working memory integrated
- [ ] Memory retrieval works
- [ ] Efficiency gain over no-memory baseline

### nexus_v3 Success
- [ ] Chain-of-thought module works
- [ ] GSM8K > 30%
- [ ] Reasoning traces improve accuracy

### nexus_v4 Success
- [ ] World model improves causal reasoning
- [ ] Counterfactual reasoning works
- [ ] Physics intuition validated

### nexus_v5 Success
- [ ] Self-diagnosis identifies errors
- [ ] Strategy modification works
- [ ] Continuous improvement confirmed

---

## Timeline

**Phase 1 (nexus_v1):** 1-2 weeks (validation)
**Phase 2 (nexus_v2):** 2-4 weeks (working memory)
**Phase 3 (nexus_v3):** 4-8 weeks (reasoning)
**Phase 4 (nexus_v4):** 8-12 weeks (world model)
**Phase 5 (nexus_v5+):** Ongoing (self-improvement)

Total: 3-6 months to full NEXUS-R architecture

---

## Open Questions

1. **How to validate reasoning?** - Need good benchmarks beyond perplexity
2. **World model architecture?** - Not established yet
3. **Self-improvement safely?** - Need safeguards
4. **When is "enough"?** - Define success criteria

---

## Contributing

1. Pick a component from roadmap
2. Implement in isolation
3. Validate against baseline
4. Document results
5. If improvement, merge. If not, try different approach.

---

Last updated: 2026-03-23
