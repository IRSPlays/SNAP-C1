# Nexus-R V1: Comprehensive Debug Report
**Generated:** 2026-04-21  
**Scope:** Architecture, training pipeline, memory/performance, design decisions, integration  
**Status:** Ready for handoff to fix agent

---

## Executive Summary

Nexus-R V1 has **5 critical bugs**, **9 high-severity issues**, and **12+ medium/low issues**. The architecture is functional (trains on TinyShakespeare without NaN), but several design decisions are theoretically unsound, memory-inefficient, or dead code. The most severe problems: dynamic halting is fake, progression loss is a no-op, the model never touches GPU, and the dual-stream MLA innovation lacks empirical validation.

**Priority fix order:**
1. Fix halting or remove it (don't lie about early stopping)
2. Kill prog_loss (pure compute waste)
3. Add `.to(device)` everywhere
4. Ablate dual-stream MLA vs. standard self-attention
5. Replace char tokenizer with BPE before any reasoning benchmark

---

## CRITICAL ISSUES

### 1. Dynamic Halting is Fake — Never Actually Stops
- **File:** `recursive_block.py:174-184`
- **Severity:** CRITICAL
- **Problem:** `halt_threshold` and `max_halt_steps` are computed and stored but the H-cycle loop never breaks early. The model always burns full `H_cycles * L_cycles * L_layers` compute. The `converged_early` flag is purely decorative.
- **Fix:** Add a real break condition:
  ```python
  if prev_thought is not None and sim.item() > (1.0 - self.halt_threshold):
      break
  if h_step >= self.max_halt_steps:
      break
  ```
  Or be honest: remove `halt_threshold` and rename `H_cycles` to `num_reasoning_steps`.

### 2. Progression Loss is a No-Op (Zero Gradient)
- **File:** `nexus_r.py:240-250`
- **Severity:** CRITICAL
- **Problem:** Both tensors in `prog_loss` are `.detach()`ed:
  ```python
  intermediates[j].detach().flatten(1),
  intermediates[j-1].detach().flatten(1),
  ```
  This makes `prog_loss` a constant scalar. It adds to the loss value but contributes **nothing** to backprop. Pure compute waste.
- **Fix:** **Delete prog_loss entirely.** `RecursiveReasoner` already computes `diversity_loss` with correct gradient flow (only `prev_thought` detached). If you want the `1-cos` formulation, replace `diversity_loss` inside `RecursiveReasoner`, but don't duplicate.

### 3. Model Never Touches GPU
- **File:** `train_v1.py:141-184`
- **Severity:** CRITICAL
- **Problem:** No `.to(device)` anywhere. Training runs on CPU (~50-100x slower). For a 30M param model, this makes the training pipeline unusable.
- **Fix:**
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  input_ids, labels = input_ids.to(device), labels.to(device)
  ```

### 4. `intermediates` List Pins Full Computation Graph
- **File:** `recursive_block.py:171, 187`
- **Severity:** CRITICAL
- **Problem:** Stores every H-cycle `thought` tensor with full autograd graph. For `build_nexus_small` (batch=8, seq=512, d=512): ~12MB raw data per batch, but the real cost is O(HxL) activations kept alive. Prevents garbage collection during forward pass.
- **Fix:** Don't return `intermediates` by default. If intermediate supervision is needed, compute loss inside `RecursiveReasoner` and discard tensors. Remove `'intermediates': intermediates` from `info`.

### 5. Embedding Noise Destroys Signal
- **File:** `nexus_r.py:199-201`
- **Severity:** CRITICAL
- **Problem:** Embeddings init with `std=0.02`, but noise is `std=0.1`. SNR = 0.2. The model receives mostly Gaussian garbage at every forward pass. Combined with the training loop never annealing it, this alone can prevent convergence.
- **Fix:** Either remove embedding noise entirely, or set `noise_scale <= 0.01` and anneal to 0 in the training loop.

---

## HIGH SEVERITY ISSUES

### 6. No LR Warmup
- **File:** `train_v1.py:162-165`
- **Severity:** HIGH
- **Problem:** `CosineAnnealingLR` starts at full 3e-4 LR. For recursively-unrolled architecture, cold-starting at peak LR causes early instability.
- **Fix:** Add linear warmup:
  ```python
  total_steps = n_epochs * len(loader)
  warmup_steps = 100
  def lr_lambda(step):
      if step < warmup_steps: return step / warmup_steps
      return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
  scheduler = LambdaLR(optimizer, lr_lambda)
  ```

### 7. Weight Decay Applied to Norm & Embedding Params
- **File:** `train_v1.py:162`
- **Severity:** HIGH
- **Problem:** `AdamW(weight_decay=0.01)` decays **all** parameters. Standard practice excludes biases, normalization weights, and embeddings.
- **Fix:**
  ```python
  decay, no_decay = [], []
  for n, p in model.named_parameters():
      if not p.requires_grad: continue
      if 'norm' in n or 'bias' in n or 'embed' in n:
          no_decay.append(p)
      else:
          decay.append(p)
  optimizer = torch.optim.AdamW([
      {'params': decay, 'weight_decay': 0.01},
      {'params': no_decay, 'weight_decay': 0.0}
  ], lr=3e-4)
  ```

### 8. No TRM Gradient Checkpointing / `no_grad` Trick
- **File:** `recursive_block.py:148-150`
- **Severity:** HIGH
- **Problem:** Every H-cycle stores activations for backward. Effective depth = 36 passes for `build_nexus_small`. Activation memory ~1.3GB just for the reasoner. The comment says "VRAM to spare" — this is wrong for scaling.
- **Fix:** Implement Samsung TRM pattern:
  ```python
  with torch.no_grad():
      for h_step in range(self.H_cycles - 1):
          thought = self._run_L_cycle(...)
  thought = thought.detach()
  # Final H-step WITH gradients
  for _l in range(self.L_cycles):
      thought = self._run_L_cycle(...)
  ```
  This makes memory O(1) in H-cycles instead of O(H).

### 9. `repeat_interleave` Wastes Memory for GQA
- **File:** `dual_stream_mla.py:93-98`, `layers.py:165-168`
- **Severity:** HIGH
- **Problem:** `repeat_interleave(rep, dim=1)` allocates new contiguous tensors, expanding K,V memory by `n_heads / n_kv_heads` (e.g., 2x). `F.scaled_dot_product_attention` natively supports GQA via broadcasting since PyTorch 2.0.
- **Fix:** **Delete the GQA expand blocks entirely.** Pass Q as `[B, n_heads, T, head_dim]` and K,V as `[B, n_kv_heads, T, head_dim]` directly into SDPA.

### 10. `generate()` Only Supports Batch Size 1
- **File:** `nexus_r.py:298-313`
- **Severity:** HIGH
- **Problem:** `logits[0, token_id]` and `next_token.item()` assume batch_size=1. Will crash for batch_size > 1.
- **Fix:** Vectorize over batch dimension or add `assert input_ids.shape[0] == 1`.

### 11. Generation Tensor on CPU
- **File:** `nexus_r.py:238-239`
- **Severity:** HIGH
- **Problem:** `torch.tensor([prompt_ids], dtype=torch.long)` creates tensor on CPU. If model is on GPU, `generate()` crashes with device mismatch.
- **Fix:**
  ```python
  device = next(self.parameters()).device
  prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
  ```

### 12. No EOS Stopping in Generation
- **File:** `nexus_r.py:281-317`
- **Severity:** HIGH
- **Problem:** Generation always runs for exactly `max_new_tokens`. Doesn't stop early on EOS, wasting compute and degrading output quality.
- **Fix:**
  ```python
  if eos_token_id is not None and next_token.item() == eos_token_id:
      break
  ```

### 13. Orphaned `nexus-r/` at Workspace Root
- **File:** `nexus-r/nexus_v1/` (workspace root)
- **Severity:** HIGH
- **Problem:** The workspace root `nexus-r/` has no `architecture/` directory. All active code is in `SNAP-C1/nexus-r/`. Running commands from workspace root fails.
- **Fix:** Decide on project root. Either move `SNAP-C1/nexus-r/` to workspace root, or always operate from `SNAP-C1/`.

### 14. `finetune_reasoning.py` is Completely Broken
- **File:** `nexus-r/nexus_v1/training/finetune_reasoning.py`
- **Severity:** HIGH
- **Problem:** Imports non-existent `NexusV7`. Uses hardcoded Windows paths. Completely incompatible with current `NexusR` architecture.
- **Fix:** Rewrite or delete.

---

## MEDIUM SEVERITY ISSUES

### 15. Noise Scale & Repulsion Tau Never Annealed
- **File:** `train_v1.py:175-213`
- **Severity:** MEDIUM
- **Problem:** `nexus_r.py` reads `_current_noise_scale` and `recursive_block.py` reads `_repulsion_tau`, but the training loop never sets them. They remain at defaults (0.1 and 0.5) forever.
- **Fix:** Anneal each epoch:
  ```python
  progress = epoch / max(n_epochs - 1, 1)
  model._current_noise_scale = 0.1 * (1.0 - progress)
  model.reasoner._repulsion_tau.fill_(0.5 - 0.3 * progress)  # 0.5 -> 0.2
  ```

### 16. Trailing Tokens Discarded by Chunking
- **File:** `train_v1.py:91`
- **Severity:** MEDIUM
- **Problem:** `range(0, len(all_ids) - seq_len, seq_len // 2)` discards tokens after the last valid start index.
- **Fix:** Pad final partial chunk or use `collate_fn` with dynamic padding.

### 17. Only Final Checkpoint Saved (No Best Model)
- **File:** `train_v1.py:248-259`
- **Severity:** MEDIUM
- **Problem:** No validation set or best-loss tracking. Saved model might be overfit. Missing optimizer/scheduler state (can't resume training).
- **Fix:** Save best checkpoint based on validation loss and include optimizer state.

### 18. `BlockAttnRes` Defined But Never Used
- **File:** `layers.py:181-206`
- **Severity:** MEDIUM
- **Problem:** Imported in `__init__.py` but no model or training script uses it. Dead code.
- **Fix:** Remove from codebase and `__init__.py` until validated.

### 19. Unused Imports
- **File:** `dual_stream_mla.py:23-26`, `recursive_block.py:28`, `nexus_r.py:24`
- **Severity:** MEDIUM
- **Problem:** `RMSNorm`, `rms_norm`, `apply_rotary_pos_emb`, `RotaryEmbedding` imported but never used.
- **Fix:** Remove unused imports.

### 20. Magic Numbers Should Be Config-Driven
- **File:** `nexus_r.py:253`, `recursive_block.py:179`, `nexus_r.py:200`
- **Severity:** MEDIUM
- **Problem:** `aux_coeff=0.1`, `step_bias_scale=0.1`, `noise_scale=0.1` are hardcoded deep in model code.
- **Fix:** Add to `NexusConfig` dataclass.

### 21. Scalar Tensor Construction in Hot Loop
- **File:** `recursive_block.py:176-179`
- **Severity:** MEDIUM
- **Problem:** `torch.tensor(h_step, device=input_embeddings.device)` creates a new tensor each H-cycle iteration, causing CPU-GPU sync overhead.
- **Fix:** Use direct indexing: `self.step_embeds.weight[h_step]`.

### 22. `eval_suite_runner.py` Hardcodes Paths
- **File:** `eval_suite_runner.py:165-166`
- **Severity:** MEDIUM
- **Problem:** Paths hardcoded. Can't evaluate different checkpoints without editing source.
- **Fix:** Use `argparse` for CLI arguments.

### 23. `train_v1.py` Data Path Not Validated
- **File:** `train_v1.py:116-120`
- **Severity:** MEDIUM
- **Problem:** Looks for `data/team_thinking/train.jsonl` with no existence check. Crashes with unhelpful `FileNotFoundError`.
- **Fix:** Add check:
  ```python
  if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data not found at {data_path}")
  ```

### 24. `eval_suite_runner.py` Uses `strict=False`
- **File:** `eval_suite_runner.py:21`
- **Severity:** MEDIUM
- **Problem:** `load_state_dict(..., strict=False)` silently ignores missing/extra keys. Mismatches between training and eval go unnoticed.
- **Fix:** Use `strict=True` unless there's a specific reason.

---

## DESIGN DECISION EVALUATION

### 25. Dual-Stream MLA: Unvalidated Innovation
- **Verdict:** ABLATE BEFORE KEEPING
- **Severity:** HIGH
- **Analysis:** Frozen K,V + evolving Q is mathematically encoder-decoder cross-attention. Standard causal self-attention with residuals already preserves input information. No empirical evidence this beats self-attention.
- **Fix:** Run head-to-head ablation: standard self-attention vs. dual-stream on same data. If dual-stream doesn't win by >5% perplexity, remove it.

### 26. H-Cycle/L-Cycle Nesting: Redundant
- **Verdict:** SIMPLIFY
- **Severity:** MEDIUM
- **Analysis:** L-cycles apply the same weight-tied stack repeatedly (`f^4(x)`). H-cycles inject step embedding and repeat. Mathematically equivalent to deeper single loop. No basis in literature for "two timescales."
- **Fix:** Set `L_cycles=1`. Let H-cycles be the only loop. Increase `L_layers` if more depth per step is needed.

### 27. Step Embeddings: Breaks Weight Tying Concept
- **Verdict:** REPLACE
- **Severity:** MEDIUM
- **Analysis:** `nn.Embedding(H_cycles, d_model)` hard-caps model to exactly `H_cycles` steps. Can't generalize to fewer/more steps at inference. Breaks the core weight-tying premise.
- **Fix:** Use sinusoidal embeddings (generalize to arbitrary depth) or small MLP taking `step_idx / max_steps`.

### 28. Diversity Loss / Repulsion: Anti-Convergence
- **Verdict:** REMOVE
- **Severity:** HIGH
- **Analysis:** `F.relu(sim - tau)` actively punishes convergence. In iterative refinement, stability is a feature, not a bug. Forces model to keep changing its mind even after finding the answer. No theoretical foundation.
- **Fix:** Delete diversity_loss entirely. Trust CE loss. If collapse is a real problem, use attention entropy regularization instead.

### 29. Character-Level Tokenizer: Wrong for Reasoning
- **Verdict:** REPLACE
- **Severity:** HIGH
- **Analysis:** Char-level is fine for smoke tests (Shakespeare) but destroys semantic locality for reasoning. Numbers split into digits, words fragment, effective context shrinks 4-5x. Can't validate reasoning architecture on char-level data.
- **Fix:** Use GPT-2 BPE or `tiktoken` before any GSM8K/MATH benchmarking. Keep char-level only for "does it run" tests.

### 30. No Attention Mask / Padding Handling
- **Verdict:** FIX
- **Severity:** HIGH
- **Analysis:** All sequences assumed same length. No padding mask support. `ignore_index=-100` in CE loss is dead code because dataset never produces `-100`.
- **Fix:** Add `attention_mask` parameter to `DualStreamMLA` and `Attention`. Pass to SDPA. Update `TextDataset` to pad and mask.

### 31. Weight Tying (lm_head = embed): Correct
- **Verdict:** KEEP
- **Severity:** N/A
- **Analysis:** Well-justified. Reduces parameters by ~25% in small models. Standard practice (Phi-1, TinyLlama).

---

## INTEGRATION / PROJECT STRUCTURE ISSUES

### 32. AGENTS.md Documents Non-Existent Files
- **File:** `SNAP-C1/nexus-r/AGENTS.md`, `SNAP-C1/AGENTS.md`
- **Severity:** HIGH
- **Problem:** Root AGENTS.md references `v6_core` (stale). Nexus-R AGENTS.md references `nexus_v1.py`, `flash_attention.py`, `train.py` — none exist.
- **Fix:** Rewrite both AGENTS.md files to match actual file structure.

### 33. Missing Training Subpackage Exports
- **File:** `nexus-r/nexus_v1/training/__init__.py` (empty)
- **Severity:** MEDIUM
- **Problem:** Can't do `from nexus_v1.training import train`.
- **Fix:** Add explicit exports.

### 34. Testing Gaps
- **Severity:** MEDIUM
- **Problem:** Only end-to-end smoke tests exist. No unit tests for `DualStreamMLA`, `RecursiveBlock`, `AnchorEncoder`, tokenizer, or masking.
- **Fix:** Add component-level tests:
  ```python
  def test_dual_stream_mla():
      mla = DualStreamMLA(d_model=256, n_heads=8, n_kv_heads=4)
      thought = torch.randn(2, 16, 256)
      k = v = torch.randn(2, 4, 16, 32)
      out = mla(thought, k, v)
      assert out.shape == (2, 16, 256)
  ```

---

## RECOMMENDED FIX ORDER

### Phase 1: Critical (Do First)
1. Add `.to(device)` in `train_v1.py`
2. Fix or remove fake halting in `recursive_block.py`
3. Delete `prog_loss` from `nexus_r.py`
4. Remove `intermediates` from `RecursiveReasoner` info dict
5. Fix embedding noise (remove or anneal to 0)

### Phase 2: High Priority
6. Add LR warmup to `train_v1.py`
7. Fix weight decay (exclude norm/bias/embed)
8. Implement TRM `no_grad` trick for memory
9. Remove `repeat_interleave` GQA expansion (use native SDPA)
10. Fix `generate()` for batch_size > 1 and device mismatch
11. Add EOS stopping to `generate()`
12. Ablate dual-stream MLA vs. standard self-attention

### Phase 3: Medium Priority
13. Anneal noise_scale and repulsion_tau
14. Replace char tokenizer with BPE
15. Add attention mask / padding support
16. Fix AGENTS.md documentation
17. Remove dead code (`BlockAttnRes`, unused imports)
18. Add `argparse` to eval scripts

### Phase 4: Design Evaluation
19. Remove diversity_loss / repulsion
20. Simplify H/L nesting to single loop
21. Replace step embeddings with sinusoidal
22. Add component-level unit tests

---

## FILES TO MODIFY

| File | Issues | Priority |
|------|--------|----------|
| `nexus-r/nexus_v1/architecture/recursive_block.py` | 1, 4, 8, 15, 21, 27, 28 | CRITICAL |
| `nexus-r/nexus_v1/architecture/nexus_r.py` | 2, 5, 10, 11, 12, 20, 25 | CRITICAL |
| `nexus-r/nexus_v1/training/train_v1.py` | 3, 6, 7, 13, 15, 16, 17, 23 | CRITICAL |
| `nexus-r/nexus_v1/architecture/dual_stream_mla.py` | 9, 19, 25 | HIGH |
| `nexus-r/nexus_v1/architecture/layers.py` | 9, 18, 19 | MEDIUM |
| `nexus-r/nexus_v1/training/eval_suite_runner.py` | 22, 24 | MEDIUM |
| `nexus-r/nexus_v1/training/finetune_reasoning.py` | 14 | HIGH |
| `AGENTS.md` (root + nexus-r) | 32 | MEDIUM |
| `nexus-r/nexus_v1/training/__init__.py` | 33 | LOW |

---

## ARCHITECTURE COMPARISON: Nexus-R vs. OpenMythos

| Feature | OpenMythos | Nexus-R V1 | Verdict |
|---------|-----------|------------|---------|
| Recurrence mechanism | Full hidden state: `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` | Attention-specific: only Q evolves, K,V frozen | Nexus-R is cheaper per step |
| Input preservation | Re-injects encoded input via learned A,B matrices | Implicit: frozen anchor K,V | Nexus-R avoids unstable A,B learning |
| Stability | LTI constraint, spectral radius ρ(A) < 1 | RMSNorm + residuals + cosine halting | Nexus-R simpler, less formal |
| MoE | Yes (sparse routed + shared) | No (SwiGLU only) | Add MoE for scale |
| Loop differentiation | Hypothesizes depth-wise RoPE | H_cycles + step embeddings | Both underdeveloped |
| Memory per loop | Full transformer block | Only Q generation | **Nexus-R wins** |
| Parameter efficiency | Same params, more loops | Same params, more loops | Tie |
| Inference scaling | More loops = deeper reasoning | More loops = deeper reasoning | Same theory |

**Bottom line:** Nexus-R's asymmetric recursive design is more parameter-efficient than OpenMythos. But both are unproven on reasoning benchmarks. The ideas validate each other.
