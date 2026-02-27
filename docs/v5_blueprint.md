# RX.AI V5 — BLUEPRINT: SELF-MODIFYING PERSONAL AI

> **Status:** Planning  
> **Date:** 2025  
> **Research basis:** `docs/Arxiv_Research_Report_Top_200.md` (200 papers, 4 domains)  
> **Author:** Haziq + GitHub Copilot  

---

## 0. THE MISSION

> V5 must be **VERY POWERFUL, SELF LEARN, change its OWN CODE, LEARN USING TERMINAL, PERSONAL ASSISTANT**.

That maps to 6 concrete pillars:

| Pillar | One-Liner |
|--------|-----------|
| **A. Code-as-Data Engine** | Model reads, diffs, and rewrites its own Python source files |
| **B. Terminal Learning Loop** | Model executes code in a sandbox, learns from stdout/stderr |
| **C. Online DPO** | Unit-test results replace human labels as reward signal |
| **D. Mamba-SSM Core** | O(n) memory replaces attention — fits 3x more model on 8GB VRAM |
| **E. Riemannian ODE Backbone** | Non-Euclidean continuous dynamics for hierarchical code reasoning |
| **F. Personal Assistant Layer** | ChromaDB long-term memory + per-user LoRA + proactive suggestions |

---

## 1. WHAT CHANGES FROM V4

```
V4 ─── Fixed architecture ─── Trains on curated dataset ─── Answers questions
V5 ─── Self-modifying code ─── Trains on its own experience ─── Grows over time
```

V4 was "build a smart decoder on the RX 7600."  
V5 is "build a system that improves itself every day you use it."

---

## 2. PILLAR A — CODE-AS-DATA ENGINE (Self-Modifying)

### Concept
The model treats its own source files as an editable knowledge base. At any point it can:
1. **Read** a `.py` file into an AST
2. **Reason** about what to change (new feature, bug fix, optimization)  
3. **Generate** a patch as an AST diff  
4. **Apply** the patch via `ast.unparse()`
5. **Validate** by running unit tests  
6. **Commit** if tests pass — otherwise revert

### Architecture
```
User Request
    │
    ▼
┌───────────────────────────────────┐
│  CODE INTROSPECTION MODULE        │
│  - ast.parse() own source files   │
│  - Build call-graph (CFG)         │
│  - Store in ChromaDB as embeddings│
└─────────────┬─────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│  V5 DECODER (Pointer-Generator)   │
│  - Input:  user intent + AST ctx  │
│  - Output: AST patch tokens       │
│  - Validates syntax via grammar   │
└─────────────┬─────────────────────┘
              │
              ▼
┌───────────────────────────────────┐
│  PATCH SAFETY GATE                │
│  - ast.unparse() → .py file       │
│  - Run pytest in subprocess       │
│  - If pass → git commit           │
│  - If fail → revert + DPO penalty │
└───────────────────────────────────┘
```

### Key Files to Create
- `v5_core/architecture/code_introspector.py` — AST parser + CFG builder
- `v5_core/architecture/patch_applier.py` — diff, unparse, test runner
- `v5_core/training/self_modify_loop.py` — the main loop

### Research Basis
- **Paper: AI-Driven Code Refactoring with GNNs** — GNNs achieve 92% accuracy on refactoring, reduce cyclomatic complexity 35%, on 2M CodeSearchNet snippets. We adapt this for *self-refactoring*.
- **Paper: Self-Supervised Learning to Prove Program Equivalence** — transformer learns rewrite rules. We use this to validate that a patch preserves semantics.
- **Paper: ASTormer** — AST-structure-aware transformer decoder, directly applicable to our pointer-generator.

---

## 3. PILLAR B — TERMINAL LEARNING LOOP

### Concept
Every time the model runs code in a terminal, it converts the execution result into a training signal. This is the core of "learning using terminal."

```
┌─────────────────────────────────────────────────────────┐
│  TERMINAL AGENT LOOP (replaces static dataset training) │
│                                                         │
│  1. User gives a task (e.g., "fix this function")       │
│  2. Model generates code                                │
│  3. Code runs in subprocess sandbox                     │
│  4. stdout/stderr → structured observation              │
│  5. Observation → reward signal                         │
│  6. (code, reward) pair → DPO buffer                    │
│  7. Every N pairs → micro fine-tune LoRA head           │
└─────────────────────────────────────────────────────────┘
```

### Sandbox Design
- `subprocess.run(code, timeout=10, capture_output=True)` — safe isolation
- Parse exit code: 0 = pass, nonzero = fail
- Parse tracebacks with `traceback.extract_tb()` → structured error tokens
- Parse test results with `pytest --json-report` 

### Reward Function (no human labels)
```python
def terminal_reward(exit_code, stdout, stderr, expected_output=None):
    r = 0.0
    if exit_code == 0:
        r += 1.0                             # ran without crash
    if expected_output and expected_output in stdout:
        r += 2.0                             # matched expected output
    if "test passed" in stdout.lower():
        r += 3.0                             # pytest pass
    if "error" in stderr.lower():
        r -= 1.0                             # runtime error
    if "SyntaxError" in stderr:
        r -= 2.0                             # syntax error (severe)
    return r
```

### Online Learning Cadence
- Collect 64 (code, reward) pairs → 1 gradient step on LoRA head
- No full retraining; only the LoRA adapters update
- Base model frozen unless weekly full DPO run scheduled

### Research Basis
- **Paper: OpAgent** — online RL webagent trained via direct interaction (not static dataset). Proves offline datasets cause distributional shift. We apply this to code execution.
- **Paper: Reinforcement Inference** — entropy-aware inference-time control, model uses own uncertainty to decide when to retry. We use this in the terminal loop retry logic.
- **Paper: SPARC** — automated C unit test generation. Shows "leap-to-code" failure mode is fixed by planning before coding. Our terminal loop adds a planning phase.

---

## 4. PILLAR C — UNIT-TEST DPO (Self-Supervised Reward)

### Concept
Classic DPO needs human preference labels. We replace them entirely with programmatic test results:

```
Chosen response  = code that passes unit tests (reward > 0)
Rejected response = code that fails unit tests (reward ≤ 0)
```

This means the model can train 24/7 without any human annotation.

### DPO Buffer Structure
```python
{
    "prompt": "def sort_list(lst): ...",
    "chosen": "return sorted(lst)",           # test PASSED
    "rejected": "lst.sort(); return lst",     # test FAILED (modifies in place)
    "chosen_reward": 3.0,
    "rejected_reward": -1.0
}
```

### AutoDPO Loop (`training/auto_dpo_v5.py`)
```
Every hour:
  1. Pull 64 pairs from terminal buffer
  2. Filter: only pairs where chosen_reward > rejected_reward by margin > 1.0
  3. Run DPO gradient step (lr=1e-6, only LoRA params)
  4. Save adapter to snapshot_v5_lora_latest.pt
  5. Log loss to wandb/tensorboard
```

### Anti-Collapse Guard
- KL divergence from base model must stay < 0.3 per step
- If KL > 0.3, skip the gradient step and log warning
- Weekly: compare model on fixed benchmark (3 held-out tasks) to detect regression

### Research Basis
- **Paper: CoRefine** — 211k-param Conv1D controller atop frozen LLM achieves 190x compute reduction via targeted self-refinement. We adopt the lightweight controller concept.
- **Paper: IR³** — prevents reward hacking by reverse-engineering implicit objectives. We use this to audit whether terminal reward is being gamed.
- **Paper: Gradient Regularization prevents Reward Hacking in RLHF** — gradient-based constraint keeps policy in regions where reward is accurate.

---

## 5. PILLAR D — MAMBA-SSM CORE

### Why Replace Attention?
On 8GB VRAM (RX 7600), attention is `O(n²)` memory. With Mamba-2 (selective SSM), it's `O(n)`.  
This means for the same VRAM budget, we can run **3-4x longer context**.

### Architecture Change
```
V4:  Compressed → + Attention MHA → SSD-MoE experts → Decoder
V5:  Compressed → + Mamba-2 SSM  → MoE experts     → Decoder
                        ↑
              Selective gating: input-dependent A matrix
              Linear time, constant memory per step
```

### MODE Integration (from arxiv paper 17)
We combine **Mamba-2** with **Low-Rank Neural ODEs** for the reasoning core:
- Mamba handles long-range token dependencies (efficiency)
- Low-rank ODE handles continuous state updates between tokens (quality)
- Result: continuous-time Mamba — best of both worlds

```python
class V5MambaODECore(nn.Module):
    def __init__(self, d_model=1024, d_state=64, ode_rank=32):
        self.mamba = Mamba2Block(d_model, d_state)
        self.ode_proj = LowRankODE(d_model, rank=ode_rank)   # from MODE paper
    
    def forward(self, x):
        h_mamba = self.mamba(x)                  # O(n) sequence modeling
        h_ode   = self.ode_proj(h_mamba)         # continuous refinement
        return h_mamba + h_ode
```

### DirectML Note
Mamba2 uses selective scan which may have scatter issues on DirectML. Implementation plan:
1. Use chunked selective scan (similar to chunked_softmax fix in V4)
2. Replace `torch.cumsum` backward with manual prefix-sum via loop on DML
3. Test isolated before integrating into assembly

### Research Basis
- **Paper: 2Mamba2Furious** — simplifies Mamba-2 to essential components (A-mask + order increase). We use their ablation to pick only what matters.
- **Paper: Bayesian Optimality of In-Context Learning with SSMs** — proves SSMs implement Bayes-optimal prediction. Validates quality of replacing attention.
- **Paper: MODE** — Low-Rank NODEs + Mamba = best time-series model. Core inspiration for V5 Mamba-ODE fusion.

---

## 6. PILLAR E — RIEMANNIAN ODE BACKBONE

### Why Riemannian?
Standard ODEs evolve in Euclidean space (flat). Code has inherent hierarchical structure (class → method → statement → expression). Riemannian manifolds capture curved, hierarchical geometry.

From the arxiv report (paper 9, Domain 1):
> "Riemannian LTC networks evolve on non-Euclidean manifolds, capturing curved geometry — better generalization on hierarchical structures"

### Architecture
```
V4 ODE:  dx/dt = f(x, t)              ← Euclidean, flat
V5 ODE:  Dx/Dt = f_R(x, t)           ← Riemannian covariant derivative
         where f_R uses the metric tensor of the code AST manifold
```

In practice:
- Use **hyperboloid model** `H^n` for tree-structured data (ASTs are trees)
- Hyperbolic space naturally embeds trees with low distortion
- Replace `nn.Linear` with `HyperbolicLinear` in the ODE function

### Implementation Approach
Use `geoopt` library (PyTorch-compatible Riemannian optimization):
```python
import geoopt
manifold = geoopt.manifolds.PoincareBall(c=1.0)
# Replace Euclidean representations with Poincaré ball model
x_hyp = manifold.expmap0(x_euclidean)  # lift to hyperbolic space
h = riemannian_ode_solver(x_hyp)       # evolve on manifold
x_out = manifold.logmap0(h)             # project back
```

### Research Basis
- **Paper: Riemannian LTC** — LTC networks on Riemannian manifolds outperform Euclidean LTCs on hierarchical data (domain 1 of arxiv report).
- **Paper: Alzheimer's Manifold Mapping** — Manifold mapping for irregular longitudinal data, shows non-Euclidean space better captures structural progression.

---

## 7. PILLAR F — PERSONAL ASSISTANT LAYER

### Long-Term Memory
ChromaDB is already in V4. V5 adds structured memory types:

```python
class V5Memory:
    EPISODIC  = "episodic"   # What happened in past sessions
    SEMANTIC  = "semantic"   # Facts the model learned about you
    PROCEDURAL= "procedural" # How you like things done (coding style, etc.)
    EMOTIONAL = "emotional"  # Tone calibration (how formal/casual)
```

Each message gets embedded and stored tagged by type + timestamp.  
On each new message, the top-5 most relevant memories are retrieved and prepended to context.

### Per-User LoRA Profiles
```
user_haziq.lora → trained on your past conversations
user_default.lora → general assistant behavior
```
At inference: blend base model + user LoRA with interpolation weight α.  
At the end of each session: auto-fine-tune user LoRA on the session's accepted outputs.

### Proactive Suggestions
The model monitors:
- `git diff` on your repo every 30 min
- File modification times in watched directories
- Error logs from running processes

When it detects something interesting (new bug, uncommitted changes, etc.), it proactively surfaces a suggestion.

```python
# proactive_monitor.py
class ProactiveMonitor:
    def watch_git(self, repo_path):
        # Every 30 min: git diff HEAD
        # If diff contains new error patterns → trigger suggestion
    
    def watch_training(self, log_path):
        # Monitor training logs for loss spikes
        # If loss suddenly increases → alert user
```

### Metacognitive Framework (Think²)
From paper "Think²: Grounded Metacognitive Reasoning":
> "Ann Brown's regulatory cycle: Planning → Monitoring → Evaluation"

V5 implements this as a 3-phase inner loop before every response:
1. **Plan**: "What does the user need? What steps will I take?"
2. **Monitor**: "Am I staying on track? Any uncertainty?"
3. **Evaluate**: "Is my answer complete and correct? Should I try again?"

---

## 8. V5 FILE STRUCTURE

```
v5_core/
├── architecture/
│   ├── v5_assembly.py              # Main model: Mamba-ODE + MoE + Decoder
│   ├── mamba_ode_core.py           # Pillar D: Mamba-2 + Low-Rank ODE
│   ├── riemannian_ode.py           # Pillar E: Hyperbolic ODE solver
│   ├── code_introspector.py        # Pillar A: AST reader + CFG builder
│   ├── patch_applier.py            # Pillar A: diff + unparse + test runner
│   └── ast_decoder_v5.py           # Upgraded decoder with execution guidance
├── training/
│   ├── terminal_loop.py            # Pillar B: subprocess sandbox + reward
│   ├── auto_dpo_v5.py              # Pillar C: unit-test DPO
│   ├── self_modify_loop.py         # Pillar A: self-code-edit training
│   └── metacognitive_trainer.py    # Pillar F: Think² training
├── memory/
│   ├── v5_memory_manager.py        # Pillar F: typed ChromaDB memory
│   ├── user_lora_manager.py        # Pillar F: per-user LoRA
│   └── proactive_monitor.py        # Pillar F: git/file watcher
├── inference/
│   ├── v5_pipeline.py              # Full V5 inference with all pillars
│   ├── think2_controller.py        # Pillar F: Plan-Monitor-Evaluate loop
│   └── execution_guided_decoder.py # Pillar B: beam search + sandbox prune
└── evaluation/
    ├── self_benchmark.py            # KL-divergence collapse detector
    └── held_out_tasks.py            # 3 fixed tasks for regression testing
```

---

## 9. TRAINING PLAN (PHASES)

### Phase 1: Mamba Core (Week 1-2)
- Port V4 assembly to replace attention with Mamba-2
- Verify DirectML compatibility (chunked scan, no scatter)
- Train 1 epoch on existing instruction dataset
- Checkpoint: Mamba model beats V4 on NLL loss

### Phase 2: Terminal Loop (Week 3-4)
- Implement `terminal_loop.py` with subprocess sandbox
- Generate 1000 (code, terminal_result) pairs organically
- Run first unit-test DPO pass
- Checkpoint: Model improves code pass rate by >10%

### Phase 3: Self-Modification (Week 5-6)
- Implement `code_introspector.py` + `patch_applier.py`
- Test on simple cases first: docstring edits, variable renames
- Add safety gate (pytest must pass before commit)
- Checkpoint: Model can rename a function in its own codebase correctly

### Phase 4: Personal Assistant (Week 7-8)
- Integrate typed memory manager
- Build user LoRA training pipeline
- Implement proactive monitor
- Checkpoint: Model remembers your coding style across sessions

### Phase 5: Riemannian ODE (Week 9-10)
- Install `geoopt`, implement `riemannian_ode.py`
- Replace Euclidean ODE in assembly
- Ablation: Euclidean vs Riemannian on AST prediction tasks
- Checkpoint: Riemannian ODE >= Euclidean on code hierarchies

---

## 10. HARDWARE OPTIMIZATION (RX 7600)

| V5 Component | VRAM Cost (approx) | DirectML Notes |
|---|---|---|
| Mamba-2 core | ~1.5GB | Chunked scan, no `aten::cumsum` backward |
| LoRA adapters | ~200MB | Standard, no issues |
| ChromaDB index | ~300MB | CPU-side, no VRAM |
| Riemannian ODE | ~500MB | geoopt ops — verify DML compat |
| AST decoder V5 | ~1.2GB | Same scatter-free approach as V4 |
| **Total** | **~3.7GB** | Leaves 4.3GB headroom on RX 7600 |

With Mamba replacing attention, context length can increase from ~512 to ~2048 tokens at the same VRAM.

---

## 11. INNOVATION SUMMARY

What makes V5 genuinely novel:

1. **First self-modifying local AI** — runs on consumer GPU, edits its own source code, validates with tests
2. **Terminal-as-teacher** — no human labels ever; execution results are the curriculum
3. **Continuous-time Mamba** — fuses Mamba-2 efficiency with ODE quality (novel architecture)
4. **Riemannian code reasoning** — hyperbolic embeddings for AST hierarchies (direct arxiv research application)
5. **Typed episodic memory** — not just vector similarity but structured memory taxonomy
6. **Think² metacognition** — explicit Plan/Monitor/Evaluate before every response (not just "think step by step")

---

## 12. RISKS & MITIGATIONS

| Risk | Mitigation |
|---|---|
| Self-modification corrupts model | Safety gate: pytest must pass; git revert if not |
| DPO mode collapse | KL divergence monitor; weekly benchmark regression test |
| Mamba DirectML scatter issue | Chunked scan implementation (same approach as V4 chunked_softmax) |
| geoopt not DML-compatible | Fallback: use Euclidean approximation for non-Riemannian ops |
| Terminal sandbox escape | `subprocess.run` with `shell=False`, no `os.system`, timeout enforced |

---

## 13. START TODAY — NEXT 3 COMMANDS

1. **Finish V4 training first** (resume from batch 1170):
```powershell
.\venv\Scripts\python.exe v4_core/training/v4_instruction_trainer.py `
  --weights v4_core/snapshot_v4_instruct.pt `
  --resume v4_core/snapshot_v4_instruct_batch1170.pt `
  --dataset v4_core/data/v4_instruction_dataset.json `
  --output v4_core/snapshot_v4_instruct.pt `
  --epochs 3 --start_epoch 1 --batch_size 1 --lr 5e-5
```

2. **Install Mamba and geoopt** (V5 dependencies):
```powershell
.\venv\Scripts\pip.exe install mamba-ssm geoopt
```

3. **Generate V5 training data** from terminal interactions:
```powershell
.\venv\Scripts\python.exe v5_core/training/terminal_loop.py --collect 1000
```

---

*This document is the living architecture plan for RX.AI V5. Update as implementation progresses.*
