# SNAP-C1 V5: The Living Model

## Codename: RESONANCE

### "The model that gets smarter every day you use it."

---

## Table of Contents

1. [Why V5 Exists — V1-V4 Autopsy Summary](#1-why-v5-exists)
2. [The Core Insight — Dual-Speed Learning](#2-the-core-insight)
3. [Architecture Overview](#3-architecture-overview)
4. [Novel Component 1: Binary Embedding (scatter-free)](#4-binary-embedding)
5. [Novel Component 2: Resonance Blocks (spectral + local attention)](#5-resonance-blocks)
6. [Novel Component 3: Elastic Hierarchical Context (8192 tokens)](#6-elastic-context)
7. [Novel Component 4: Observation Encoder](#7-observation-encoder)
8. [Novel Component 5: Action Decoder (not text completion)](#8-action-decoder)
9. [Novel Component 6: Outcome Predictor](#9-outcome-predictor)
10. [Novel Component 7: Fast Brain — Episodic Memory](#10-fast-brain)
11. [Novel Component 8: Slow Brain — Weight Consolidation](#11-slow-brain)
12. [Agent Action Loop](#12-agent-loop)
13. [Tool Registry](#13-tool-registry)
14. [Experience Buffer & Trace Format](#14-experience-buffer)
15. [Self-Generated Curriculum](#15-curriculum)
16. [Self-Modification via Code Introspector](#16-self-modification)
17. [Federated Multi-User Learning](#17-federation)
18. [Training Pipeline](#18-training)
19. [Parameter Budget & VRAM Analysis](#19-params)
20. [File Map](#20-file-map)
21. [What's Kept vs. Deleted from V1-V4](#21-kept-vs-deleted)
22. [DirectML Compatibility Guarantees](#22-directml)

---

## 1. Why V5 Exists — V1-V4 Autopsy Summary <a name="1-why-v5-exists"></a>

### V1 (LoRA Wrapper): 99.97% parameters frozen, model ceiling = Qwen 3-4B
- LoRA adapters could steer formatting but couldn't teach new reasoning
- No GPU training at all — everything on CPU in fp32

### V2 (Fractal Recurrent Core): 58% parameter utilization
- 102.7M embedding permanently frozen on AMD (scatter_add_)
- 69.2M HyperNetwork defined but NEVER CALLED from any training loop
- Pre-training used `torch.randint()` random targets — trained on noise
- RLFS "reward" was `dummy_logits.mean() * reward` — not real RL
- Sequential SSM scan: Python for-loop, 256 kernel launches per input

### V3 (ODE/LTC + AST): 17% parameter utilization
- 83% of all params = frozen embedding (dead weight)
- ODE core cut from 1,200 effective layers (V2) to 200 — 6x less capacity
- AST decoder limited to 50 semantic strings — can't spell real variable names
- V2→V3 weight transfer silently fails (wrong key mapping)
- Adversarial verifier uses nn.GRU → crashes on DirectML

### V4 (Hyper-Routing Reasoner): 23% parameter utilization
- 65% of 237M params are frozen embeddings (154M, dead weight)
- 256-token pointer window — SWE-bench needs 5,000-50,000 tokens
- Expert bank returns `torch.randn()` — no expert files ever created
- Beam search multiplies copy distribution by zeros (pointer disabled)
- 5,000 toy training examples ("write a function that adds two numbers")
- ODE convergence ≠ correct next token

### The Pattern Across All Versions
```
V1: 99.97% frozen  →  V2: 42% frozen  →  V3: 83% frozen  →  V4: 65% frozen
```
Every version carries massive dead weight. The embedding table alone is 100M+ params
that can NEVER be trained on AMD hardware because nn.Embedding.backward uses scatter_add_.

V5 kills all of this. **100% of parameters are trainable. Zero dead weight.**

---

## 2. The Core Insight — Dual-Speed Learning <a name="2-the-core-insight"></a>

Every AI system today follows this lifecycle:
```
Train (expensive, one-time) → Deploy (frozen) → Serve (no learning)
```

GPT-4, Claude, Gemini, DeepSeek, Llama — ALL frozen at deployment. They never learn
from mistakes. Every session starts from zero. They can't specialize.

V5 breaks this with **Dual-Speed Complementary Learning**, modeled after the human brain:

### Fast Brain (Hippocampus analog)
- **What**: ChromaDB vector database storing complete episode traces
- **Speed**: Instant — saves after every task, retrieves in <50ms
- **Durability**: Fragile — individual memories, not generalized knowledge
- **Function**: "Last time I saw this error, I fixed it by changing the import path"

### Slow Brain (Neocortex analog)
- **What**: The 1.5B Resonance Core model weights, updated via LoRA
- **Speed**: Slow — fine-tunes periodically on accumulated traces
- **Durability**: Permanent — patterns baked into weights
- **Function**: "Import errors usually mean a module was moved or renamed"

### How They Interact
```
Day 1: Novel task → Fast Brain has nothing → Slow Brain guesses → Score result → Save trace
Day 2: Similar task → Fast Brain retrieves Day 1 trace → Bias the Slow Brain's decision → Better result
Day 7: 5 similar tasks accumulated → Slow Brain fine-tunes on all 5 → Pattern learned in weights
Day 30: Fast Brain full of traces → Slow Brain absorbs all → Generalized understanding
Day 90: Model handles tasks it's NEVER seen by combining learned patterns
```

**Nobody has built this.** Google uses federated learning for keyboard predictions.
OpenAI uses RLHF for one-time training. Nobody does continuous dual-speed
complementary learning for a code agent.

---

## 3. Architecture Overview <a name="3-architecture-overview"></a>

```
                    ┌──────────────────────────────────────────────────────┐
                    │                V5 LIVING MODEL                       │
                    │                                                      │
  User Request ────►│  ┌────────────────────────────────────────────┐      │
                    │  │         OBSERVATION ENCODER                 │      │
                    │  │  User text + tool outputs + error msgs     │      │
                    │  │  → Binary Embedding → Elastic Context      │      │
                    │  └─────────────────┬──────────────────────────┘      │
                    │                    │                                  │
                    │                    ▼                                  │
                    │  ┌─────────────────────────────────────┐             │
                    │  │       FAST BRAIN (ChromaDB)          │             │
                    │  │  Retrieve top-K similar past traces  │             │
                    │  │  → Memory-biased context vector      │             │
                    │  └─────────────┬───────────────────────┘             │
                    │                │                                      │
                    │                ▼                                      │
                    │  ┌─────────────────────────────────────┐             │
                    │  │     8× RESONANCE BLOCKS              │             │
                    │  │  Path A: Sliding Window Attention     │             │
                    │  │  Path B: Spectral FFT Mixing          │             │
                    │  │  → Gated dual-path fusion             │             │
                    │  └─────────────┬───────────────────────┘             │
                    │                │                                      │
                    │         ┌──────┴──────┐                              │
                    │         ▼             ▼                              │
                    │  ┌────────────┐ ┌───────────────┐                    │
                    │  │  ACTION     │ │  OUTCOME       │                   │
                    │  │  DECODER    │ │  PREDICTOR     │                   │
                    │  │            │ │                 │                   │
                    │  │  Tool ID   │ │  P(success)     │                   │
                    │  │  + Args    │ │  before acting  │                   │
                    │  └─────┬──────┘ └───────┬─────────┘                  │
                    │        │                │                             │
                    │        ▼                ▼                             │
                    │  ┌─────────────────────────────────────┐             │
                    │  │        AGENT ACTION LOOP             │             │
                    │  │  Execute tool → Observe → Score      │             │
                    │  │  → Save trace → Loop or respond      │             │
                    │  └─────────────────┬───────────────────┘             │
                    │                    │                                  │
                    │                    ▼                                  │
                    │  ┌─────────────────────────────────────┐             │
                    │  │       SLOW BRAIN (LoRA Updater)      │             │
                    │  │  Every N traces: fine-tune weights   │             │
                    │  │  Consolidate fast→slow knowledge     │             │
                    │  └─────────────────────────────────────┘             │
                    │                                                      │
                    └──────────────────────────────────────────────────────┘
```

---

## 4. Novel Component 1: Binary Embedding <a name="4-binary-embedding"></a>

### The Problem
`nn.Embedding` uses a lookup table. Its backward pass uses `scatter_add_` to route
gradients to the correct row. DirectML (AMD) rejects scatter ops. This means:
- V2: 102.7M embedding params frozen (25% of model)
- V3: 102.7M frozen (83% of model)
- V4: 154M frozen across two embedding tables (65% of model)

### The Invention
Represent each token ID as its **binary bit pattern**, then learn the embedding
via standard matrix multiplies (which DirectML supports perfectly).

```python
# Traditional (BROKEN on DirectML backward):
embed = nn.Embedding(100279, 1024)  # 102.7M params, scatter_add_ in backward
x = embed(token_ids)               # [B, T] → [B, T, 1024]

# V5 Binary Embedding (FULLY TRAINABLE on DirectML):
# token_id 4821 → binary [1,0,0,1,0,1,1,0,1,1,0,0,1,0,1,0,0]  (17 bits)
# 2^17 = 131072 > 100279 vocab size ✓
bits = ((token_ids.unsqueeze(-1) >> bit_positions) & 1).float()  # [B, T, 17]
x = gelu(linear1(bits))  # [B, T, 256]   — Linear(17, 256)
x = linear2(x)           # [B, T, 1024]  — Linear(256, 1024)
```

### Why This Works
- `>>` (bit shift) and `& 1` (bitwise AND) have no backward — they're applied to integer
  indices, not learnable parameters
- `linear1` and `linear2` are standard `nn.Linear` — backward is pure matmul, no scatter
- **17 × 256 + 256 × 1024 = 266,240 params** — replaces 102.7M frozen params
- 100% trainable. Every param gets gradients.

### Multi-Scale Variant (for richer representations)
```python
# Hash-based multi-scale: 3 different "views" of each token ID
hash1 = token_id % 2048   → nn.Linear(11, 128)  # coarse semantic cluster
hash2 = token_id % 16384  → nn.Linear(14, 128)  # medium detail
hash3 = binary(token_id)  → nn.Linear(17, 128)  # exact identity
concat [h1, h2, h3] → project to 1024           # fused embedding
```

This gives the model three semantic "lenses" to understand each token:
what cluster it belongs to, a finer grouping, and its exact identity.

### Parameters
| Component | Params | Trainable |
|---|---|---|
| Bit projection (17→256→1024) | 266K | 100% |
| Multi-scale hash projections | ~200K | 100% |
| Fusion projection (384→1024) | 394K | 100% |
| Layer norms | 2K | 100% |
| **Total** | **~860K** | **100%** |

Compare: V4 embedding = 154M (65% of model), 0% trainable on AMD.
V5 embedding = 860K (0.06% of model), 100% trainable. **180x smaller, fully learnable.**

---

## 5. Novel Component 2: Resonance Blocks <a name="5-resonance-blocks"></a>

### The Problem
- V2 used a 12-layer dense recurrent loop (214M params). Decent capacity but sequential.
- V3 used 4-block ODE integration (16.8M params). Too small. Convergence ≠ correctness.
- V4 inherited V3's ODE. Still too small, still meaningless convergence.
- Standard Transformers use O(n²) full attention. Too expensive for 8K context.
- Mamba uses sequential SSM. Can't parallelize on DirectML (no Triton kernel).

### The Invention: Dual-Path Resonance Block

Each block has two parallel processing paths that capture different types of patterns:

```
Input x ─────────────┬──────────────────────────────┐
                     │                               │
                     ▼                               ▼
          ┌──────────────────┐           ┌──────────────────────┐
          │  PATH A: LOCAL   │           │  PATH B: GLOBAL      │
          │  Sliding Window  │           │  Spectral FFT Mix    │
          │  Attention       │           │                      │
          │  window=128      │           │  FFT → filter → iFFT │
          │  O(n × 128)      │           │  O(n log n)          │
          │                  │           │                      │
          │  Sees nearby     │           │  Sees repetition     │
          │  tokens clearly  │           │  patterns across     │
          │                  │           │  entire sequence     │
          └────────┬─────────┘           └──────────┬───────────┘
                   │                                │
                   ▼                                ▼
          ┌────────────────────────────────────────────────────┐
          │              GATED FUSION                           │
          │  gate = sigmoid(W_gate · [local; global])          │
          │  output = gate * local + (1 - gate) * global       │
          │  + SwiGLU FFN + RMSNorm                            │
          └────────────────────────────────────────────────────┘
                                   │
                                   ▼
                               Output
```

### Path A: Sliding Window Attention
```python
# Causal attention with a sliding window of 128 tokens
# Each token attends to at most 128 previous tokens
# O(n × window) instead of O(n²)

Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)  # [B, T, d] each
# Build causal mask limited to window_size
attn = scaled_dot_product_attention(Q, K, V, attn_mask=window_mask)
```

Why sliding window: Code has strong LOCAL dependencies. The `if` on line 5 controls
the `else` on line 7. Variable declared on line 10 is used on line 12. You don't
need full attention to capture these — a 128-token window covers ~50 lines of code.

### Path B: Spectral Global Mixing
```python
# Transform to frequency domain
X_freq = torch.fft.rfft(x, dim=1)  # [B, T//2+1, d] complex

# Learnable frequency filter — model decides which "rhythms" matter
X_filtered = X_freq * self.freq_filter  # element-wise, learnable [T//2+1, d]

# Transform back
x_global = torch.fft.irfft(X_filtered, n=T, dim=1)  # [B, T, d]
```

Why spectral: Code has GLOBAL repetition patterns. Function definitions repeat every
~30 lines. Import blocks cluster at the top. Class methods follow `def ... self ...`
patterns. These are literally frequencies in the token sequence. The FFT turns them
into explicit signals that the model can filter and amplify.

**This combination has never been done.** Existing architectures pick one:
- Transformers: full attention (O(n²), captures everything, way too expensive)
- Mamba/RWKV: sequential recurrence (linear, captures long-range, can't parallelize)
- Hyena: implicit long convolution (O(n log n), but no explicit local attention)

V5 Resonance Blocks get BOTH local precision AND global pattern recognition,
at O(n × 128 + n log n) cost.

### DirectML Safety
- `torch.fft.rfft` / `torch.fft.irfft`: Pure math ops, no scatter. ✓
- Sliding window attention: Uses `F.scaled_dot_product_attention` with a mask. ✓
- `StableSigmoid` for gating (tanh-based, proven in V3). ✓
- SwiGLU FFN: matmuls + elementwise. ✓
- RMSNorm: `x * rsqrt(mean(x²))`. ✓

### Parameters (per block)
| Component | Params |
|---|---|
| QKV projection (d → 3d) | 3 × d² = 3.1M |
| Attention out projection | d² = 1.05M |
| Frequency filter | (T//2+1) × d ≈ 0.5M |
| Spectral out projection | d² = 1.05M |
| Gate projection (2d → d) | 2d² = 2.1M |
| SwiGLU FFN (d → 4d → d) | 3 × d × 4d = 12.6M |
| RMSNorm × 3 | 3d = 3K |
| **Total per block** | **~20.4M** |

**8 blocks × 20.4M = ~163M total** — all trainable.

---

## 6. Novel Component 3: Elastic Hierarchical Context <a name="6-elastic-context"></a>

### The Problem
V4 hardcodes 256-token context. SWE-bench needs 5,000-50,000. The pointer-generator
can only copy from tokens it can see.

### The Invention
Multi-resolution compression: recent tokens at full detail, older tokens compressed.

```
Input: 8192 tokens from conversation history

Level 0 (recent):   tokens[0:1024]      → kept at full resolution      → 1024 slots
Level 1 (medium):   tokens[1024:3072]   → avg_pool1d(kernel=4, stride=4) → 512 slots
Level 2 (distant):  tokens[3072:8192]   → avg_pool1d(kernel=16, stride=16) → 320 slots
                                                                   Total: 1856 slots
```

### Implementation
```python
class ElasticContext(nn.Module):
    def __init__(self, d_model=1024, levels=3):
        self.level_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(levels)])
        self.level_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(levels)])
        # Learnable scale weights — model decides how much to trust each level
        self.level_gates = nn.Parameter(torch.ones(levels) / levels)

    def forward(self, tokens_embedded):
        # tokens_embedded: [B, T_full, d] where T_full can be up to 8192
        chunks = [
            tokens_embedded[:, :1024, :],                              # Level 0: full
            F.avg_pool1d(tokens_embedded[:, 1024:3072, :].transpose(1,2),
                        kernel_size=4).transpose(1,2),                  # Level 1: 4:1
            F.avg_pool1d(tokens_embedded[:, 3072:, :].transpose(1,2),
                        kernel_size=16).transpose(1,2),                 # Level 2: 16:1
        ]
        # Normalize and project each level
        processed = []
        for i, chunk in enumerate(chunks):
            processed.append(self.level_projs[i](self.level_norms[i](chunk)))

        # Concatenate: [B, 1024+512+320, d] = [B, 1856, d]
        return torch.cat(processed, dim=1)
```

### Why This Works
- **Recent code (1024 tokens)**: Full resolution. The model sees every token in the
  current file/function being edited. Enough for copying exact variable names.
- **Medium range (2048→512 tokens)**: 4:1 compression. The model sees the structure
  of surrounding code — class definitions, import blocks, function signatures.
- **Long range (5120→320 tokens)**: 16:1 compression. The model sees the project
  architecture — which files exist, what modules are imported, the overall shape.

### DirectML Safety
- `F.avg_pool1d`: Fully supported on DirectML. No scatter. ✓
- Concatenation: `torch.cat`. ✓
- RMSNorm + Linear: Pure matmul. ✓

### Parameters
| Component | Params |
|---|---|
| 3× RMSNorm | 3K |
| 3× Level projections (d→d) | 3.1M |
| Level gates | 3 |
| **Total** | **~3.1M** |

---

## 7. Novel Component 4: Observation Encoder <a name="7-observation-encoder"></a>

### Purpose
Processes ALL information the model receives: user messages, tool outputs (file contents,
terminal output, error messages, search results), and memory retrieval results.

### Design
```
User message:     "Fix the import error in server.py"
Tool output:      "FileNotFoundError: No module named 'utils.helper'"
Memory retrieval: [Trace #38: similar import fix, solution was path correction]
                          │
                          ▼
              ┌──────────────────────┐
              │  Binary Embedding     │  (Component 1)
              │  token_ids → vectors │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Type Embedding       │
              │  [USER, TOOL_FILE,   │
              │   TOOL_TERM, ERROR,  │
              │   MEMORY, SYSTEM]    │
              │  6 types × d dim     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Elastic Context      │  (Component 3)
              │  8192 → 1856 slots   │
              └──────────┬───────────┘
                         │
                         ▼
              Context tensor [B, 1856, 1024]
```

Each input segment is tagged with a TYPE embedding so the model knows whether it's
reading user text, file contents, terminal output, or a past memory. This is critical
because the same text means different things in different contexts:

```python
"ImportError: No module named 'utils'"
# As TOOL_TERM output: this is the current error to fix
# As MEMORY trace: this is how a similar error was fixed before
```

### Parameters
| Component | Params |
|---|---|
| Type embedding (6 × d, no scatter — tiny, uses matmul projection) | 6K |
| Segment position encoding (sinusoidal buffer) | 0 (buffer) |
| **Total** | **~6K** |

(Binary Embedding and Elastic Context are counted separately.)

---

## 8. Novel Component 5: Action Decoder <a name="8-action-decoder"></a>

### The Problem With Every Existing Agent
GPT-4, Claude, etc. are TEXT COMPLETION models forced to act as agents:
```
<|system|>You are a coding assistant with tools...
<|user|>Fix the bug...
<|assistant|>I'll search for the file.\n<tool_call>{"name": "search", "query": "bug"}</tool_call>
```

The model generates arbitrary text and the framework PARSES it for tool calls.
This is fragile: the model might generate malformed JSON, hallucinate tool names,
or produce text that looks like a tool call but isn't.

### The V5 Way: Action Decoder (structured output, not text parsing)
The Action Decoder outputs structured decisions directly from the neural network,
not as parsed text:

```python
class ActionDecoder(nn.Module):
    def __init__(self, d_model=1024, n_tools=8, max_arg_tokens=512):
        # Head 1: Which tool to use
        self.tool_head = nn.Linear(d_model, n_tools)  # softmax → tool selection

        # Head 2: Confidence — should I act or think more?
        self.confidence_head = nn.Linear(d_model, 1)  # sigmoid → P(ready to act)

        # Head 3: Argument generator — for EDIT/RESPOND tools
        # Uses the pointer-generator mechanism from V4 (proven, but with 1856 slots)
        self.arg_generator = PointerGeneratorHead(d_model, vocab_size, context_slots=1856)

    def forward(self, hidden_state, context):
        # hidden_state: [B, d] — last resonance block output, pooled
        tool_logits = self.tool_head(hidden_state)        # [B, n_tools]
        confidence = stable_sigmoid(self.confidence_head(hidden_state))  # [B, 1]

        if confidence < threshold:
            return Action(tool="THINK", args=None)  # Loop back through resonance blocks

        tool_id = tool_logits.argmax(dim=-1)
        if tool_requires_text_output(tool_id):
            args = self.arg_generator(hidden_state, context)  # Use pointer-gen
        else:
            args = self.arg_head(hidden_state)  # Simple MLP for search queries, etc.

        return Action(tool=TOOL_NAMES[tool_id], args=args)
```

### Tool Action Space
```
ID 0: SEARCH   — grep/semantic search in workspace     (arg: query string)
ID 1: READ     — read a file                           (arg: file path + line range)
ID 2: EDIT     — modify a file                         (arg: file path + old + new text)
ID 3: RUN      — execute a terminal command             (arg: command string)
ID 4: THINK    — internal reasoning step, no external action
ID 5: RESPOND  — generate text response to user         (arg: response text)
ID 6: RECALL   — explicitly query Fast Brain memory     (arg: query)
ID 7: INTROSPECT — read/modify own architecture         (arg: file path + patch)
```

### The THINK Action (Internal Reasoning)
When the model picks THINK, it doesn't execute any tool. Instead, the THINK action's
output is appended to the context as a `SYSTEM` type segment, and the model loops
through the Resonance Blocks again. This is chain-of-thought reasoning, but
**architecturally native** — not prompt-engineering.

```
Iteration 1: User asks "fix the bug in auth.py"
  → Model picks THINK: "I should first read auth.py to see the current code"
  → THINK output appended to context
Iteration 2: Context now includes the THINK
  → Model picks READ: args = ("auth.py", lines 1-50)
  → Tool returns file contents
Iteration 3: Context now includes file contents
  → Model picks THINK: "Line 23 imports from 'utils.helper' but file is 'utils.helpers'"
  → THINK output appended
Iteration 4: THINK + file contents in context
  → Model picks EDIT: args = ("auth.py", "utils.helper", "utils.helpers")
  → Tool applies edit
Iteration 5: Edit confirmed
  → Model picks RUN: args = ("python -m pytest tests/test_auth.py")
  → Tool returns test results
Iteration 6: Tests pass
  → Model picks RESPOND: "Fixed the import in auth.py — changed 'utils.helper' to 'utils.helpers'"
```

### Parameters
| Component | Params |
|---|---|
| Tool selection head (d → 8) | 8K |
| Confidence head (d → 1) | 1K |
| Argument MLP (d → d → max_arg) | 2.1M |
| Pointer-generator head (from V4, upgraded) | ~8M |
| **Total** | **~10M** |

---

## 9. Novel Component 6: Outcome Predictor <a name="9-outcome-predictor"></a>

### Purpose
BEFORE executing an action, predict whether it will succeed. This serves two functions:
1. **Efficiency**: Don't waste time on actions predicted to fail
2. **Training signal**: The prediction error (predicted success vs actual outcome)
   is a free gradient signal for the model

### Design
```python
class OutcomePredictor(nn.Module):
    def __init__(self, d_model=1024):
        self.predictor = nn.Sequential(
            nn.Linear(d_model + 8, d_model // 4),   # +8 for tool one-hot
            nn.GELU(),
            nn.Linear(d_model // 4, 3),  # 3 outcomes: SUCCESS, PARTIAL, FAILURE
        )
```

### Training
The outcome predictor is trained with simple cross-entropy against actual outcomes:
```
Predicted: SUCCESS (0.7)  |  Actual: FAILURE  |  Loss: -log(0.15) = 1.9
```

Over time, the model learns to predict which actions will work BEFORE trying them.
This is a form of **world modeling** — the model builds an internal simulation of
what will happen, without actually executing the action.

### Parameters: ~0.5M

---

## 10. Novel Component 7: Fast Brain — Episodic Memory <a name="10-fast-brain"></a>

### Purpose
Instant one-shot learning. After every completed task (success or failure), store the
full action trace as a retrievable episode.

### Implementation
```python
class FastBrain:
    def __init__(self, db_path="v5_core/chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"}
        )

    def store_episode(self, trace: ActionTrace):
        """Store a complete task trace after it finishes."""
        embedding = self.encode_trace(trace)  # Use the model's own encoder
        self.collection.add(
            ids=[trace.id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "task_type": trace.task_type,
                "success": trace.success,
                "num_steps": trace.num_steps,
                "timestamp": trace.timestamp,
                "score": trace.score,
            }],
            documents=[trace.to_json()]
        )

    def recall(self, current_context: str, top_k: int = 3) -> List[ActionTrace]:
        """Retrieve similar past traces for the current task."""
        results = self.collection.query(
            query_texts=[current_context],
            n_results=top_k,
            where={"success": True}  # Prefer successful traces
        )
        return [ActionTrace.from_json(doc) for doc in results["documents"][0]]
```

### How Memory Biases Current Decisions
Retrieved traces are injected into the Observation Encoder with `MEMORY` type tags:
```
[USER] Fix the circular import in models.py
[MEMORY] Trace #38 (score=0.95): Similar task 3 days ago.
  Step 1: SEARCH("circular import" in models/) → found in models/user.py line 5
  Step 2: READ(models/user.py) → saw "from models.order import Order"
  Step 3: READ(models/order.py) → saw "from models.user import User" ← CIRCULAR
  Step 4: EDIT(models/order.py, moved import inside function) → resolved
  Score: 0.95 (tests passed)
```

The model sees EXACTLY what worked before and can replicate or adapt the strategy.

### Storage Requirements
- Each trace: ~2-5 KB compressed JSON
- 1,000 traces: ~5 MB
- 10,000 traces: ~50 MB (1 year of heavy use)
- ChromaDB index: <100 MB total

### Parameters: 0 (uses model's existing encoder for embeddings)

---

## 11. Novel Component 8: Slow Brain — Weight Consolidation <a name="11-slow-brain"></a>

### Purpose
Periodically distill accumulated traces into the model weights via LoRA fine-tuning.
This converts fast, fragile memories into slow, permanent knowledge.

### Local Learning Loop (runs on RX 7600)
```python
class SlowBrain:
    def consolidate(self, model, fast_brain, n_traces=50):
        """Fine-tune model on best traces. Runs locally on RX 7600."""
        # 1. Sample top traces by score
        traces = fast_brain.get_top_traces(n=n_traces, min_score=0.5)

        # 2. Format as training examples
        #    Input: observation sequence (user + tool outputs)
        #    Target: correct action sequence (tool_id + args)
        dataset = [trace.to_training_pair() for trace in traces]

        # 3. LoRA fine-tune (rank=8, only on resonance blocks)
        lora_config = LoRAConfig(
            target_modules=["qkv_proj", "out_proj", "ffn"],
            rank=8, alpha=16
        )
        trainer = LocalLoRATrainer(model, lora_config, lr=1e-5)
        trainer.train(dataset, epochs=2, batch_size=1)

        # 4. Merge LoRA into base weights
        trainer.merge_and_save("v5_core/snapshot_v5_evolved.pt")
```

### Remote Learning Loop (runs on RunPod A100)
```python
class FederatedConsolidation:
    def full_retrain(self, trace_archive, base_weights):
        """Full fine-tune on all accumulated traces. Runs on RunPod A100."""
        # 1. Load ALL traces (local + federated from other users)
        all_traces = trace_archive.load_all()

        # 2. Full fine-tune (not LoRA — update all weights)
        trainer = FullTrainer(base_weights, lr=2e-5)
        trainer.train(all_traces, epochs=3, batch_size=16)

        # 3. Export optimized checkpoint
        trainer.save("v5_core/snapshot_v5_retrained.pt")
```

### Consolidation Schedule
```
Every 50 traces:   Local LoRA update (10 min on RX 7600, batch_size=1)
Every 500 traces:  Upload traces to RunPod, full retrain (2-4 hours on A100)
Every 5000 traces: Federated merge with other users' traces + full retrain
```

### Parameters: 0 additional (updates existing model weights)

---

## 12. Agent Action Loop <a name="12-agent-loop"></a>

### The Main Loop
```python
def agent_loop(model, user_request, fast_brain, tools, max_steps=20):
    # Initialize context with user request
    context = ObservationEncoder.encode(user_request, type="USER")

    # Retrieve relevant memories
    memories = fast_brain.recall(user_request, top_k=3)
    for mem in memories:
        context = context + ObservationEncoder.encode(mem, type="MEMORY")

    trace = ActionTrace(user_request)

    for step in range(max_steps):
        # Run through Resonance Blocks
        hidden = model.resonance_forward(context)

        # Predict action
        action = model.action_decoder(hidden, context)

        # Predict outcome before acting
        predicted_outcome = model.outcome_predictor(hidden, action)

        if action.tool == "THINK":
            # Internal reasoning — no tool execution
            think_text = model.generate_thought(hidden)
            context = context + ObservationEncoder.encode(think_text, type="SYSTEM")
            trace.add_step("THINK", think_text, None)
            continue

        if action.tool == "RESPOND":
            # Task complete — generate response
            response = model.action_decoder.generate_text(hidden, context)
            trace.finalize(response, success=True)
            break

        # Execute the tool
        result = tools.execute(action)

        # Score this step
        step_score = trace_scorer.score_step(action, result, predicted_outcome)

        # Add result to context
        context = context + ObservationEncoder.encode(result, type=tool_type(action))
        trace.add_step(action.tool, action.args, result, step_score)

    # Save complete trace to Fast Brain
    trace.compute_final_score()
    fast_brain.store_episode(trace)

    return trace
```

### Step Budget
- Max 20 actions per task
- THINK steps don't count against limit (internal only)
- If model hits 20 steps without RESPOND, forced to respond with current best

---

## 13. Tool Registry <a name="13-tool-registry"></a>

### Built-in Tools
```python
TOOLS = {
    "SEARCH": {
        "description": "Search for text patterns in workspace files",
        "args": {"query": str, "path": Optional[str], "regex": bool},
        "executor": grep_search_tool,
    },
    "READ": {
        "description": "Read contents of a file",
        "args": {"path": str, "start_line": int, "end_line": int},
        "executor": read_file_tool,
    },
    "EDIT": {
        "description": "Replace text in a file",
        "args": {"path": str, "old_text": str, "new_text": str},
        "executor": edit_file_tool,
    },
    "RUN": {
        "description": "Execute a terminal command",
        "args": {"command": str, "timeout": int},
        "executor": run_command_tool,
    },
    "THINK": {
        "description": "Internal reasoning step (no external action)",
        "args": {},
        "executor": None,  # Handled by agent loop
    },
    "RESPOND": {
        "description": "Generate final response to user",
        "args": {"text": str},
        "executor": None,  # Handled by agent loop
    },
    "RECALL": {
        "description": "Query episodic memory for similar past experiences",
        "args": {"query": str, "top_k": int},
        "executor": fast_brain_recall,
    },
    "INTROSPECT": {
        "description": "Read or modify own source code",
        "args": {"file": str, "action": str, "patch": Optional[str]},
        "executor": code_introspector_tool,
    },
}
```

### Tool Safety
- `RUN` commands execute in a sandboxed subprocess with timeout
- `EDIT` creates a backup before modifying
- `INTROSPECT` patches are validated (syntax check) before applying
- All tool outputs are truncated to 4096 tokens max before feeding back to context

---

## 14. Experience Buffer & Trace Format <a name="14-experience-buffer"></a>

### Trace Schema
```json
{
    "id": "trace_00482",
    "timestamp": "2026-02-28T14:30:00Z",
    "user_request": "Fix the import error in server.py",
    "steps": [
        {
            "step": 0,
            "action": "THINK",
            "args": null,
            "thought": "I should read server.py first to see the error",
            "result": null,
            "predicted_success": null,
            "actual_success": null
        },
        {
            "step": 1,
            "action": "READ",
            "args": {"path": "server.py", "start_line": 1, "end_line": 30},
            "result": "from utils.helper import process_data\n...",
            "predicted_success": 0.85,
            "actual_success": 1.0
        },
        {
            "step": 2,
            "action": "SEARCH",
            "args": {"query": "def process_data", "path": "utils/"},
            "result": "utils/helpers.py:15: def process_data(...):",
            "predicted_success": 0.72,
            "actual_success": 1.0
        },
        {
            "step": 3,
            "action": "EDIT",
            "args": {
                "path": "server.py",
                "old_text": "from utils.helper import",
                "new_text": "from utils.helpers import"
            },
            "result": "Edit applied successfully",
            "predicted_success": 0.91,
            "actual_success": 1.0
        },
        {
            "step": 4,
            "action": "RUN",
            "args": {"command": "python server.py", "timeout": 10},
            "result": "Server started on port 8080",
            "predicted_success": 0.88,
            "actual_success": 1.0
        },
        {
            "step": 5,
            "action": "RESPOND",
            "args": {"text": "Fixed: changed 'utils.helper' to 'utils.helpers' in server.py"},
            "result": null,
            "predicted_success": null,
            "actual_success": null
        }
    ],
    "final_score": 0.95,
    "task_type": "bug_fix",
    "num_steps": 6,
    "success": true
}
```

### Scoring
```python
class TraceScorer:
    def score(self, trace):
        score = 0.0

        # Did the task complete? (40% weight)
        if trace.success:
            score += 0.4

        # Efficiency — fewer steps = better (20% weight)
        efficiency = max(0, 1.0 - trace.num_steps / 20)
        score += 0.2 * efficiency

        # Prediction accuracy — did outcome predictor match reality? (20% weight)
        pred_errors = [abs(s.predicted - s.actual) for s in trace.steps if s.predicted]
        pred_accuracy = 1.0 - (sum(pred_errors) / max(len(pred_errors), 1))
        score += 0.2 * pred_accuracy

        # Final verification — did RUN at the end pass? (20% weight)
        if trace.has_final_test and trace.final_test_passed:
            score += 0.2

        return score
```

---

## 15. Self-Generated Curriculum <a name="15-curriculum"></a>

### How the Model Identifies Its Weaknesses
```python
class CurriculumGenerator:
    def analyze_weaknesses(self, fast_brain):
        """Find task types where the model fails most."""
        all_traces = fast_brain.get_all_traces(last_n=200)

        # Group by task type and compute success rates
        by_type = defaultdict(list)
        for trace in all_traces:
            by_type[trace.task_type].append(trace.score)

        # Find weak spots
        weaknesses = []
        for task_type, scores in by_type.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.6:  # Below 60% success
                weaknesses.append((task_type, avg_score, len(scores)))

        return sorted(weaknesses, key=lambda x: x[1])  # Worst first
```

### How It Generates Practice Problems
```python
    def generate_practice(self, weakness_type, n=10):
        """Create synthetic tasks for the model's weakest area."""
        templates = {
            "async_debugging": [
                "Fix the race condition in {module}.py where {resource} is accessed concurrently",
                "Add proper async/await to the {function} that currently blocks the event loop",
                "Debug why asyncio.gather fails when {condition}",
            ],
            "import_resolution": [
                "Fix the circular import between {module_a} and {module_b}",
                "Resolve the ModuleNotFoundError for {package} in {file}",
            ],
            # ... more templates per task type
        }
        # Generate from templates with random fills
        # The model solves each one, scores itself, and trains on results
```

### Practice Loop (runs during idle time)
```
1. Analyze recent traces → find weakness: "async_debugging" (avg score 0.35)
2. Generate 10 async debugging practice tasks
3. Set up sandbox workspace with test files
4. Solve each task via agent_loop (with fast_brain retrieval)
5. Score results automatically (did the fix work?)
6. Successful traces → training data for slow brain
7. Repeat with next weakness
```

---

## 16. Self-Modification via Code Introspector <a name="16-self-modification"></a>

### Already Built (from previous session)
`v5_core/architecture/code_introspector.py` can:
- Parse any Python file as an AST
- Build a call graph (which functions call which)
- Compute file hashes for change detection
- Apply AST-level patches (add/remove/modify functions)

### V5 Integration
The Code Introspector becomes a TOOL the model can call:

```
Scenario: Model notices it's slow at generating long responses

Step 1: INTROSPECT("v5_core/architecture/resonance_block.py", action="read")
  → Model reads its own block implementation

Step 2: THINK("The spectral path processes all frequencies equally.
  I could add a top-k frequency selection to skip irrelevant frequencies.")

Step 3: INTROSPECT("v5_core/architecture/resonance_block.py",
  action="patch",
  patch="add top_k selection after rfft, before filter multiply")

Step 4: RUN("python -c 'import v5_core; v5_core.test_resonance_block()'")
  → Validates the modification works

Step 5: If test passes → commit the change and retrain on existing traces
```

### Safety Guardrails
- All patches create backups first
- Syntax validation before applying
- Automatic rollback if tests fail after patch
- Cannot modify safety guardrail code itself (hardcoded exclusion list)

---

## 17. Federated Multi-User Learning <a name="17-federation"></a>

### Architecture
```
  Haziq's Instance          Friend A's Instance       Friend B's Instance
  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
  │ V5 + LoRA_H  │          │ V5 + LoRA_A  │          │ V5 + LoRA_B  │
  │ 500 traces   │          │ 300 traces   │          │ 400 traces   │
  │ ML/Python    │          │ React/TS      │          │ C++/embedded │
  └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
         │                         │                         │
         └─────────┐     ┌─────────┘     ┌───────────────────┘
                   │     │               │
                   ▼     ▼               ▼
            ┌─────────────────────────────────┐
            │     RunPod A100: Federation     │
            │                                 │
            │  1. Collect anonymized traces   │
            │  2. Deduplicate & filter        │
            │  3. Full fine-tune base model   │
            │  4. Distribute updated base     │
            │                                 │
            │  Each user keeps personal LoRA  │
            │  on top of shared base          │
            └─────────────────────────────────┘
```

### Privacy
- Traces are anonymized: file paths stripped, variable names randomized
- Only ACTION PATTERNS are shared, not actual code content
- Each user's personal LoRA stays local

### Scaling Effect
```
1 user  × 100 traces/week = base model sees 100 patterns/week
3 users × 100 traces/week = base model sees 300 patterns/week
10 users × 100 traces/week = base model sees 1000 patterns/week

More users → base model improves faster → each user benefits → positive feedback loop
```

---

## 18. Training Pipeline <a name="18-training"></a>

### Phase 1: Pre-training on RunPod A100 (3-5 days, ~$200-400)

**Dataset construction:**
```
Source 1: The Stack v2 (subset) — 50GB of Python, JavaScript, TypeScript code
  → Tokenize with tiktoken (cl100k_base, same as V4)
  → Pre-training objective: next-token prediction
  → This teaches code KNOWLEDGE (syntax, APIs, patterns)

Source 2: Synthetic agent traces — 50,000 generated traces
  → Templates of (task_description, correct_action_sequence)
  → Pre-training objective: next-action prediction
  → This teaches AGENCY (when to search, read, edit, test)

Source 3: SWE-bench training set — 2,000+ real bug-fix traces
  → Convert bug reports + gold patches into action traces
  → Fine-tuning objective: next-action prediction on real data
```

**Model size for pre-training: 1.5B params**
```
Binary Embedding:       ~0.9M
8× Resonance Blocks:    ~163M   (d=1536 for 1.5B scale)
Elastic Context:        ~4.7M
Observation Encoder:    ~7K
Action Decoder:         ~15M
Outcome Predictor:      ~0.8M
Output LM Head:         ~1.5B × ... (standard tied projection)

VRAM for training (fp16, A100 80GB):
  Model: ~3 GB
  Optimizer (AdamW): ~6 GB
  Activations (batch=16, seq=2048): ~20 GB
  Gradient checkpointing: saves ~40%
  Total: ~29 GB → fits in A100 80GB ✓
```

### Phase 2: Deploy locally on RX 7600

**Quantize to 4-bit:**
```
1.5B params × 4 bits / 8 = ~750 MB VRAM
+ KV cache (1856 slots × 1536 dim × 2 × 2 bytes) = ~11 MB
+ Activations = ~50 MB
Total: ~810 MB → fits in RX 7600 8GB with 7.2 GB to spare ✓
```

### Phase 3: Continuous self-evolution (runs on RX 7600 forever)
```
Every task:
  → Run agent loop (inference only, ~810 MB VRAM)
  → Save trace to Fast Brain (ChromaDB, disk)

Every 50 traces:
  → Load LoRA adapters (~5 MB VRAM)
  → Fine-tune on top 50 traces (batch_size=1, ~2 GB additional VRAM)
  → Merge LoRA into weights
  → Total during LoRA: ~2.8 GB → fits ✓

Weekly (optional, RunPod):
  → Upload trace archive
  → Full retrain on A100
  → Download updated weights
```

---

## 19. Parameter Budget & VRAM Analysis <a name="19-params"></a>

### For local inference (RX 7600, 4-bit quantized)

| Component | Params | Trainable | VRAM (Q4) |
|---|---|---|---|
| Binary Embedding (multi-scale) | 860K | 860K (100%) | 0.4 MB |
| 8× Resonance Blocks | 163M | 163M (100%) | 81.5 MB |
| Elastic Context | 3.1M | 3.1M (100%) | 1.6 MB |
| Observation Encoder | 6K | 6K (100%) | 3 KB |
| Action Decoder + Pointer-Gen | 10M | 10M (100%) | 5 MB |
| Outcome Predictor | 0.5M | 0.5M (100%) | 0.25 MB |
| LM Head (tied with embedding) | — | — | — |
| **Total** | **~177M** | **177M (100%)** | **~89 MB** |

At 1.5B scale (RunPod trained):

| Component | Params | VRAM (Q4) |
|---|---|---|
| Binary Embedding | 1.4M | 0.7 MB |
| 12× Resonance Blocks (d=1536) | 1.3B | 650 MB |
| Elastic Context | 7M | 3.5 MB |
| Observation Encoder | 10K | 5 KB |
| Action Decoder | 25M | 12.5 MB |
| Outcome Predictor | 1M | 0.5 MB |
| Misc (norms, projections) | 10M | 5 MB |
| **Total** | **~1.35B** | **~672 MB** |

### Comparison to previous versions

| Version | Total Params | Trainable on AMD | Utilization | VRAM |
|---|---|---|---|---|
| V1 | 4B (Qwen) | 3.2M (LoRA only) | 0.08% | N/A (CPU) |
| V2 | 414M | 242M | 58% | ~1.6 GB |
| V3 | 123M | 21M | 17% | ~500 MB |
| V4 | 237M | 55M | 23% | ~950 MB |
| **V5 (local)** | **177M** | **177M** | **100%** | **~89 MB** |
| **V5 (RunPod)** | **1.35B** | **1.35B** | **100%** | **~672 MB (Q4)** |

**V5 at 1.35B has more TRAINABLE parameters than V1-V4 COMBINED.**

---

## 20. File Map <a name="20-file-map"></a>

```
v5_core/
│
├── __init__.py
│
├── architecture/
│   ├── __init__.py
│   ├── binary_embedding.py        ← Component 1: Scatter-free trainable embedding
│   │                                 - BitEncoder: token_id → 17-bit binary → MLP → vector
│   │                                 - MultiScaleHash: 3 hash views per token
│   │                                 - FusionProjection: concat → d_model
│   │
│   ├── resonance_block.py         ← Component 2: Dual-path spectral + local blocks
│   │                                 - SlidingWindowAttention: O(n × window)
│   │                                 - SpectralMixer: FFT → filter → iFFT
│   │                                 - GatedFusion: gate * local + (1-gate) * global
│   │                                 - ResonanceBlock: full block with FFN + norms
│   │
│   ├── elastic_context.py         ← Component 3: Multi-resolution 8K context
│   │                                 - ElasticContext: 3-level pooling hierarchy
│   │                                 - LevelGating: learnable scale weights
│   │
│   ├── observation_encoder.py     ← Component 4: Encodes all model inputs
│   │                                 - TypeEmbedding: [USER, TOOL, MEMORY, SYSTEM, ...]
│   │                                 - ObservationEncoder: binary_embed + type_embed + elastic
│   │
│   ├── action_decoder.py          ← Component 5: Structured action output
│   │                                 - ToolHead: softmax over tool IDs
│   │                                 - ConfidenceHead: P(ready to act)
│   │                                 - ArgumentGenerator: pointer-gen for text args
│   │
│   ├── outcome_predictor.py       ← Component 6: Predicts action success before execution
│   │                                 - OutcomePredictor: MLP → P(success)
│   │
│   ├── v5_assembly.py             ← Master assembly: wires all components together
│   │                                 - V5ResonanceModel: full forward pass
│   │                                 - Handles THINK loops internally
│   │
│   └── code_introspector.py       ← Already built: AST-based self-modification engine
│
├── agent/
│   ├── __init__.py
│   ├── agent_loop.py              ← Main agent execution loop (max 20 steps)
│   ├── tool_registry.py           ← SEARCH, READ, EDIT, RUN, THINK, RESPOND, RECALL, INTROSPECT
│   ├── fast_memory.py             ← Component 7: ChromaDB episodic memory (Fast Brain)
│   ├── experience_buffer.py       ← Trace storage, priority replay, archival
│   ├── trace_scorer.py            ← Automatic scoring of action traces
│   └── curriculum_generator.py    ← Self-generated practice problems from weaknesses
│
├── training/
│   ├── __init__.py
│   ├── pretrain_agent.py          ← RunPod A100: pre-train on The Stack + synthetic traces
│   ├── slow_learner.py            ← Component 8: Local LoRA fine-tune on accumulated traces
│   ├── federated_sync.py          ← Multi-user trace sharing and model merging
│   ├── data_generator.py          ← Generate synthetic agent traces for pre-training
│   └── terminal_loop.py           ← Already built: background code execution validator
│
├── utils/
│   ├── __init__.py
│   ├── dml_ops.py                 ← DirectML-safe ops: StableSigmoid, chunked_softmax, DML_GRUCell
│   ├── tokenizer.py               ← tiktoken cl100k_base wrapper
│   └── device.py                  ← Device detection (privateuseone:1 for RX 7600)
│
├── chroma_db/                     ← Fast Brain persistent storage
│
└── snapshots/                     ← Model checkpoints
    ├── v5_pretrained.pt           ← From RunPod pre-training
    ├── v5_evolved_latest.pt       ← Latest self-evolved weights
    └── lora/                      ← Active LoRA adapters
```

### Total new files: ~20
### Reused from previous sessions: 3 (code_introspector.py, terminal_loop.py, device.py)

---

## 21. What's Kept vs. Deleted from V1-V4 <a name="21-kept-vs-deleted"></a>

### KEPT (proven, battle-tested)

| Component | Origin | Why Kept |
|---|---|---|
| `chunked_softmax()` | V4 ast_decoder.py | Proven DirectML workaround for 100K+ dim softmax |
| `DML_GRUCell` | V4 ast_decoder.py | DirectML-safe GRU replacement (no fused_gru_cell) |
| `StableSigmoid` | V3 | tanh-based sigmoid avoids AMD aten::sigmoid crash |
| Pointer-generator concept | V4 ast_decoder.py | Copy attention for zero-shot vocab is correct; just needs more slots |
| ChromaDB integration | V4 | RAG retrieval infrastructure, repurposed for Fast Brain |
| Code Introspector | V5 (prev session) | AST-based self-modification, already tested |
| Terminal Loop | V5 (prev session) | Background validator, demoted from main training |
| `device.py` | V4 utils | RX 7600 → privateuseone:1 detection |
| `.detach()` patterns | V4 | Softmax max/sum detach for scatter-free backward |
| `==` broadcast masks | V4 | Replaces F.one_hot/gather for index selection |

### DELETED (dead weight, proven failures)

| Component | Origin | Why Deleted |
|---|---|---|
| `nn.Embedding` (100K×d) | V2-V4 | 100M+ params, FROZEN on DirectML. Replaced by Binary Embedding. |
| HolographicCompressor | V2-V4 | Sequential Python for-loop, 256 kernel launches. Dead. |
| ContinuousRecurrentCore (ODE) | V3-V4 | Convergence ≠ correctness. 16.8M params wasted. |
| FractalRecurrentCore | V2 | 214M params, trained on random targets. |
| HyperNetwork | V2 | 69.2M params, NEVER CALLED from any training loop. |
| AdversarialVerifier | V3 | Uses nn.GRU → crashes on DirectML. Unreachable. |
| AST Decoder (node types) | V3 | 50 semantic strings. Can't spell real code. |
| SSD Expert Bank | V4 | Returns `torch.randn()`. No expert files exist. |
| V4ContextRouter | V4 | Routes to phantom experts. |
| Loss head (4-class quality) | V4 | Structural quality ≠ code correctness. |
| RLFS trainer | V2-V3 | `dummy_logits.mean() * reward` is not real RL. |

---

## 22. DirectML Compatibility Guarantees <a name="22-directml"></a>

### Banned Operations (crash or produce wrong results on DirectML)

| Operation | Why It Crashes | V5 Alternative |
|---|---|---|
| `scatter_` | DirectML doesn't support partial dim scatter | Avoided entirely |
| `scatter_add_` | Same | Avoided entirely |
| `nn.Embedding.backward` | Uses scatter_add_ | Binary Embedding (matmul only) |
| `F.one_hot` | Uses scatter_ internally | `==` broadcast comparison |
| `torch.gather.backward` | Uses scatter_add_ | Advanced indexing or broadcast mask |
| `torch.max(dim=).backward` | Uses scatter_ | `.detach()` (safe for softmax) |
| `aten::_thnn_fused_gru_cell` | Not implemented in DirectML | DML_GRUCell (manual gates) |
| `aten::sigmoid` | AMD driver crash on some dims | StableSigmoid (tanh-based) |
| `aten::lerp` | Not implemented | Avoided (use manual interpolation) |
| `nn.GRU(bidirectional)` | Uses fused_gru_cell | Not used in V5 |

### Verified Safe Operations Used in V5

| Operation | Used In |
|---|---|
| `nn.Linear` (matmul) | Everything — backbone of all components |
| `F.scaled_dot_product_attention` | Resonance Block Path A |
| `torch.fft.rfft` / `torch.fft.irfft` | Resonance Block Path B |
| `F.avg_pool1d` | Elastic Context compression |
| `torch.cat` / `torch.stack` | Context assembly |
| `F.gelu` / `F.silu` | Activations |
| `RMSNorm` (manual) | Normalization (x * rsqrt(mean(x²))) |
| `F.softmax` (small dim) | Tool selection head (8-way) |
| `chunked_softmax` (large dim) | Vocabulary projection if used |
| Sinusoidal positional encoding | Buffer, no backward needed |
| `==` broadcast comparison | Binary embedding bit extraction |
| Bitwise `>>` and `& 1` | Binary encoding (integer ops, no backward) |

### Test Protocol
Before any new operation enters the codebase:
```python
# test_dml_op.py — run on RX 7600 before committing
x = torch.randn(2, 128, 1024, device="privateuseone:1", requires_grad=True)
y = NEW_OPERATION(x)
loss = y.sum()
loss.backward()  # If this crashes → operation is banned
print("✓ Safe for DirectML")
```

---

## Summary

V5 "RESONANCE" is a **Living Model** — it doesn't just run, it grows.

| Dimension | V4 (best previous) | V5 |
|---|---|---|
| Architecture | SSM + ODE + Pointer-Gen | Spectral Resonance + Action Decoder |
| Param utilization | 23% trainable | **100% trainable** |
| Context window | 256 tokens | **8,192 tokens** |
| Learning after deploy | None (frozen) | **Continuous (dual-speed)** |
| Memory | None | **Episodic (ChromaDB)** |
| Self-modification | None | **Code Introspector** |
| Self-improvement | None | **Curriculum generator** |
| Multi-user scaling | None | **Federated trace sharing** |
| Training paradigm | Next-token prediction | **Next-action prediction** |
| DirectML dead weight | 154M frozen params | **0 frozen params** |
| Self-awareness | None | **Reads and modifies own code** |

**The model that gets smarter every day you use it — and smarter for everyone who uses it.**
