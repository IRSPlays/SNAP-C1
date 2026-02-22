# SNAP-C1 V4 (Hyper-Routing Reasoner)
## Architectural Blueprint & Theoretical Foundations

*The V3 Architecture mathematically solved the problem of Logical Hallucination by eliminating string-token prediction in favor of natively decoding Continuous Abstract Syntax Tree (AST) Graphs. V4 is the master synthesis: scaling the flawless logic of V3 to handle infinite context and arbitrary repositories (SWE-Bench) using the Micro-MoE (Mixture of Experts) sub-systems proven in V2.*

---

### The Problem: The Vocabulary Barrier
V3's `semantic_classifier` successfully maps geometrical AST nodes (like `FunctionDef` or `Assign`) to dynamic strings (like `check_even` or `var_a`). However, it hits a hard mathematical wall when trying to memorize the entire PyPI ecosystem (Pandas, Django, Requests). Scaling the final generic Linear Layer to millions of outputs instantly crashes an 8GB VRAM GPU.

### The V4 Solution: The Sub-Word BPE Auto-Regressive Head
Instead of classifying one massive variable name out of an unwieldy dictionary matrix:
1. V4 retains the **Continuous AST Geometric graph** to guarantee flawless syntax.
2. When the Graph Decoder dictates a variable name is required (e.g. `ast.Name`), V4 routes the localized context vector to a miniature **Byte-Pair Encoding (BPE)** Language Model.
3. This tiny LLM auto-regressively spells out the variable name in sub-tokens (e.g., `[Http] + [Response] + [Redirect]`), allowing it to dynamically generate *infinite* un-seen class names and external dependencies while keeping the vocabulary vector extremely small and efficient.

#### The Zero-Shot Semantic Gap (Pointer-Generator Matrix)
As correctly identified, BPE alone only shifts the memory constraint into a *training* constraint. If the model has never been trained on the string `requests.get()`, the BPE head won't know to sequentially generate those specific sub-tokens from its latent weights.
**Solution: The Copy Mechanism**
Instead of forcing the BPE head to guess novel vocabulary purely from its pre-trained state, Phase 13 will implement a **Pointer-Generator Network**. When V4 is fed context (like a retrieved SWE-Bench Python file), the BPE head calculates a probability distribution over the *injected context tokens* alongside its own vocabulary. This allows the model to dynamically "point to" and copy external dependencies (like `HttpResponseRedirect`) directly from the user's prompt into the AST Graph, achieving true zero-shot reasoning on completely un-seen codebases.

---

### The Problem: The Context Window Barrier
To fix a bug in SWE-Bench, the model needs to understand ten entirely distinct Python files simultaneously to trace the faulty variable back to its parent class. V3's `HolographicCompressor` (Mamba) cannot mathematically squeeze 10,000 lines of complex Python logic into a single 1024-dimension VRAM vector without losing catastrophic amounts of resolution.

### The V4 Solution: SSD-Streamed Micro-MoE
We bring back the crowning jewel of V2: The **Softmax Router**.
1. **The Context Database:** We instantiate a local Vector Database (like ChromaDB). The target GitHub repository is completely ingested, chunked, and embedded into this database.
2. **The Discovery Head:** When the user provides a prompt ("Fix the `KeyError` in `models.py`"), V4 uses a tiny query-expert network to physically search the Vector Database for the 3 most mathematically relevant context blocks across the entire repository.
3. **The Micro-MoE Architecture:** V4 loads *only* the specific Mixture-of-Experts weights required to solve the target sub-problem (e.g. streaming the `Django_SQL_Expert.safetensors` matrix directly from the NVMe SSD into VRAM cache).
4. **The Synthesizer:** The highly localized, extremely relevant context is fed into the Liquid Time-Constant (LTC) ODE solvers, bypassing the context-window ceiling entirely.

---

### Phase 13: Preparing the V4 Infrastructure
To start building V4, we need to begin isolating the V3 Graph Decoder and preparing the Context Retrieval mechanisms.

1. **Implement `v4_core/data/bpe_tokenizer.py`:** A hybrid wrapper that utilizes OpenAI's `tiktoken` purely for the semantic variable generation head (abandoning the hardcoded `1000`-item integer dictionary).
2. **Implement `v4_core/memory/chroma_indexer.py`:** An offline script that walks through massive local codebases, chunks the Python syntax structures smartly, and stores them in a dense vector database mapped for offline cosine-similarity queries.
3. **Draft the `V4HyperAssembly` component:** The master PyTorch module that queries the `chroma_indexer`, routes to the `SSDStreamer`, feeds the Liquid ODE logic, and maps the output back to `bpe_tokenizer.py` + `ast.unparse()`.
