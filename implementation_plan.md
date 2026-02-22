# SNAP-C1 V3 Implementation Plan: Generative Reasoning Engine

This document outlines the systematic codebase overhaul required to transition the project from the V2 Subprocess Sandbox (RLFS) to the V3 Offline Reasoning Architecture.

## Proposed Changes

### 1. Data Processing & Tokenization Paradigm Shift
We must abandon flat string tokenization (`tiktoken`) in favor of mathematical graph serialization.

#### [NEW] `v3_core/data/ast_parser.py`
- Implements the bridging layer to parse standard Python/Multi-language code into standardized JSON/Tensor represented Abstract Syntax Trees (ASTs).
- Creates specialized Dataloaders that yield graph-node tensors instead of 1D arrays of BPE tokens.

#### [NEW] `v3_core/data/trace_simulator.py`
- A utility script that takes valid dataset code, executes it natively, and records the `locals()` and `globals()` variable states at every line.
- Constructs the new `[CODE] ... [MEM] ...` interleaved training strings.

---

### 2. Neuromorphic Decoder Rewrite (AST & Trace Support)
The decoder must stop predicting single alphabetic characters and start generating logic rules.

#### [MODIFY] [v2_core/architecture/neuromorphic_decoder.py](file:///c:/Users/haziq/Documents/RX.AI/v2_core/architecture/neuromorphic_decoder.py) (Renamed to `v3`)
- **Action:** Replace the standard linear `lm_head` with a **Graph Neural Network (GNN)** prediction head capable of branching child nodes (`ASTDecoder`).
- **Trace Support:** Add secondary cross-attention heads specifically trained to decode the `[MEM]` state representations interleaved between code nodes.

---

### 3. Core Architecture: Liquid Time-Constants (LTC)

#### [MODIFY] [v2_core/architecture/recurrent_core.py](file:///c:/Users/haziq/Documents/RX.AI/v2_core/architecture/recurrent_core.py) (Renamed to `v3`)
- **Action:** Remove the fixed `max_loops` integer and the discrete [HaltGate](file:///c:/Users/haziq/Documents/RX.AI/v2_core/architecture/recurrent_core.py#15-38).
- **Implementation:** Introduce `LiquidTimeBlock` modules based on ODE boundary solvers to allow continuous-time feature processing. The input context mathematically "flows" down the recurrent gradient until it reaches thermodynamic equilibrium (the answer).

---

### 4. V3 RLFS Offline Evaluator (No Subprocesses)

#### [NEW] `v3_core/training/rlfs/ast_evaluator.py`
- Replaces [sandbox.py](file:///c:/Users/haziq/Documents/RX.AI/v2_core/rlfs_sandbox.py).
- **Action:** Parses the decoder's AST output natively in PyTorch memory. Compares the structural Tree Edit Distance against known valid dataset ASTs. Outputs a fast continuous `-Float` penalty directly to the loss graph without touching the underlying Windows OS.

#### [NEW] `v3_core/training/rlfs/adversarial_loop.py`
- **Action:** Implements the Bi-Directional Verifier. Houses both the [Generator](file:///c:/Users/haziq/Documents/RX.AI/v2_core/generate_logic_dataset.py#20-143) and `Verifier` neural blocks, routing the outputs between them as competing PyTorch loss functions.

---

## Verification Plan

### Automated Tests
- Run `pytest` on `v3_core/data/ast_parser.py` to ensure it successfully converts complex Python code into unbroken mathematical tensors and back again perfectly.
- Run `v3_core/training/rlfs/ast_evaluator.py` against syntactically invalid output to verify it accurately outputs massive negative penalty gradients in < 0.05 seconds locally (compared to 1.0s OS timeouts).

### Manual Verification
- Execute a mini-batch training loop utilizing the RX 7600 GPU. Watch the CPU utilization: if it remains low while VRAM usage stays high, the V3 mathematical off-loading has successfully bypassed the OS threading bottlenecks.
