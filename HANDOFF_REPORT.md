# SNAP-C1 v3: Progress Report & Future Architecture
**Date**: February 21, 2026
**Current Focus**: Migration to 8GB VRAM (AMD RX 7600) & AGI Architecture
**Project Lead**: Agent Antigravity & USER (Haziq)

---

## 🚀 Executive Summary
The SNAP-C1 project has fundamentally evolved from a raw prompt-engineering exercise on a 1.7B model to a **Continuous Self-Evolution Architecture (MoLoRA)** designed for AGI-level abstraction. 

We have hit the ceiling of what a 1.7B parameter model (Qwen 1.7B) can achieve zero-shot in an open-ended sandbox. The model exhibits "The Generalization Wall"—when placed in an unstructured environment (`curiosity_engine.py`), it falls back to reciting training data scripts rather than authentic reasoning.

**The Strategic Pivot**: We are officially migrating the project to target an **8GB VRAM AMD RX 7600** environment. Instead of trying to "pre-train" a weak model from scratch, we will perform a **"Brain Transplant"**, utilizing an 8-Billion parameter class model (e.g., Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct) quantized to 4-bit. This larger brain possesses the requisite world knowledge to utilize the advanced AGI scaffolding we have built.

---

## 🏗️ What Was Built In This Session (The AGI Scaffolding)
We have successfully constructed the infrastructure for a continuously learning agent. These scripts are functional but require the smarter 8B model weights to execute effectively.

### 1. `inference/flow_controller.py` (The Reproducer Loop)
Replaces standard generation. Forces the model into a strict "Test-Time Compute" flow to solve SWE-bench issues:
1.  **Gather Context**: AST tools to read files.
2.  **Reproduce**: *Crucial Step*. The model must write a standalone `reproduce.py` script that throws the exact error mentioned in the GitHub issue.
3.  **Fix & Verify**: Propose patches until `reproduce.py` runs cleanly.

### 2. `training/dpo_collector.py` (Failure-Contrastive Learning)
The heart of our self-evolution. When the `FlowController` runs, the model inevitably makes mistakes. When it finally succeeds, the `DPOCollector` instantly slices the execution log:
-   **Rejected**: The flawed reasoning and patch that failed the `reproduce.py` test.
-   **Chosen**: The successful reasoning and patch that fixed it.
-   *Result*: Automated `(prompt, chosen, rejected)` pairs for real-time finetuning.

### 3. `training/infinite_loop.py` (The AGI Engine)
An orchestration script designed to run 24/7.
1.  Throws the model at SWE-bench issues via `FlowController`.
2.  Harvests DPO pairs via `DPOCollector`.
3.  **Real-Time Hot Swap**: Dynamically switches the model from Inference to `trl.DPOTrainer`. Performs an online gradient update on the `team_thinking` MoLoRA adapter using the exact mistake it just made, then hot-swaps back to inference for the next issue.

### 4. `inference/curiosity_engine.py` (Phase 8 - Open Exploration)
When idle, the model is placed in an isolated file sandbox.
-   It is prompted to brainstorm an unknown topic/tool.
-   It has access to `<tool_call>` to write and execute code (e.g., write `scraper_tool.py`).
-   Once it validates a new tool, it executes `<store_skill>`, which dynamically injects the implementation into our **ChromaDB vector store** (`memory_manager.py`). 
-   *Status*: Working script, but the 1.7B model hallucinates the execution. Awaits 8B model upgrade.

---

## 🗺️ Architectural Roadmap for the Next Agent

### Immediate Next Steps (The AMD Migration)
The next agent taking over MUST prioritize the hardware migration before executing the loops.

1.  **ROCm / DirectML Setup (Critical)**:
    -   The target hardware is an AMD RX 7600 (8GB VRAM) on Windows.
    -   Standard `pip install torch` will install the CUDA version, which will fall back to the CPU (extremely slow) on an AMD card.
    -   *Action*: The agent must configure the environment to use PyTorch for ROCm (if on WSL/Linux) or PyTorch-DirectML (if on native Windows) to ensure the 8B model fits in VRAM and runs on the GPU.
2.  **Base Model "Brain Transplant"**:
    -   Update `config/base_model.yaml` to point to a high-reasoning 8B target.
    -   *Recommendation*: `meta-llama/Meta-Llama-3.1-8B-Instruct` (Quantized to 4-bit config).
3.  **Adapter Retraining (Epoch Zero)**:
    -   The existing `adapters/` directory contains LoRA weights tuned for Qwen 1.7B. These are fundamentally incompatible with Llama-3/Qwen-7B.
    -   *Action*: Run `training/train_lora.py` with the existing `train_v3_agentic.jsonl` dataset to map the `[Architect]/[Critic]` deliberative debate format onto the new 8B model.

### Long-Term AGI Vision
Once the 8B model is online and fine-tuned for debate:
1.  **Start `infinite_loop.py`**: Connect the script to the Dockerized SWE-bench Verified dataset. Let the model train itself 24/7 using Failure-Contrastive DPO. Monitor the loss curves.
2.  **Activate `curiosity_engine.py`**: In parallel to SWE-bench, let the model explore the internet and build a massive library of dynamic tools stored in ChromaDB, expanding its action space infinitely beyond standard shell commands.
3.  **Model Merging / Distillation**: Eventually, freeze the best MoLoRA adapters and merge them into the base 8B model to establish a new, permanently smarter baseline model for SNAP-C2.
