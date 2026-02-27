# SNAP-C1 v2 Implementation Plan: The "Functional AGI" Engine

**Goal:** Outperform 100x larger models (e.g., Claude 3.5 Sonnet) on complex reasoning and coding tasks.
**Philosophy:** "Depth over Width." Use inference-time compute (recursive thought loops) to simulate intelligence.
**Architecture:** Single-Model Recursive Agent with Episodic Memory & Continuous Self-Evolution.

---

## 🧠 Phase 1: The "System 2" Recursive Core (The Brain)
**Objective:** Replace linear generation with a dynamic, self-directed thought loop. The model stops "guessing" and starts "thinking."

### 1. The `ThoughtController` (New Engine)
*   **File:** `inference/thought_controller.py`
*   **Mechanism:**
    *   Implements a state machine: `Idle -> Thinking -> Researching -> Critiquing -> Answering`.
    *   **Token Triggers:**
        *   `<think>...</think>`: Internal monologue (invisible to user).
        *   `<research>query</research>`: Pauses generation -> Runs Web/Doc Search -> Injects result -> Resumes.
        *   `<code_sandbox>code</code_sandbox>`: Pauses -> Runs code -> Injects stdout/stderr -> Resumes.
        *   `<memory_store>fact</memory_store>`: Saves key insight to Long-Term Memory.
        *   `<final_answer>...</final_answer>`: The definitive output.
*   **"Tree Search" (MCTS-Lite):**
    *   For critical decisions, the Controller generates **3 parallel thought paths**.
    *   It uses a lightweight **Critic Head** (or self-reflection) to score them.
    *   It prunes the weak paths and continues with the best one.

### 2. The "Universal Mind" Prompt
*   **File:** `config/prompts/recursive_system.yaml`
*   **Logic:**
    *   Deprecate individual persona prompts.
    *   New System Prompt: *"You are a recursive intelligence. Do not answer immediately. Debate internally. If uncertain, research. If you write code, test it mentally or in the sandbox first."*
    *   Forces the model to output a **Thought Program** before the answer.

---

## 🧠 Phase 2: The "Hippocampus" (Episodic Memory & Research)
**Objective:** Give the model a permanent, searchable memory of its own experiences.

### 1. Active Memory Store (ChromaDB Integration)
*   **File:** `memory/memory_manager.py` (Enhance)
*   **Functionality:**
    *   **Store:** `add_experience(query, solution, outcome, feedback)`.
    *   **Retrieve:** `recall_relevant(query)` -> Returns top-3 similar past problems/solutions.
*   **Trigger:**
    *   **Success:** If a code solution runs without error, automatically save: "Solution to X is Y".
    *   **Failure:** If it fails, save: "Approach Z failed for problem X because...".

### 2. Autonomous Research Loop
*   **File:** `inference/tools/researcher.py` (New)
*   **Logic:**
    *   When the model encounters a gap (e.g., "I don't know the React 19 API"), it triggers `<research>React 19 hooks</research>`.
    *   The system:
        1.  Searches the web/docs.
        2.  Summarizes the top 3 results.
        3.  **Injects the summary into the context window.**
        4.  **Saves the summary to Memory** for future zero-shot retrieval.

---

## 🧠 Phase 3: The "Dojo" (Self-Evolution & DPO)
**Objective:** The system improves its own weights (or memory) over time.

### 1. The "Experience Collector"
*   **File:** `training/experience_collector.py` (New)
*   **Logic:**
    *   Logs every interaction: `(Prompt, Thought, Action, Result, Feedback)`.
    *   **Auto-Labeling:**
        *   Did the code run? (Binary Reward)
        *   Did the user accept the answer? (Human Reward)
        *   Was the thought process coherent? (Model-based Critique)
    *   High-quality pairs are saved to `data/self_improving/buffer.jsonl`.

### 2. Episodic Fine-Tuning (The "Nightly Build")
*   **File:** `training/auto_finetune.py` (New)
*   **Logic:**
    *   **Trigger:** Every N successful interactions (e.g., 50).
    *   **Action:**
        1.  Load the current `team_thinking` adapter.
        2.  Mix `buffer.jsonl` (new skills) with `core_curriculum.jsonl` (replay buffer).
        3.  Run **DPO (Direct Preference Optimization)** to align the model with "winning" strategies.
        4.  Save as `team_thinking_vX`.
        5.  **Hot-Swap:** The inference pipeline reloads the new adapter automatically.

---

## 🧠 Phase 4: The Interface (God Mode)
**Objective:** Expose this power to the user.

### CLI Upgrade (`snap-cli`)
*   **Command:** `snap run "Build a React app"`
*   **Output:**
    *   `[Thinking...]` (Stream internal monologue)
    *   `[Researching: "React 19 docs"]`
    *   `[Coding: "src/App.js"]`
    *   `[Testing: "npm test"]` -> *Fail* -> `[Debugging...]` -> *Success*
    *   `[Final Answer]`
*   **Transparency:** User sees the *effort*, not just the result.

---

## 📅 Execution Roadmap

1.  **Day 1 (Core Loop):** Implement `ThoughtController` and `<research>`/`<think>` token handling. Basic "stop-and-go" generation.
2.  **Day 2 (Memory):** Meaningful `memory_manager.py` integration. Connect ChromaDB.
3.  **Day 3 (Tools & Prompts):** Upgrade `tool_executor.py` for autonomous use. Write the "Recursive Debate" system prompt.
4.  **Day 4 (Integration):** Stitch it all together. Run benchmarks. Verify it "pauses to think".
