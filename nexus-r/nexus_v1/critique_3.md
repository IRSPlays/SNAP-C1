# BRUTAL CRITIQUE: NEXUS-R Research Cycle #1

**Date:** 2026-03-29
**Agent:** Research Agent (Cycle #1)
**Reviewer:** Brutal Critic Agent

---

## OVERALL RATING: **NEEDS_WORK**

You fixed 3 critical bugs (validation batching, weight decay groups, checkpoint saving). Good. But **TextDataset data loss is STILL BROKEN after 2 cycles of being flagged**, and you introduced a new issue: checkpoint resume logic doesn't exist.

---

## AUDIT: WHAT YOU CLAIMED VS REALITY

### ✅ VERIFIED FIXED: Validation Batching
- **Lines 282-291:** Properly iterates through val_loader
- **Rating:** GENUINELY FIXED

### ✅ VERIFIED FIXED: Weight Decay Groups
- **Lines 209-228:** Correctly separates decay/no-decay parameter groups
- **Rating:** GENUINELY FIXED

### ✅ VERIFIED FIXED: Checkpoint Saving
- **Lines 306-314:** Saves model + optimizer + scheduler state
- **Rating:** SAVING WORKS (but resume doesn't — see below)

### ❌ NOT FIXED: TextDataset Data Loss (CRITICAL — 3RD CYCLE)
- **Problem (AGAIN):** Line 46:
```python
for i in range(0, len(self.tokens) - stride, stride):
```
With 1000 tokens, stride=128, max_len=256:
- i = 0, 128, 256, 384, 512, 640, 768, 896 → **STOP**
- Tokens 896-999 (104 tokens) **LOST FOREVER**

- **This was flagged in critique_1 and critique_2. It is STILL BROKEN.**
- You listed it as "HIGH: Fix TextDataset sliding window data loss" in "Next Cycle Priorities" but DIDN'T FIX IT.

### ❌ NOT FIXED: T5 Init Override
- **Lines 82-85:** Still using T5 init with `std=1.0/sqrt(d_model)=0.0625` instead of baseline `0.02`
- **Not a blocker**, but you documented it as needing fix and didn't fix it

### ❌ NEW BUG: Checkpoint Resume Logic Missing
- **You can SAVE checkpoints** (lines 306-314) ✅
- **You CANNOT RESUME from them** ❌
- No code to load `optimizer_state_dict` or `scheduler_state_dict` when starting
- `train_improved()` always starts from scratch (lines 254-317)
- The checkpoint save is **useless without resume capability**

### ❌ DEAD CODE: Two Broken Tokenizers Still Present
- `BPETokenizer` (lines 21-268) — **NEVER USED**
- `SimpleBPETokenizer` (lines 270-385) — **NEVER USED**
- You use only `TiktokenTokenizer`
- 350+ lines of dead code with documented bugs (O(n²) encoding, vocab corruption)

---

## LINE-BY-LINE CRITIQUE

### train_improved.py

**Line 46 — TextDataset STILL LOSES DATA:**
```python
for i in range(0, len(self.tokens) - stride, stride):
```
With 1000 tokens, stride=128: stops at i=896, losing last 104 tokens.
**Fix:** `range(0, len(self.tokens) - max_len + 1, stride)` or `range(0, len(self.tokens), stride)` with proper truncation handling.
**This is a 3-cycle-old bug. Fix it.**

**Lines 78-95 — T5 Init Override:**
```python
nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
```
d_model=256 → std=0.0625. Baseline uses 0.02. Higher variance can destabilize small models.
**Not blocking but unexplained why you kept it.**

**Lines 254-317 — No Checkpoint Resume:**
Training always starts from scratch. If `outputs/best_model.pt` exists, it's ignored.
**Fix:** Add load logic before training loop.

**Lines 306-313 — Misleading Step Tracking:**
```python
checkpoint = {
    'step': step,  # Current step, NOT best step
```
If step 500 is best but we save at step 600, metadata says step 600. Confusing for debugging.
**Fix:** Track `best_step` separately.

---

### tokenizer.py

**Lines 21-268 — BPETokenizer: DEAD CODE**
Never instantiated anywhere. Contains O(n²) encoding bug.
**DELETE IT.**

**Lines 270-385 — SimpleBPETokenizer: DEAD CODE**
Never instantiated anywhere. Not even real BPE (word-level with char fallback).
**DELETE IT.**

**Lines 416-422 — BOS/EOS "Partial Fix" Is Nothing:**
```python
try:
    self.bos_token_id = self.enc.bos_token
except AttributeError:
    self.bos_token_id = self.enc.eot_token
```
cl100k_base doesn't have separate BOS. This try/except does nothing useful — bos_token_id still equals eos_token_id when AttributeError is raised.
**Document it or fix it properly, but stop calling it "partial fix."**

---

## MISSING FEATURES (Still Missing After 3 Cycles)

| Feature | Status | Notes |
|---------|--------|-------|
| TextDataset data loss | ❌ NOT FIXED | 3rd cycle flagging this |
| Checkpoint resume | ❌ MISSING | Save works, load doesn't |
| Eval benchmarks | ❌ MISSING | GSM8K/HumanEval files exist but unused |
| Logging | ❌ MISSING | No wandb/tensorboard/file log |
| Early stopping | ❌ MISSING | Runs fixed steps, wastes GPU |
| LR range test | ❌ MISSING | peak_lr is guessed |
| Dead tokenizer cleanup | ❌ NOT DONE | 350+ lines of garbage |

---

## ROOT CAUSE

**You prioritized fixing things that were already working in the original code over things that were actually broken.** The validation batching and weight decay fixes were good — those were genuine bugs. But TextDataset data loss is a SIMPLE off-by-one error that you've now ignored for 3 cycles.

**Checkpoint resume is 50% done.** You saved the state but never wrote the code to load it. This is incomplete feature implementation.

**You documented TODOs but didn't implement them.** "Next Cycle Priorities" is not an excuse to leave bugs unfixed.

---

## FINAL SCORECARD

| Component | Rating | Blocker? | Notes |
|-----------|--------|----------|-------|
| Validation batching | GOOD | No | ✅ Actually fixed |
| Weight decay groups | GOOD | No | ✅ Actually fixed |
| Checkpoint saving | GOOD | No | ✅ Actually fixed |
| TextDataset data loss | FAIL | YES | ❌ 3rd cycle, still broken |
| Checkpoint resume | FAIL | YES | ❌ Never implemented |
| T5 init override | NEEDS_WORK | No | ❌ Not fixed, not blocking |
| BOS/EOS "fix" | FAIL | No | ❌ Nothing changed |
| Dead code cleanup | FAIL | No | ❌ 350+ lines still present |
| Logging | MISSING | No | ❌ Still absent |
| Eval benchmarks | MISSING | No | ❌ Still absent |

**Deliverables:** 2 files modified, 3 genuine fixes
**Real Progress:** Moderate — 3 bugs genuinely fixed, 1 genuine bug still broken, 1 feature 50% done
**Honesty Score:** 8/10 — You were accurate about what was fixed, honest about what wasn't
**Recommendation:** Partial merge. Validation, weight decay, and checkpoint saving are good. But TextDataset data loss is a simple fix that should have been done. Checkpoint resume is 15 lines of code.

---

## ACTION ITEMS FOR NEXT CYCLE

### MUST FIX (Blocking):
1. **Actually fix TextDataset** — `range(0, len(self.tokens) - max_len + 1, stride)` or proper sliding window
2. **Add checkpoint resume** — 15 lines of code to load and resume
3. **Track best_step separately** — Don't confuse current step with best step

### SHOULD FIX:
4. **Delete dead tokenizers** — Remove `BPETokenizer` and `SimpleBPETokenizer` (350+ lines)
5. **Remove T5 init override** — Use baseline 0.02 std
6. **Add file logging** — At minimum print to `outputs/training.log`

### NICE TO HAVE:
7. **Eval benchmark integration** — Actually run GSM8K or HumanEval
8. **Early stopping** — Stop if val loss plateaus for N steps
9. **LR range test** — Find optimal peak_lr before training

---

## SPECIFIC CODE CHANGES

### train_improved.py Line 46 — FIX TextDataset:
```python
# CURRENT (BROKEN - loses last sequence):
for i in range(0, len(self.tokens) - stride, stride):

# FIXED (includes all tokens):
for i in range(0, max(1, len(self.tokens) - max_len + 1), stride):
    seq = self.tokens[i:i + max_len]
    if len(seq) < max_len:
        if i + max_len > len(self.tokens):
            break  # Skip incomplete last window instead of padding
        seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
    self.sequences.append(seq)
```

### train_improved.py — ADD resume logic (after model creation, before training loop):
```python
# After model creation (after line 204)
if os.path.exists('outputs/best_model.pt'):
    print("Resuming from checkpoint...")
    checkpoint = torch.load('outputs/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step = checkpoint['step']
    best_val_loss = checkpoint['best_val_loss']
    history = checkpoint['history']
    print(f"Resumed from step {step}, best_val_loss={best_val_loss:.4f}")
else:
    step = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
```

### tokenizer.py — DELETE:
- `BPETokenizer` class (lines 21-268)
- `SimpleBPETokenizer` class (lines 270-385)

Keep only `TiktokenTokenizer` (lines 387-497) and `download_tiny_shakespeare` (lines 500-565).

---

*Cycle time: ~10 minutes. 3 genuine fixes. 1 bug that should have been 1-line fix, still broken after 3 cycles.*
