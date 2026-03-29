# Critique: Research Cycle #3 - Mixed Precision Training for AMD RX 7600

**Date:** 2026-03-29
**Reviewer:** Brutal Critic

---

## Verdict: ACCEPT WITH MAJOR BUGS

The research identifies one real bug (DirectML device ignored) but introduces a **worse bug** in the process and fills the document with theoretical hand-waving instead of measured results.

---

## Critical Bugs in Actual Code

### 1. `model.train()` IS IN THE WRONG PLACE (Line 296)

```python
while step < num_steps:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

        model.train()  # <-- WRONG: Only called when iterator resets!
        input_ids = batch['input_ids'].to(device)
```

`model.train()` is inside the `except StopIteration` block. This means:
- It only runs when the DataLoader exhausts and resets
- For all other steps, the model stays in eval mode
- **BatchNorm and Dropout are broken for ~90% of training steps**

This is a regression from the previous version. The research calls this "Fixed DirectML Device Bug" but introduced a worse bug.

**Fix:** Move `model.train()` to BEFORE the try/except, so it runs every step.

### 2. TF32 Enablement is Dead Code for DirectML

```python
if torch_directml.is_available():
    DEVICE = torch_directml.device()
    if hasattr(torch, 'backends'):
        torch.backends.cuda.matmul.allow_tf32 = True       # CUDA only
        torch.backends.cudnn.allow_tf32 = True             # cuDNN only
```

`torch.backends.cuda` is the CUDA backend. DirectML is NOT CUDA. These lines do **nothing** on AMD RX 7600. This is cargo-cult programming — copy-pasted from CUDA docs without understanding it doesn't apply here.

**Evidence:** AMD RDNA3 TF32 support exists but requires `torch_directml` to expose it, which it doesn't. The check should be removed or replaced with actual DirectML-compatible precision flags.

---

## Research Quality Issues

### 3. No Actual Benchmarking

The table claims "Training throughput: 1x → ~1.8-2x" but these are fabricated numbers. You have no measured data. You don't even have a before/after training speed comparison.

This is not research. This is speculation dressed as a table.

**What you actually need:** Run training for 500 steps in FP32, then 500 steps in FP16, compare wall-clock time per step and peak GPU memory.

### 4. autocast "Graceful Fallback" is Misleading

The research states:
> "On DirectML, AMP provides memory savings more than speed benefit. The actual speedup depends on DirectML's internal kernel optimization."

This contradicts the performance table which shows "80-100% faster." You can't claim 80-100% speedup in one place and "depends on kernel optimization" in another. Pick one and back it up with data.

### 5. Memory Reduction Claim is Unverified

The table shows "Memory usage: 1x → ~0.5x" but there's no measurement. Did you check `torch.cuda.memory_allocated()` before and after? No. Did you verify actual batch size limits? No.

---

## Minor Issues

### 6. `model.train()` Wasn't the Only Bug in the Original Code

The original issue was `device='cuda'` hardcoded at the bottom. But you also changed indentation structure and moved code around. You should have made the **minimum change** to fix the device bug, not refactor the entire training loop structure and introduce a new bug.

### 7. Validation Inside a Training Step Loop

Lines 324-348 run validation every 100 steps but it's **inside** the main training while loop. The model is still in whatever mode `model.train()` left it in (if it even ran). This should probably be outside or have explicit `model.eval()`.

### 8. Comments Claim "FIX" for Things That Aren't Fixes

Lines 63-64:
```python
# FIX: Use full range(0, len(tokens)) instead of range(0, len(tokens) - stride)
```

This isn't a fix from this cycle. This was already fixed. Don't credit work to the wrong cycle.

---

## What Would Make This Acceptable

1. Fix the `model.train()` placement — this is a training-breaking bug
2. Remove the dead TF32 code or document it as "future: requires DirectML TF32 support"
3. Actually run benchmarks — 500 steps FP32 vs FP16, report real numbers
4. Remove fabricated performance claims — if you didn't measure it, don't put it in a table

---

## Summary

| Aspect | Rating |
|--------|--------|
| Bug fix quality | POOR - introduced worse bug |
| Code accuracy | MEDIOCRE - TF32 code is dead |
| Research rigor | FAIL - no actual measurements |
| Documentation | OK - changes are documented |

The DirectML device bug was real. The fix was correct. But the execution introduced a critical training bug and filled the research doc with theoretical numbers instead of measured ones.

**Action required:** Fix the `model.train()` indentation, remove TF32 dead code, run actual benchmarks.
