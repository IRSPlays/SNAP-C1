# NEXUS Research Log - Iteration 2026-03-29 (Continued)

## Improvement #1: Tiktoken BPE Tokenizer (2026-03-29)

### Problem
The SimpleBPETokenizer was NOT actually BPE - it was word-level tokenization with only ~100 vocab tokens. This was identified as the PRIMARY BOTTLENECK in previous iteration.

### Solution
Replaced with TiktokenTokenizer - OpenAI's production-grade BPE tokenizer:
- 100,277 vocab size (vs ~100 before)
- True byte-pair encoding (not word approximation)
- Pre-trained on diverse text
- Used by GPT-4

### Implementation
**File:** `tokenizer.py`

Added `TiktokenTokenizer` class:
```python
class TiktokenTokenizer:
    def __init__(self, encoding_name: str = 'cl100k_base'):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab  # 100,277
```

Also added:
- `download_tiny_shakespeare()` - Downloads 1.1M char dataset
- Fixed `max_len=None` to allow full text encoding

### Results
- Before: 1.1M chars → ~50K tokens (word-level, wrong)
- After: 1.1M chars → 301,829 tokens (proper BPE)
- Dataset: 2,358 sequences of 256 tokens each

### Files Changed
- `tokenizer.py` - Added TiktokenTokenizer class
- `training/train_improved.py` - Updated to use tiktoken + TinyShakespeare

### Verification
```
Text: 1,115,394 chars
Tokens: 301,829
Dataset: 2,358 sequences
Model: 29.6M params
Forward pass OK (loss=11.88)
Backward pass OK
Generation OK
```

### Next Steps
1. Run actual training on TinyShakespeare for 5K+ steps
2. Compare perplexity with previous run
3. Proceed to Phase 2: Working Memory

---

# NEXUS Research Log - Iteration 2026-03-29

## Research Session Summary

### Goal
Improve NEXUS V7 architecture through research-backed enhancements.

### Research Phase

#### Papers Reviewed
1. **2602.06797** - WSD Learning Rate Schedules
   - Key insight: Warmup-Stable-Decay outperforms cosine annealing
   - Implementation: Straightforward, well-documented algorithm

2. **2603.18620** - Learning to Self-Evolve
   - Key insight: Outcome-guided weight modification
   - Status: Too complex for quick implementation, deferred

3. **2603.19172** - DyMoE (Dynamic Mixture of Experts)
   - Key insight: Dynamic expert orchestration
   - Status: Requires significant architecture changes, deferred

4. **2512.24617** - DLCM (Dynamic Concept Models)
   - Key insight: Concept-level reasoning
   - Status: Related to existing LatentConceptExperts, deferred

### Implemented Improvements

#### 1. Word-Piece Tokenizer (SimpleBPETokenizer)
**File:** `tokenizer.py`

**What was done:**
- Created word-piece tokenizer (word-level with char fallback)
- Vocab built from frequent words in training text
- Supports encoding/decoding with special tokens

**Testing:**
- Training: 131 word vocab from 438K chars
- Encoding works correctly
- Round-trip decode verified

**Limitations:**
- NOT true BPE (uses word-level, not subword merging)
- Limited vocab size
- Should use HuggingFace `tokenizers` library for production

**Result:** Partial success - tokenization works but not optimal

#### 2. WSD Learning Rate Scheduler
**File:** `scheduler.py`

**What was done:**
- Implemented Warmup-Stable-Decay schedule per paper 2602.06797
- Includes WarmupCosineScheduler for comparison
- Factory function for easy scheduler creation

**Testing:**
- Verified at key steps:
  - Step 0: lr=0.000000 (start)
  - Step 50: lr=0.000500 (warmup)
  - Step 100: lr=0.001000 (peak)
  - Step 200: lr=0.001000 (stable)
  - Step 1599: lr=0.000011 (decaying)

**Result:** Success - correctly implements WSD schedule

#### 3. Improved Architecture (ImprovedNexusV7)
**File:** `training/train_improved.py`

**What was done:**
- T5-style embedding initialization
- Better linear layer initialization
- Gradient clipping

**Testing:**
- 5.9M parameter model
- Forward/backward pass OK
- No NaN gradients

**Result:** Success - architecture is sound

### Extended Training Test

**Configuration:**
- Model: 5.9M params (256 d_model, 6 layers)
- Dataset: 7 sequences from 438K chars
- Steps: 500
- Scheduler: WSD (100 warmup, 500 stable, 2000 decay)

**Results:**
- Initial loss: 5.3525
- Final loss: 2.4249
- Improvement: 54.7% reduction
- Time: ~40 seconds on CPU

### Critique

#### What Worked
1. WSD scheduler implementation was straightforward and effective
2. Architecture base (NexusV7) is solid
3. Loss decreased meaningfully (54.7%)

#### What Didn't Work / Limitations
1. **Tokenizer is bottleneck** - Word-piece is not true BPE
2. **Dataset too small** - Only 7 sequences
3. **Scale too tiny** - 5.9M params is toy model

#### Root Cause Analysis
The improvements work at the architecture level, but:
- Tokenization is MORE important than architecture
- Need proper BPE with 8K-32K vocab
- Need MB-scale data for real learning

### Next Steps

#### Immediate (Next Session)
1. Replace SimpleBPETokenizer with proper BPE using `tokenizers` library
2. Train on TinyShakespeare dataset (~1MB)
3. Increase to 10K+ steps

#### Medium Term
1. GPU training when available
2. Scale model to 30M+ params
3. Implement proper validation monitoring

#### Long Term
1. Re-add MoE components from V4 (but properly)
2. Implement self-evolution from paper 2603.18620
3. Add working memory system

### Files Changed

| File | Status | Notes |
|------|--------|-------|
| `nexus_v1/tokenizer.py` | NEW | Word-piece tokenizer |
| `nexus_v1/scheduler.py` | NEW | WSD and cosine schedulers |
| `nexus_v1/training/train_improved.py` | NEW | Training script with improvements |
| `nexus_v1/IMPROVEMENTS.md` | NEW | Critique and analysis |

### Commit Plan

**Proposed commit message:**
```
feat: Add BPE tokenizer, WSD scheduler, and improved training

- SimpleBPETokenizer: word-piece tokenization (not true BPE yet)
- WSDScheduler: implements Warmup-Stable-Decay per paper 2602.06797
- ImprovedNexusV7: T5-style initialization
- train_improved.py: full training loop with new components

Testing:
- 5.9M param model
- 54.7% loss reduction in 500 steps
- Loss: 5.35 → 2.42

Note: Tokenizer needs upgrade to proper BPE (use tokenizers library)
```

### Key Learnings

1. **Tokenization > Architecture** - A good tokenizer matters more than clever architecture changes
2. **Scale is fundamental** - 5.9M params can't learn complex patterns
3. **WSD scheduler works** - Straightforward implementation, effective results
4. **Simple is often better** - V7 simplified architecture works better than V1-V6 complex attempts

### Session Stats

- **Duration:** ~1 hour
- **Files created:** 4
- **Lines added:** ~600
- **New components:** 3 (tokenizer, scheduler, improved training)
- **Loss improvement:** 54.7%
- **Next session priority:** Proper BPE tokenizer