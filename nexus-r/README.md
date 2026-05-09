# NEXUS-R

NEXUS-R is the active research workspace for a consumer-GPU recursive language
model. The current implementation lives in `nexus_v1/` and experiments with a
split architecture: an anchor stream encodes the prompt once, while a thought
stream revisits that frozen context through recursive reasoning steps.

## Current Focus

The code in `nexus_v1/` is centered on three questions:

- Can a small model use repeated cross-attention over a frozen prompt instead
    of standard self-attention-only decoding?
- Can that recursive path stay trainable on 4 GB class GPUs?
- Does the architecture learn useful reasoning traces on real supervision such
    as GSM8K, not just synthetic toy prompts?

## Active Architecture

`nexus_v1/architecture/` defines the current model stack:

- `layers.py`: shared low-level components such as RMSNorm, RoPE, SwiGLU, and
    grouped-query attention helpers.
- `dual_stream_mla.py`: the anchor/thought cross-attention module.
- `recursive_block.py`: the weight-tied recursive reasoning loop with halting
    diagnostics and diversity regularization.
- `nexus_r.py`: top-level model assembly, generation, and small builder
    configs.

The runtime path is:

`tokens -> embedding -> anchor encoder -> frozen K/V -> recursive thought updates -> LM head`

## Training Entry Points

The main training scripts are in `nexus_v1/training/`.

- `train_bpe.py`: current BPE trainer. Supports restricted-vocab and full GPT-2
    BPE runs, answer-only masking, EMA checkpoints, and dataset profiles such as
    `gsm8k`, `diverse_qa`, and `mixed`.
- `train_v1.py`: earlier character-level validation trainer kept for simpler
    debugging runs.
- `train_improved.py`: older experimental trainer with WSD scheduling and BPE
    utilities.

Typical launch from this folder:

```bash
cd nexus-r
python -m nexus_v1.training.train_bpe
```

Useful environment variables:

- `NEXUS_DATA_PROFILE=gsm8k|diverse_qa|mixed`
- `NEXUS_TRAIN_PROFILE=baseline|rtx2050-expanded|rtx2050-restricted-wide`
- `NEXUS_VOCAB_MODE=restricted|full`

## Data Workflow

The current real-data path is GSM8K.

- Prepare local files from the repo root:

```bash
python data/gsm8k/prepare_gsm8k.py
```

- This writes `data/gsm8k/train.jsonl` and `data/gsm8k/eval.jsonl` in the same
    `instruction/output` format used by the trainer.

Synthetic and mixed-data experiments still exist under `data/diverse_qa/`.

## Evaluation

Checkpoint evaluation scripts live beside the trainers:

- `eval_suite_runner.py`: full checkpoint evaluation over a JSONL suite.
- `eval_greedy_only.py`: faster greedy-only pass for quick comparisons.

Example:

```bash
cd nexus-r
python -m nexus_v1.training.eval_suite_runner --checkpoint nexus_v1/checkpoints/nexus_r_v1_best.pt
```

## Repository Layout

```text
nexus-r/
├── legacy/               # older architecture generations and archived experiments
├── nexus_v1/             # active model code
│   ├── architecture/     # recursive model components
│   ├── training/         # trainers and evaluation runners
│   ├── tests/            # smoke tests
│   ├── tokenizer.py      # tokenizer wrapper utilities
│   └── scheduler.py      # scheduler experiments
├── logs/                 # run logs and debugging output
└── README.md             # this file
```

## Notes

- This is a research repository. Interfaces change as training issues and
    architectural failures are investigated.
- The `legacy/` tree is kept for reference, but current work should target the
    active `nexus_v1/` code.
