"""
V5 Pre-training Data Loader
============================
Loads Python source files, tokenizes with tiktoken cl100k_base,
and produces next-token prediction samples.

Data sources (in priority order):
  1. --data_dir: any directory of .py files
  2. Auto-collect: scans the RX.AI project itself as seed data
  3. Can download open-source repos for more data

Each sample:
  - token_ids: [seq_len] — BPE token IDs
  - type_ids:  [seq_len] — all SegmentType.USER (0) for pre-training
  - labels:    [seq_len] — shifted token_ids (predict next token)
"""

import os
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import tiktoken
except ImportError:
    tiktoken = None


class CodeTokenizer:
    """Wraps tiktoken cl100k_base with caching."""

    def __init__(self):
        if tiktoken is None:
            raise ImportError("pip install tiktoken")
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab  # 100277
        self.eos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)


class CodeFileDataset(Dataset):
    """
    Pre-training dataset: loads .py files, tokenizes, and creates
    fixed-length samples for next-token prediction.

    Files are concatenated with <|endoftext|> separator, then chunked
    into seq_len+1 tokens (input = chunk[:-1], label = chunk[1:]).
    """

    def __init__(self, data_dirs: List[str], seq_len: int = 512,
                 max_files: int = 10000, min_file_chars: int = 100,
                 extensions: Tuple[str, ...] = ('.py',)):
        self.seq_len = seq_len
        self.tokenizer = CodeTokenizer()

        # Collect all source files
        all_files = []
        for data_dir in data_dirs:
            for root, dirs, files in os.walk(data_dir):
                # Skip hidden dirs, __pycache__, venv, node_modules
                dirs[:] = [d for d in dirs if not d.startswith('.')
                          and d not in ('__pycache__', 'venv', 'node_modules',
                                       '.git', 'chroma_db')]
                for f in files:
                    if any(f.endswith(ext) for ext in extensions):
                        fp = os.path.join(root, f)
                        all_files.append(fp)

        random.shuffle(all_files)
        all_files = all_files[:max_files]
        print(f"  Found {len(all_files)} source files")

        # Tokenize all files into one big token stream
        all_tokens = []
        files_loaded = 0
        for fp in all_files:
            try:
                text = Path(fp).read_text(encoding='utf-8', errors='ignore')
                if len(text) < min_file_chars:
                    continue
                tokens = self.tokenizer.encode(text)
                all_tokens.extend(tokens)
                all_tokens.append(self.tokenizer.eos_token)
                files_loaded += 1
            except Exception:
                continue

        print(f"  Loaded {files_loaded} files → {len(all_tokens):,} tokens")

        # Chunk into samples of seq_len+1
        self.samples = []
        chunk_size = seq_len + 1  # +1 for labels
        for i in range(0, len(all_tokens) - chunk_size, chunk_size):
            chunk = all_tokens[i:i + chunk_size]
            self.samples.append(chunk)

        # Also add overlapping samples for more data
        if len(self.samples) < 100:
            stride = max(seq_len // 2, 64)
            for i in range(stride, len(all_tokens) - chunk_size, stride):
                chunk = all_tokens[i:i + chunk_size]
                self.samples.append(chunk)

        random.shuffle(self.samples)
        print(f"  Created {len(self.samples)} training samples (seq_len={seq_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        token_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        type_ids = torch.zeros_like(token_ids)  # All SegmentType.USER for pretrain
        return token_ids, type_ids, labels


class InstructionDataset(Dataset):
    """
    Instruction-tuning dataset for agent training.

    Expects JSON/JSONL with schema:
      {"instruction": "...", "response": "...", "tool_id": 0-7}

    Or converts from V4 format:
      {"prompt": "...", "target_code": "..."}
    """

    def __init__(self, data_path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.tokenizer = CodeTokenizer()
        self.samples = []

        data_path = Path(data_path)
        if data_path.suffix == '.jsonl':
            with open(data_path, encoding='utf-8', errors='ignore') as f:
                raw = [json.loads(line) for line in f if line.strip()]
        else:
            with open(data_path, encoding='utf-8', errors='ignore') as f:
                raw = json.load(f)

        for item in raw:
            # Support multiple formats
            if 'instruction' in item and 'response' in item:
                prompt = item['instruction']
                response = item['response']
            elif 'instruction' in item and 'output' in item:
                prompt = item['instruction']
                # self_correction format: prepend initial_response
                if 'initial_response' in item:
                    response = item['initial_response'] + "\n" + item['output']
                else:
                    response = item['output']
            elif 'prompt' in item and 'target_code' in item:
                prompt = item['prompt']
                response = item['target_code']
            else:
                continue

            # Tokenize as: [PROMPT] <sep> [RESPONSE]
            prompt_tokens = self.tokenizer.encode(prompt)
            sep = self.tokenizer.encode("\n---\n")
            response_tokens = self.tokenizer.encode(response)

            combined = prompt_tokens + sep + response_tokens
            # Need seq_len+1 tokens for label shifting (like CodeFileDataset)
            if len(combined) > seq_len + 1:
                combined = combined[:seq_len + 1]

            # Pre-shift labels: labels[i] = combined[i+1] (next-token prediction)
            # forward_pretrain does NOT shift internally, so we must do it here.
            # This matches CodeFileDataset: token_ids = chunk[:-1], labels = chunk[1:]
            prompt_len = len(prompt_tokens) + len(sep)
            token_ids = combined[:-1]
            shifted_labels = list(combined[1:])

            # Mask prompt labels: only train on predicting response tokens
            # Response starts at combined[prompt_len], which appears in
            # shifted_labels at index prompt_len - 1
            for i in range(min(prompt_len - 1, len(shifted_labels))):
                shifted_labels[i] = -100

            # Pad if needed
            pad_len = seq_len - len(token_ids)
            if pad_len > 0:
                token_ids += [0] * pad_len
                shifted_labels += [-100] * pad_len

            self.samples.append({
                'token_ids': token_ids[:seq_len],
                'labels': shifted_labels[:seq_len],
                'type_id': item.get('tool_id', 0),
            })

        print(f"  Loaded {len(self.samples)} instruction samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        token_ids = torch.tensor(s['token_ids'], dtype=torch.long)
        labels = torch.tensor(s['labels'], dtype=torch.long)
        type_ids = torch.zeros_like(token_ids)
        return token_ids, type_ids, labels


def create_pretrain_loader(data_dirs: List[str], seq_len: int = 512,
                           batch_size: int = 1, **kwargs) -> Optional[DataLoader]:
    """Create pre-training DataLoader. Returns None if no data found."""
    dataset = CodeFileDataset(data_dirs, seq_len=seq_len, **kwargs)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      drop_last=True, num_workers=0, pin_memory=False)
