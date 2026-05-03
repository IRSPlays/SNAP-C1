"""BPE tokenizer wrapper for Eidos V1.

Uses tiktoken GPT-2 BPE with optional restricted vocabulary
to keep embedding parameters manageable.
"""

from typing import List, Optional, Tuple, Dict


def get_tokenizer(vocab_size: int = 5000):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    return enc


def build_restricted_vocab(texts: List[str], enc,
                           min_count: int = 1) -> Tuple[Dict[int, int], Dict[int, int], int]:
    from collections import Counter
    counter = Counter()
    for text in texts:
        tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
        counter.update(tokens)

    active = sorted([t for t, c in counter.items() if c >= min_count])
    eot = enc.eot_token
    if eot not in active:
        active.append(eot)
    active.sort()

    bpe_to_local = {}
    local_to_bpe = {0: -1}
    for local_id, bpe_id in enumerate(active, start=1):
        bpe_to_local[bpe_id] = local_id
        local_to_bpe[local_id] = bpe_id

    vocab_size = len(active) + 1
    return bpe_to_local, local_to_bpe, vocab_size


def encode_texts(texts: List[str], enc, seq_len: int = 192,
                 bpe_to_local: Optional[Dict[int, int]] = None,
                 answer_only: bool = True):
    import torch
    eot = enc.eot_token
    eot_local = bpe_to_local.get(eot, 0) if bpe_to_local else eot
    PAD = 0

    marker_tokens = enc.encode("\nA:", allowed_special={'<|endoftext|>'})
    if bpe_to_local:
        marker_tokens = [bpe_to_local.get(t, 0) for t in marker_tokens]

    examples = []
    for text in texts:
        tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
        if bpe_to_local:
            tokens = [bpe_to_local.get(t, 0) for t in tokens]
        tokens.append(eot_local)

        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]

        real_len = len(tokens)
        n_pad = seq_len - real_len
        padded = tokens + [PAD] * (n_pad + 1)
        inp = torch.tensor(padded[:seq_len], dtype=torch.long)
        lbl = torch.tensor(padded[1:seq_len + 1], dtype=torch.long)

        if real_len < seq_len:
            lbl[real_len - 1:] = -100

        if answer_only:
            marker_len = len(marker_tokens)
            answer_start = -1
            for pos in range(len(tokens) - marker_len + 1):
                if tokens[pos:pos + marker_len] == marker_tokens:
                    answer_start = pos + marker_len
                    break
            if answer_start > 0:
                lbl[:answer_start - 1] = -100

        examples.append((inp, lbl))

    return examples
