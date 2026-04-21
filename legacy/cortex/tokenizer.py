"""BPE Tokenizer — train from data or load existing.

Uses HuggingFace tokenizers library (fast Rust implementation).
"""

import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train_tokenizer(
    data_paths: list[str | Path],
    vocab_size: int = 8192,
    save_path: str | Path = 'cortex/tokenizer.json',
) -> Tokenizer:
    """Train a BPE tokenizer from JSONL data files.

    Args:
        data_paths: list of .jsonl files with 'instruction' and 'output' fields
        vocab_size: target vocabulary size
        save_path: where to save the trained tokenizer

    Returns:
        Trained Tokenizer instance
    """
    # Collect all text
    texts = []
    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'instruction' in item:
                    texts.append(item['instruction'])
                if 'output' in item:
                    texts.append(item['output'])

    # Build BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processor for BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f'[BOS] $A [EOS]',
        pair=f'[BOS] $A [EOS] $B:1 [EOS]:1',
        special_tokens=[
            ('[BOS]', tokenizer.token_to_id('[BOS]')),
            ('[EOS]', tokenizer.token_to_id('[EOS]')),
        ],
    )

    tokenizer.enable_padding(pad_id=PAD_ID, pad_token='[PAD]')

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f'Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})')

    return tokenizer


def load_tokenizer(path: str | Path = 'cortex/tokenizer.json') -> Tokenizer:
    """Load a pre-trained tokenizer."""
    return Tokenizer.from_file(str(path))
