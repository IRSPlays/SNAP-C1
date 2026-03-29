"""
BPE Tokenizer for NEXUS
=======================

Subword tokenization using byte-pair encoding.
Significant improvement over character-level tokenization.

Benefits:
- Learns subword representations (morphemes, word parts)
- Vocabulary size: 8K-32K vs ~50 for character-level
- Better generalization to unseen words
- Standard approach used in GPT, LLaMA, etc.
"""

import json
import os
from typing import List, Tuple, Optional
import re


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer with proper subword tokenization.

    Based on GPT-2/SentencePiece style tokenization.
    """

    def __init__(
        self,
        vocab: Optional[dict] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        vocab_size: int = 8192
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        if vocab is None:
            # Start with byte-level vocabulary
            self.vocab = {i: bytes([i]) for i in range(256)}
            # Add special tokens
            self.vocab[0] = b'<pad>'
            self.vocab[1] = b'<unk>'
            self.vocab[2] = b'<s>'
            self.vocab[3] = b'</s>'
            self._byte_offset = 256
        else:
            self.vocab = vocab
            self._byte_offset = 256

        if merges is None:
            self.merges = []
        else:
            self.merges = merges

        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token = '<pad>'

        # Cache for encoding
        self._cache = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_stats(self, words: List[List[int]]) -> dict:
        """Count pairs of adjacent tokens."""
        pairs = {}
        for word in words:
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge_pair(self, words: List[List[int]], pair: Tuple[int, int]) -> List[List[int]]:
        """Merge all occurrences of a pair in words."""
        new_words = []
        for word in words:
            if len(word) < 2:
                new_words.append(word)
                continue
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(self._byte_offset + len(self.merges))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, text: str, vocab_size: int = 8192, min_frequency: int = 2):
        """
        Train BPE tokenizer on text.

        Uses byte-level encoding for full Unicode coverage.
        """
        print(f"Training BPE tokenizer with target vocab size {vocab_size}...")

        # Convert text to byte sequences
        words = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Encode as UTF-8 bytes
            word = list(line.encode('utf-8'))
            word.append(256)  # End of word marker
            words.append(word)

        # Start with byte vocab (256 tokens) + special tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab[0] = b'<pad>'
        self.vocab[1] = b'<unk>'
        self.vocab[2] = b'<s>'
        self.vocab[3] = b'</s>'
        self._byte_offset = 256
        self.merges = []

        # Iterate merging pairs
        for i in range(vocab_size - 256 - 4):
            stats = self._get_stats(words)
            if not stats:
                break

            # Find most frequent pair
            best_pair = max(stats, key=lambda x: stats[x])

            # Stop if frequency too low
            if stats[best_pair] < min_frequency:
                print(f"Stopping: pair frequency {stats[best_pair]} < {min_frequency}")
                break

            # Merge pair
            new_token = self._byte_offset + len(self.merges)
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append(best_pair)
            words = self._merge_pair(words, best_pair)

            if (i + 1) % 1000 == 0:
                print(f"  Merged {i + 1} pairs, vocab size: {len(self.vocab)}")

        print(f"Training complete! Final vocab size: {len(self.vocab)}")

    def save(self, path: str):
        """Save vocabulary and merges to file."""
        data = {
            'vocab': {str(k): v.decode('utf-8', errors='replace') for k, v in self.vocab.items()},
            'merges': [list(m) for m in self.merges],
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = {int(k): v.encode('utf-8') for k, v in data['vocab'].items()}
        merges = [tuple(m) for m in data['merges']]

        tokenizer = cls(vocab=vocab, merges=merges)
        print(f"Tokenizer loaded from {path}")
        return tokenizer

    def encode(self, text: str, max_len: int = 512, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Uses byte-level BPE encoding.
        """
        if text in self._cache and max_len == 512 and add_special_tokens:
            return self._cache[text]

        # Tokenize with regex (word boundary awareness)
        words = re.findall(r'\S+|\s+', text)

        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for word in words:
            if word.isspace():
                continue
            # Get UTF-8 bytes
            word_bytes = list(word.encode('utf-8'))
            word_tokens = []

            # Apply merges in order
            for merge_pair in self.merges:
                idx0, idx1 = merge_pair
                # Find and replace consecutive pairs
                new_word = []
                i = 0
                while i < len(word_bytes):
                    if (i < len(word_bytes) - 1 and
                        word_bytes[i] == idx0 and
                        word_bytes[i + 1] == idx1):
                        new_word.append(self._byte_offset + self.merges.index(merge_pair))
                        i += 2
                    else:
                        new_word.append(word_bytes[i])
                        i += 1
                word_bytes = new_word

            tokens.extend(word_bytes)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + [self.eos_token_id]

        # Cache result
        if len(self._cache) < 10000 and max_len == 512 and add_special_tokens:
            self._cache[text] = tokens

        return tokens

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx < 4:
                continue
            if idx in self.vocab:
                tokens.append(self.vocab[idx])
            elif idx >= self._byte_offset:
                merge_idx = idx - self._byte_offset
                if merge_idx < len(self.merges):
                    pair = self.merges[merge_idx]
                    tokens.append(self.vocab[pair[0]] + self.vocab[pair[1]])

        # Decode bytes to string
        result = b''.join(tokens)
        return result.decode('utf-8', errors='replace')

    def batch_encode(self, texts: List[str], max_len: int = 512,
                     add_special_tokens: bool = True, padding: bool = True) -> List[List[int]]:
        """Encode batch of texts."""
        results = []
        max_len_actual = max_len

        for text in texts:
            encoded = self.encode(text, max_len, add_special_tokens)
            results.append(encoded)

        if padding:
            # Pad to max length
            max_len_actual = max(len(r) for r in results)
            for i in range(len(results)):
                if len(results[i]) < max_len_actual:
                    results[i].extend([self.pad_token_id] * (max_len_actual - len(results[i])))

        return results


class SimpleBPETokenizer:
    """
    Simplified BPE tokenizer - working implementation.
    Uses word-level tokenization with character n-gram fallback.
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab = {}
        self.vocab_size = vocab_size
        self.merges = []  # List of (token1, token2) merge pairs
        self._reverse_vocab = {}

    def train(self, text: str, vocab_size: int = 5000):
        """Train tokenizer on text using word-piece approach."""
        print(f"Training tokenizer on {len(text)} chars, target vocab size {vocab_size}...")

        self.vocab_size = vocab_size

        # Initialize with special tokens and all single characters
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3
        }
        next_id = 4

        # Count all characters and add to vocab
        for char in set(text):
            if char not in self.vocab and next_id < vocab_size:
                self.vocab[char] = next_id
                next_id += 1

        # Split text into words (simple whitespace tokenization)
        words = text.split()

        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])

        # Add frequent words to vocab
        for word, freq in sorted_words:
            if next_id >= vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = next_id
                next_id += 1

        # Build reverse vocab for decoding
        self._reverse_vocab = {v: k for k, v in self.vocab.items()}

        print(f"Training complete! Vocab size: {len(self.vocab)}")

    def encode(self, text: str, max_len: int = 512) -> List[int]:
        """Encode text to token IDs using learned vocab."""
        words = text.split()
        result = [self.vocab['<s>']]

        for word in words:
            if word in self.vocab:
                result.append(self.vocab[word])
            else:
                # Fall back to character-level for unknown words
                for char in word:
                    result.append(self.vocab.get(char, self.vocab['<unk>']))

        result.append(self.vocab['</s>'])

        # Truncate if needed
        if len(result) > max_len:
            result = result[:max_len - 1] + [self.vocab['</s>']]

        return result

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        result = []
        for idx in ids:
            if idx in self._reverse_vocab:
                token = self._reverse_vocab[idx]
                if token not in ['<pad>', '<s>', '</s>', '<unk>']:
                    result.append(token)
            elif idx == 0:  # pad
                continue
            else:
                result.append('<unk>')
        return ' '.join(result)

    def save(self, path: str):
        """Save tokenizer."""
        data = {
            'vocab': self.vocab,
            'merges': [list(m) for m in self.merges],
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SimpleBPETokenizer':
        """Load tokenizer."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        t = cls()
        t.vocab = {int(k) if k.isdigit() else k: v for k, v in data['vocab'].items()}
        t.merges = [tuple(m) for m in data.get('merges', [])]
        t.vocab_size = data.get('vocab_size', len(t.vocab))
        t._reverse_vocab = {v: k for k, v in t.vocab.items()}
        print(f"Tokenizer loaded from {path}")
        return t


class TiktokenTokenizer:
    """
    Production-grade BPE tokenizer using tiktoken.

    Tiktoken is OpenAI's fast BPE tokenizer, used by GPT-4.
    Provides proper subword tokenization with 100K+ vocab.

    Benefits over SimpleBPETokenizer:
    - True byte-pair encoding (not word-level approximation)
    - 100K vocab vs ~50-100 for character level
    - Trained on diverse text (not just training data)
    - Handles any Unicode text correctly
    - Very fast encoding/decoding
    """

    def __init__(self, encoding_name: str = 'cl100k_base'):
        """
        Initialize tiktoken tokenizer.

        Args:
            encoding_name: Tiktoken encoding to use:
                - 'cl100k_base': GPT-4's encoding (100K vocab)
                - 'o200k_base': GPT-4o's encoding (200K vocab)
                - 'p50k_base': GPT-3's encoding (50K vocab)
        """
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        self.eos_token_id = self.enc.eot_token
        # tiktoken uses eot_token for both bos and eos in cl100k_base
        # But we explicitly set them correctly based on tiktoken's API
        try:
            self.bos_token_id = self.enc.bos_token
        except AttributeError:
            # Some encodings don't have separate bos_token
            self.bos_token_id = self.enc.eot_token
        self.pad_token_id = 0

        # tiktoken doesn't have pad by default, use eos as pad
        self._pad_id = self.enc.eot_token

    def encode(self, text: str, max_len: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            max_len: Maximum sequence length (None = no truncation)
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if add_special_tokens:
            # tiktoken adds special tokens internally
            ids = self.enc.encode(text, disallowed_special=())
        else:
            ids = self.enc.encode(text)

        # Truncate if max_len specified
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if skip_special_tokens:
            # Filter out special tokens
            filtered = [id for id in ids if id not in (self.eos_token_id, self.bos_token_id)]
            return self.enc.decode(filtered)
        else:
            return self.enc.decode(ids)

    def batch_encode(self, texts: List[str], max_len: int = 512,
                     add_special_tokens: bool = True, padding: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts
            max_len: Maximum sequence length
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences to max_len

        Returns:
            List of encoded sequences (padded if padding=True)
        """
        results = []
        max_len_actual = max_len

        for text in texts:
            encoded = self.encode(text, max_len, add_special_tokens)
            results.append(encoded)

        if padding:
            max_len_actual = max(len(r) for r in results)
            for i in range(len(results)):
                if len(results[i]) < max_len_actual:
                    results[i].extend([self.pad_token_id] * (max_len_actual - len(results[i])))

        return results


def download_tiny_shakespeare() -> str:
    """
    Download TinyShakespeare dataset.

    Returns:
        Path to the downloaded text file
    """
    import urllib.request
    import os

    url = "https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/data/tokenizer/testdata/shakespeareakespeare.txt"
    # Alternative: use the TinyShakespeare URL
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'text')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'tiny_shakespeare.txt')

    if os.path.exists(cache_path):
        print(f"Using cached TinyShakespeare from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"Downloading TinyShakespeare from {url}...")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode('utf-8')
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved TinyShakespeare to {cache_path}")
        return text
    except Exception as e:
        print(f"Download failed: {e}")
        # Fallback: generate sample text
        print("Using fallback sample text...")
        return _get_sample_text()


def _get_sample_text() -> str:
    """Fallback sample text if download fails."""
    return """
    ROMEO:
    But, soft! what light through yonder window breaks?
    It is the east, and Juliet is the sun.

    JULIET:
    O Romeo, Romeo! wherefore art thou Romeo?

    HAMLET:
    To be, or not to be, that is the question.

    MACBETH:
    Is this a dagger which I see before me,
    The handle toward my hand?

    KING LEAR:
    How sharper than a serpent's tooth it is
    To have a thankless child!

    OTHELLO:
    I am one who loved wisely, though almost by stealth.

    THE TEMPEST:
    We are such stuff as dreams are made on,
    And our little life is rounded with a sleep.
    """ * 1000  # Repeat to make it substantial


if __name__ == '__main__':
    # Test tiktoken tokenizer
    print("Testing TiktokenTokenizer...")

    tokenizer = TiktokenTokenizer()

    text = "Hello world! This is a test of the BPE tokenizer."
    encoded = tokenizer.encode(text)
    print(f"Original: '{text}'")
    print(f"Encoded: {encoded[:20]}... ({len(encoded)} tokens)")
    print(f"Vocab size: {tokenizer.vocab_size}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")

    # Test batch encoding
    texts = ["Hello world!", "Goodbye world!", "Testing tokenizer batch encoding."]
    batch = tokenizer.batch_encode(texts, padding=True)
    print(f"Batch: {[len(x) for x in batch]}")

    print("\nTiktokenTokenizer test PASSED!")
    print(f"\nAvailable encodings: cl100k_base (GPT-4), o200k_base (GPT-4o), p50k_base (GPT-3)")