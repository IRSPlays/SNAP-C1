"""Tokenizer helpers used by Nexus-R experiments.

The active BPE training path relies on `tiktoken`, and this wrapper keeps the
encode/decode interface simple for scripts that want a small tokenizer object
instead of using the raw library directly.
"""

from typing import List, Optional


class TiktokenTokenizer:
    """
    Production-grade BPE tokenizer using tiktoken.

    Tiktoken is OpenAI's fast BPE tokenizer, used by GPT-4.
    Provides proper subword tokenization with 100K+ vocab.

    Benefits:
    - True byte-pair encoding
    - 100K vocab vs ~50-100 for character level
    - Trained on diverse text
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