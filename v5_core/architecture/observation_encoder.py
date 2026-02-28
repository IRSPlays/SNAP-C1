"""
SNAP-C1 V5: Observation Encoder (Component 4)
===============================================
Processes ALL model inputs: user messages, tool outputs (file contents,
terminal output, error messages, search results), and memory retrieval.

Each input segment is tagged with a TYPE embedding so the model knows
whether it's reading user text, file contents, terminal output, or past memory.
Same text means different things in different contexts:
  "ImportError: No module named 'utils'"
    As TOOL_TERM output: this is the current error to fix
    As MEMORY trace: this is how a similar error was fixed before

Pipeline:
  raw text → tokenize → MultiHashEmbedding → add TypeEmbedding
           → add positional encoding → ElasticContext → context tensor
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_core.architecture.multi_hash_embedding import MultiHashEmbedding
from v5_core.architecture.elastic_context import ElasticContext
from v5_core.utils.dml_ops import RMSNorm


class SinusoidalPosEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (buffer, no backward needed).
    Standard Transformer PE: PE(pos, 2i) = sin(pos / 10000^(2i/d)),
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).
    """

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x. x: [B, T, d_model]"""
        return x + self.pe[:, :x.shape[1], :]


# Input segment types
class SegmentType:
    USER = 0        # User message / instruction
    TOOL_FILE = 1   # File contents from READ tool
    TOOL_TERM = 2   # Terminal output from RUN tool
    ERROR = 3       # Error messages
    MEMORY = 4      # Retrieved past traces from Fast Brain
    SYSTEM = 5      # System prompts / tool descriptions

    COUNT = 6


class ObservationEncoder(nn.Module):
    """
    Encodes all model inputs into a unified context tensor.

    Args:
        d_model: Model dimension
        K: Number of hash functions for MultiHashEmbedding
        d_hash: Dimension per hash table
        max_seq_len: Maximum total sequence length (before compression)
        n_segment_types: Number of distinct input segment types
    """

    def __init__(self, d_model: int = 1024, K: int = 8, d_hash: int = 128,
                 max_seq_len: int = 8192, n_segment_types: int = SegmentType.COUNT):
        super().__init__()
        self.d_model = d_model

        # Token embedding (scatter-free)
        self.token_embed = MultiHashEmbedding(d_model=d_model, K=K, d_hash=d_hash)

        # Type embedding — tiny, 6 types × d_model, uses matmul projection (no scatter)
        # Stored as parameter matrix, looked up via broadcast ==
        self.type_table = nn.Parameter(torch.randn(n_segment_types, d_model) * 0.02)
        self.n_types = n_segment_types

        # Positional encoding (sinusoidal, buffer — no backward)
        self.pos_enc = SinusoidalPosEncoding(d_model, max_len=max_seq_len)

        # Context compression
        self.elastic = ElasticContext(d_model=d_model)

        # Final norm
        self.norm = RMSNorm(d_model)

    def _type_embed(self, type_ids: torch.Tensor) -> torch.Tensor:
        """Look up type embeddings using scatter-free broadcast ==.
        type_ids: [B, T] long tensor with values in [0, n_types)
        Returns: [B, T, d_model]
        """
        # one_hot: [B, T, n_types]
        arange = torch.arange(self.n_types, device=type_ids.device)
        one_hot = (arange == type_ids.unsqueeze(-1)).float()
        # matmul: [B, T, n_types] @ [n_types, d_model] → [B, T, d_model]
        return one_hot @ self.type_table

    def forward(self, token_ids: torch.Tensor,
                type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] — BPE token IDs
            type_ids: [B, T] — segment type for each token (SegmentType values)

        Returns:
            [B, total_slots, d_model] — compressed multi-resolution context
        """
        # Embed tokens (scatter-free)
        x = self.token_embed(token_ids)  # [B, T, d_model]

        # Add type embedding (scatter-free)
        x = x + self._type_embed(type_ids)  # [B, T, d_model]

        # Add positional encoding
        x = self.pos_enc(x)  # [B, T, d_model]

        # Compress via elastic context
        x = self.elastic(x)  # [B, total_slots, d_model]

        return self.norm(x)
