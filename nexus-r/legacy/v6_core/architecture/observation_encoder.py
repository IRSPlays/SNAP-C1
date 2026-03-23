"""
SNAP-C1 V6: Observation Encoder
===============================
Copied from V5 - encodes user text, tool outputs, error messages into tokens.
"""

import torch
import torch.nn as nn

from v6_core.architecture.holographic_embedding import HolographicEmbedding
from v6_core.architecture.elastic_context import ElasticContext


class SegmentType:
    USER = 0
    TOOL_FILE = 1
    TOOL_TERM = 2
    TOOL_ERROR = 3
    MEMORY = 4
    SYSTEM = 5
    COUNT = 6


class ObservationEncoder(nn.Module):
    """
    Encodes all input information (user messages, tool outputs, memory)
    into a unified tensor for the resonance stack.

    Components:
    1. Holographic Embedding: token IDs → vectors
    2. Type Embedding: distinguishes input sources
    3. Elastic Context: compresses long contexts
    """

    def __init__(self, d_model: int = 1024, K: int = 8, d_hash: int = 128,
                 max_seq_len: int = 8192, vocab_size: int = 100279):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding (replaces nn.Embedding - DirectML-safe)
        self.embedding = HolographicEmbedding(
            d_model=d_model, K=K, d_hash=d_hash
        )

        # Type embedding: distinguishes USER, TOOL_FILE, TOOL_TERM, etc.
        self.type_embedding = nn.Embedding(SegmentType.COUNT, d_model)

        # Elastic hierarchical context
        self.elastic = ElasticContext(
            d_model=d_model, max_seq_len=max_seq_len
        )

    def forward(self, token_ids: torch.Tensor,
                type_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] - BPE token IDs
            type_ids: [B, T] - segment type IDs (USER=0, TOOL_FILE=1, etc.)
                      If None, defaults to USER.

        Returns:
            [B, slots, d_model] - embedded + compressed context
        """
        B, T = token_ids.shape

        # Default type: USER
        if type_ids is None:
            type_ids = torch.zeros_like(token_ids)

        # Embed tokens
        embedded = self.embedding(token_ids)  # [B, T, d_model]

        # Add type embedding
        type_embedded = self.type_embedding(type_ids)  # [B, T, d_model]
        embedded = embedded + type_embedded

        # Apply elastic context compression
        context = self.elastic(embedded)  # [B, slots, d_model]

        return context

    def get_slot_count(self, seq_len: int) -> int:
        """How many context slots for a given sequence length?"""
        return self.elastic.get_slot_count(seq_len)
