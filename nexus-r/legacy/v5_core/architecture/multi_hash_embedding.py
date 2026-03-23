"""
SNAP-C1 V5: Multi-Hash Embedding (Component 1)
================================================
Scatter-free trainable embedding using K=8 prime-modulo hash tables.
Replaces nn.Embedding which is frozen on DirectML due to scatter_add_ in backward.

Design rationale:
- BPE token IDs are assigned by merge order, NOT semantics
- Binary bit decomposition creates arbitrary Hamming distances between similar tokens
- Multi-Hash uses K independent prime moduli → each token gets a unique fingerprint
- Collision in one hash table is resolved by the other K-1 tables (Bloom filter analogy)
- Lookup via broadcast == (no scatter) → matmul with parameter table → 100% DirectML safe

Parameters: ~3.5M (vs 154M frozen in V4). 100% trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_core.utils.dml_ops import RMSNorm


class MultiHashEmbedding(nn.Module):
    """
    K independent hash functions map token IDs to small buckets.
    Each bucket has a learnable vector. K vectors are concatenated and fused.

    Args:
        d_model: Output embedding dimension (default 1024)
        K: Number of independent hash functions (default 8)
        d_hash: Dimension per hash table lookup (default 128)
        primes: List of K coprime moduli for hashing. Chosen to minimize
                collision overlap. Default covers range from 251 to 8191.
    """

    # Default primes — coprime to each other, spread across magnitudes
    DEFAULT_PRIMES = [251, 509, 1021, 2039, 4093, 8191, 997, 1999]

    def __init__(self, d_model: int = 1024, K: int = 8, d_hash: int = 128,
                 primes: list = None):
        super().__init__()
        self.primes = primes or self.DEFAULT_PRIMES[:K]
        assert len(self.primes) == K, f"Need exactly {K} primes, got {len(self.primes)}"
        self.K = K
        self.d_hash = d_hash
        self.d_model = d_model

        # Each hash function has its own small embedding table
        # Stored as nn.Parameter (matrix), NOT nn.Embedding (no scatter in backward)
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(p, d_hash) * 0.02)
            for p in self.primes
        ])

        # Fusion: K * d_hash → d_model
        self.fusion = nn.Linear(K * d_hash, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] long tensor of BPE token IDs

        Returns:
            [B, T, d_model] float tensor — embedded representations
        """
        parts = []
        for k in range(self.K):
            prime = self.primes[k]
            bucket = token_ids % prime  # [B, T] — which bucket in table k

            # Scatter-free lookup via broadcast == comparison
            # Create one-hot: [B, T, prime_k]
            # torch.no_grad() on one_hot — it's derived from integer inputs,
            # doesn't need gradients. Saves ~300MB of backward activation memory.
            arange = torch.arange(prime, device=token_ids.device)  # [prime_k]
            with torch.no_grad():
                one_hot = (arange == bucket.unsqueeze(-1)).float()  # [B, T, prime_k]

            # Matmul lookup: [B, T, prime_k] @ [prime_k, d_hash] → [B, T, d_hash]
            # Grad still flows into self.tables[k] through the matmul
            part = one_hot @ self.tables[k]
            parts.append(part)

        # Concatenate all K views → [B, T, K * d_hash]
        x = torch.cat(parts, dim=-1)

        # Fuse to model dimension and normalize
        return self.norm(self.fusion(x))  # [B, T, d_model]

    def extra_repr(self) -> str:
        total = sum(p * self.d_hash for p in self.primes) + self.K * self.d_hash * self.d_model
        return (f"K={self.K}, d_hash={self.d_hash}, d_model={self.d_model}, "
                f"primes={self.primes}, total_params={total:,}")
