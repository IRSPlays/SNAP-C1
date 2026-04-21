"""Nexus-R V1 Architecture — Asymmetric Recursive Language Model."""

from .nexus_r import NexusR, NexusConfig, build_nexus_tiny, build_nexus_small
from .recursive_block import RecursiveReasoner, RecursiveBlock
from .dual_stream_mla import DualStreamMLA
from .layers import RMSNorm, BlockAttnRes, Attention, SwiGLU, RotaryEmbedding

__all__ = [
    'NexusR', 'NexusConfig', 'build_nexus_tiny', 'build_nexus_small',
    'RecursiveReasoner', 'RecursiveBlock', 'DualStreamMLA',
    'RMSNorm', 'BlockAttnRes', 'Attention', 'SwiGLU', 'RotaryEmbedding',
]
