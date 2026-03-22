"""
SNAP-C1 V6 Core Architecture
============================
All components for V6 WHORMHOLE.
"""

from v6_core.architecture.dml_ops import (
    RMSNorm, SwiGLU, stable_sigmoid, chunked_softmax,
    DML_GRUCell, get_device
)
from v6_core.architecture.holographic_embedding import HolographicEmbedding
from v6_core.architecture.resonance_block import (
    ResonanceBlock, ResonanceStack,
)
from v6_core.architecture.advanced_attention import (
    SlidingWindowGQA, DeltaNetAttention, MultiTokenPrediction,
    QKNorm, GroupedQueryAttention, HybridAttention
)
from v6_core.architecture.elastic_context import ElasticContext
from v6_core.architecture.observation_encoder import ObservationEncoder, SegmentType
from v6_core.architecture.action_decoder import ActionDecoder, ToolID
from v6_core.architecture.outcome_predictor import OutcomePredictor
from v6_core.architecture.self_verification import (
    SelfVerificationLoop, VerifiedActionDecoder, VerificationHead
)
from v6_core.architecture.state_space_hopper import (
    StateSpaceHopper, StateHopper, StateMemory, AssociativeMemory
)
from v6_core.architecture.plastic_weights import (
    PlasticLinear, PlasticAttention, PlasticBlock, PlasticStack,
    EligibilityTrace, convert_to_plastic
)
from v6_core.architecture.tool_melting import (
    ToolMeltingEngine, ToolSynthesizer, MeltedTool, PrimitiveRegistry,
    ToolMeltingWrapper, PRIMITIVES
)
from v6_core.architecture.v6_assembly import (
    V6ResonanceModel, build_v6_local, build_v6_small,
    build_v6_rtx6000, build_v6_large
)
from v6_core.architecture.nexus_v6 import (
    NexusV6, NexusBlock,
    build_nexus_small, build_nexus_medium, build_nexus_large,
    # Novel components
    EntanglementMixer, LatentConceptExpert, DepthAdaptiveGate,
    SelfEvolvingHebbianLayer, AdaptiveMambaAttentionHybrid,
    EvolutionaryPooling, MambaSSM,
    # NEW: 2025-2026 research components
    TreeGuidedEvolution, WSDFunction, WSDTrainer
)

__all__ = [
    # DML ops
    'RMSNorm', 'SwiGLU', 'stable_sigmoid', 'chunked_softmax',
    'DML_GRUCell', 'get_device',
    # Embedding
    'HolographicEmbedding',
    # Resonance
    'ResonanceBlock', 'ResonanceStack',
    # Advanced Attention
    'SlidingWindowGQA', 'DeltaNetAttention', 'MultiTokenPrediction',
    'QKNorm', 'GroupedQueryAttention', 'HybridAttention',
    # Context
    'ElasticContext',
    # Encoder
    'ObservationEncoder', 'SegmentType',
    # Decoder
    'ActionDecoder', 'ToolID',
    # Predictor
    'OutcomePredictor',
    # Self-Verification
    'SelfVerificationLoop', 'VerifiedActionDecoder', 'VerificationHead',
    # State Space Hopper
    'StateSpaceHopper', 'StateHopper', 'StateMemory', 'AssociativeMemory',
    # Plastic Weights (Hebbian learning during inference!)
    'PlasticLinear', 'PlasticAttention', 'PlasticBlock', 'PlasticStack',
    'EligibilityTrace', 'convert_to_plastic',
    # Tool Melting
    'ToolMeltingEngine', 'ToolSynthesizer', 'MeltedTool', 'PrimitiveRegistry',
    'ToolMeltingWrapper', 'PRIMITIVES',
    # Assembly
    'V6ResonanceModel', 'build_v6_local', 'build_v6_small',
    'build_v6_rtx6000', 'build_v6_large',
    # NEXUS - Novel Architecture
    'NexusV6', 'NexusBlock',
    'build_nexus_small', 'build_nexus_medium', 'build_nexus_large',
    'EntanglementMixer', 'LatentConceptExpert', 'DepthAdaptiveGate',
    'SelfEvolvingHebbianLayer', 'AdaptiveMambaAttentionHybrid',
    'EvolutionaryPooling', 'MambaSSM',
    # NEW: 2025-2026 research components
    'TreeGuidedEvolution', 'WSDFunction', 'WSDTrainer',
]
