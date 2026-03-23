"""
SNAP-C1 V6 Training Module
"""

from v6_core.training.v6_trainer import V6Trainer, TrainingConfig, AliveProtocolLoss
from v6_core.training.v6_agent_loop import V6AgentLoop, AgentConfig, ToolResult, run_v6_agent

__all__ = [
    'V6Trainer', 'TrainingConfig', 'AliveProtocolLoss',
    'V6AgentLoop', 'AgentConfig', 'ToolResult', 'run_v6_agent',
]
