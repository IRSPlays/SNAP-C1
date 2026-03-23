import torch
import torch.nn as nn
from loguru import logger
import sys
import os
from pathlib import Path

# Add project root explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import V2 Legacy Retained Elements (We keep these!)
from v2_core.architecture.holographic_compressor import HolographicCompressor

# Import New V3 Elements
from v3_core.architecture.recurrent_core import ContinuousRecurrentCore
from v3_core.architecture.ast_decoder import ASTDecoder
from v3_core.training.rlfs.ast_evaluator import ASTRewardEvaluator

class V3GenerativeReasoningArchitecture(nn.Module):
    """
    SNAP-C1 V3: Master Assembly File
    
    This stitches together the retained V2 Holographic Compressor 
    with the brand new V3 Liquid Time-Constant (LTC) logical core
    and the Abstract Syntax Tree (AST) Graph Decoder.
    """
    def __init__(self, d_model: int = 1024, ast_vocab_size: int = 250):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. RETAINED FROM V2: Mamba 1D Token Compressor
        # We still need to read raw text strings from the user (e.g. "Write me a function...")
        self.compressor = HolographicCompressor(d_model=d_model, state_size=64)
        
        # 2. BRAND NEW V3: Liquid Time-Constant (LTC) Core Engine
        # Replaces fixed `max_loops=15`. Dynamically solves differential equations
        # until logical equilibrium is mathematically reached.
        logger.info("Initializing Liquid Time-Constant (LTC) continuous architecture...")
        self.core = ContinuousRecurrentCore(hidden_dim=d_model, epsilon=1e-3, max_sim_time=50)
        
        # 3. BRAND NEW V3: AST Graph Neural Decoder
        # Replaces raw Python subprocess character generation (which caused execution errors).
        # Generates structurally perfect AST Graph Neural structures.
        logger.info("Initializing AST Graph Neural Network Decoder...")
        self.ast_decoder = ASTDecoder(concept_dim=d_model, ast_vocab_size=ast_vocab_size, hidden_dim=512, semantic_vocab_size=1000)
        
        # Formal Evaluator logic routes locally in loss loop
        self.reward_engine = ASTRewardEvaluator()
        
    def forward(self, input_ids: torch.Tensor, max_nodes: int = 50) -> torch.Tuple[torch.Tensor, torch.Tensor, int]:
        """
        End-to-end forward pass of the V3 Engine.
        """
        # 1. Compress prompt
        hologram_sequence = self.compressor(input_ids)
        
        # 2. Fluid continuous-time thinking
        equilibrium_vector, time_steps_taken = self.core(hologram_sequence)
        
        # 3. Generate AST Math graph
        ast_nodes, branch_probs, _, semantic_logits = self.ast_decoder(equilibrium_vector, max_nodes=max_nodes)
        
        return ast_nodes, branch_probs, time_steps_taken, semantic_logits

if __name__ == "__main__":
    print("\n=== Assembling the FULL SNAP-C1 V3 Architecture ===")
    
    model = V3GenerativeReasoningArchitecture()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Simulate a human prompt tokens (e.g. from cl100k_base)
    dummy_input = torch.randint(0, 100277, (1, 64)).to(device)
    
    ast_nodes, branch_probs, continuous_thoughts, semantic_logits = model(dummy_input, max_nodes=20)
    
    print(f"\nAssembly Validated Successfully.")
    print(f"Liquid Core Equilibrium Output Reached in {continuous_thoughts} fluid steps.")
    print(f"Final V3 AI Output shape (Graph Nodes): {ast_nodes.shape}")
    print(f"Final V3 AI Output shape (Branching Tree Weights): {branch_probs.shape}")
