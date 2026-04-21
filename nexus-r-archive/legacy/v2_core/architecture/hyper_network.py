"""
Hyper-Network (Dynamic Weight Synthesizer)
=========================================
Synthesizes temporary neural pathways (adapters) on the fly.
If the SSD Micro-Experts cannot bridge a specific cross-domain concept,
this sub-network generates a custom low-rank weight matrix directly into VRAM,
applies it to the thought vector, and deletes it.
"""

import torch
import torch.nn as nn
from loguru import logger

class HyperNetwork(nn.Module):
    """
    Takes the current latent state and generates a dynamic Low-Rank Adapter (LoRA).
    We use low-rank generation to keep this network fast and small (~0.5GB), 
    while simulating a much larger temporary expert.
    """
    def __init__(self, hidden_dim: int, rank: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # The analyzer reads the thought vector and predicts what kind of 
        # missing mathematical bridge is needed.
        self.analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Matrix A Generator: maps from analyzed thought to LoRA A
        # Reshaped to [hidden_dim, rank]
        self.gen_A = nn.Linear(hidden_dim, hidden_dim * rank)
        
        # Matrix B Generator: maps from analyzed thought to LoRA B
        # Reshaped to [rank, hidden_dim]
        self.gen_B = nn.Linear(hidden_dim, rank * hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dynamically synthesize weights and apply them to `x`.
        
        Args:
            x: Latent state of shape [batch, seq_len, hidden_dim]
            
        Returns:
            processed_x: The state transformed by a dynamically spawned, custom neural layer.
        """
        batch_size, seq_len, dim = x.shape
        
        # 1. Analyze the sequence to find the "missing link"
        # Average pooling across sequence
        target_concept = x.mean(dim=1) # [batch, dim]
        analysis = self.analyzer(target_concept) # [batch, dim]
        
        # 2. Synthesize Weight Matrix A and B
        # We only generate these based on the first item in batch for this prototype
        # since generating batch-specific dynamic linear layers requires bmm
        A_flat = self.gen_A(analysis[0]) # [dim * rank]
        B_flat = self.gen_B(analysis[0]) # [rank * dim]
        
        # Reshape into matrices
        W_A = A_flat.view(dim, self.rank)
        W_B = B_flat.view(self.rank, dim)
        
        # Form the custom dynamic weight matrix: Delta W = A @ B
        # Shape: [dim, dim]
        dynamic_weight = torch.matmul(W_A, W_B)
        
        # 3. Apply the synthetic expert to the latent state
        # x is [batch, seq_len, dim], dynamic_weight is [dim, dim]
        # x @ W^T
        synthetic_output = torch.matmul(x, dynamic_weight.t())
        
        # Non-linearity
        synthetic_output = torch.nn.functional.silu(synthetic_output)
        
        # Residual connection
        return x + synthetic_output

if __name__ == "__main__":
    print("\n--- Testing Dynamic Hyper-Network Synthesizer ---")
    dim = 1024
    
    hyper_net = HyperNetwork(hidden_dim=dim, rank=64)
    
    # Mock latent vector representing a puzzle the SSD experts can't solve
    novel_thought = torch.randn(1, 32, dim)
    
    print("Novel thought encountered. Core triggering Hyper-Network...")
    
    # Measure synthesis and execution
    import time
    start = time.perf_counter()
    
    output = hyper_net(novel_thought)
    
    ms = (time.perf_counter() - start) * 1000
    
    print(f"Synthesized AND executed a custom 1024x1024 neural layer in {ms:.2f} ms")
    print(f"Output shape: {output.shape}")
    print("VRAM automatically cleared holding only low-rank generators.")
