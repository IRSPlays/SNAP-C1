"""
Fractal Recurrent Core (FRC)
============================
The core reasoning engine of SNAP-C1 V2.
This module defines the self-recurrent neural loop that trades parameter 
count for time-depth (Test-Time Compute in latent space). 
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class HaltGate(nn.Module):
    """
    Evaluates the entropy/certainty of the latent state to determine
    if the Recurrent Core should stop looping.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # A tiny MLP to evaluate the latent vector
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a scalar probability [0, 1] representing the model's
        confidence to halt the recurrent loop.
        """
        # Average pooling across sequence length (assuming x is [batch, seq_len, dim])
        pooled = x.mean(dim=1) 
        logits = self.net(pooled)
        return torch.sigmoid(logits)

class SwiGLUBlock(nn.Module):
    """A standard dense block used inside the Recurrent Core."""
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LatentRecurrentBlock(nn.Module):
    """
    The main reasoning block that gets looped recursively.
    """
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        # In a full model, this would be an Attention or SSM layer
        # For the pure logic core mapping, we use a deep dense mixer.
        self.mixer = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = SwiGLUBlock(hidden_dim, ffn_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections inside the block
        x_mixed = x + self.mixer(self.norm1(x))
        out = x_mixed + self.ffn(self.norm2(x_mixed))
        return out

class FractalRecurrentCore(nn.Module):
    """
    The orchestrator that runs the recurrent loop and monitors the Halt Gate.
    """
    def __init__(
        self, 
        hidden_dim: int = 1024, 
        ffn_dim: int = 4096, 
        num_core_layers: int = 12,
        max_loops: int = 100
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_loops = max_loops
        self.halt_threshold = 0.95  # 95% confidence to halt
        
        # The block of layers that will be recursively executed
        self.layers = nn.ModuleList([
            LatentRecurrentBlock(hidden_dim, ffn_dim) for _ in range(num_core_layers)
        ])
        
        self.halt_gate = HaltGate(hidden_dim)
        self.final_norm = nn.RMSNorm(hidden_dim)
        
    def forward(self, hologram_state: torch.Tensor) -> tuple[torch.Tensor, int, list[float]]:
        """
        Executes the recursive reasoning loop.
        
        Args:
            hologram_state: The compressed input context mapping. Shape [batch, seq_len, dim]
            
        Returns:
            final_state: The processed latent vector ready for output/Concept generation
            loops_taken: Number of recurrent cycles executed
            halt_probs: Record of confidence scores over time for monitoring
        """
        z = hologram_state
        halt_probs = []
        
        for k in range(self.max_loops):
            # Pass through the core layers
            for layer in self.layers:
                z = layer(z)
                
            # Evaluate whether we have reached mathematical certainty
            halt_prob = self.halt_gate(z).item()
            halt_probs.append(halt_prob)
            
            if halt_prob > self.halt_threshold:
                logger.debug(f"FRC reached certainty {halt_prob:.2f} at loop {k+1}. Halting.")
                break
                
        if len(halt_probs) == self.max_loops:
            logger.warning(f"FRC hit max loops ({self.max_loops}) without reaching certainty.")
            
        final_state = self.final_norm(z)
        
        return final_state, k + 1, halt_probs

if __name__ == "__main__":
    # Test the basic mechanics
    batch_size = 1
    seq_len = 32
    dim = 1024
    
    # Mock "Hologram" output from the memory compression phase
    mock_input = torch.randn(batch_size, seq_len, dim)
    
    # Initialize the core
    core = FractalRecurrentCore(hidden_dim=dim, max_loops=50)
    
    # Run the recursive thought process
    print("Initiating Latent Thought Loop...")
    final_output, loops, confidences = core(mock_input)
    
    print(f"Thought completed in {loops} loops.")
    print(f"Final confidence sequence: {[round(c, 2) for c in confidences]}")
    print(f"Output shape: {final_output.shape}")
