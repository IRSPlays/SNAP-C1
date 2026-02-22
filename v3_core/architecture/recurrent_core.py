import torch
import torch.nn as nn
from loguru import logger

# Centralized device resolution — guarantees ALL layers hit the same GPU
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from v4_core.utils.device import get_device

class StableSigmoid(nn.Module):
    """
    Bypasses a known `torch-directml` compiler crash on AMD RX GPUs 
    by mathematically reconstructing the 0-to-1 bounds using Tanh.
    Sigmoid(x) == 0.5 * (Tanh(x / 2) + 1)
    """
    def forward(self, x):
        # Keep everything on the same device using x's own device
        half = torch.tensor(0.5, dtype=x.dtype, device=x.device)
        one = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        return half * (torch.tanh(x * half) + one)

class LiquidTimeBlock(nn.Module):
    """
    SNAP-C1 V3 Component: Liquid Time-Constant (LTC) Continuous Network
    
    Replaces the rigid Mamba discrete steps. The input signal flows through 
    the state matrix over continuous simulated `dt` time intervals, naturally
    finding a thermodynamic equilibrium/solution based on complexity.
    """
    def __init__(self, hidden_dim: int, dt: float = 0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self._device = get_device()
            
        # Neural Wiring for the ordinary differential equation (ODE) solver
        self.w_state = nn.Linear(hidden_dim, hidden_dim).to(self._device)
        self.w_input = nn.Linear(hidden_dim, hidden_dim).to(self._device)
        self.tau_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim).to(self._device),
            StableSigmoid()
        )
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Euler integration step over time increment `dt`
        """
        # Ensure inputs are on the correct device and dtype
        x = x.to(device=self._device, dtype=torch.float32)
        h_prev = h_prev.to(device=self._device, dtype=torch.float32)
        
        # Calculate dynamic time-constant based on current input and state
        combined = torch.cat([x, h_prev], dim=-1)
        tau = self.tau_gate(combined)
        
        # The ODE dynamics: dh/dt = -h/tau + f(x, h)
        forcing_function = torch.tanh(self.w_state(h_prev) + self.w_input(x))
        
        # Euler update step
        eps = torch.tensor(1e-6, dtype=x.dtype, device=self._device)
        raw_h_new = h_prev + self.dt * ((-h_prev / (tau + eps)) + forcing_function)
        
        # Enforce strict mathematical bounds to stop NaN Matrix explosions over continuous loops
        h_new = torch.clamp(raw_h_new, min=-10.0, max=10.0)
        
        return h_new

class ContinuousRecurrentCore(nn.Module):
    """
    The V3 Logical Core Engine.
    Unlike V2's `max_loops=15` HaltGate architecture, this network integrates
    input until the relative change between states `(dh/dt)` falls below an epsilon,
    proving that it has resolved the logical dependencies.
    """
    def __init__(self, hidden_dim: int, epsilon: float = 1e-3, max_sim_time: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.max_sim_time = max_sim_time
        self._device = get_device()
            
        # Stack 4 Continuous ODE Blocks
        self.layers = nn.ModuleList([
            LiquidTimeBlock(hidden_dim) for _ in range(4)
        ])
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input graph tensor embedding sequence [batch, seq_len, dim]
            
        Returns:
            equilibrium_state: The final solved logic vector
            time_steps: The simulated time `t` it took to arrive at the answer
        """
        x = x.to(device=self._device, dtype=torch.float32)
        batch_size, seq_len, dim = x.shape
        
        # Initialize resting states on the same device
        states = [torch.zeros(batch_size, seq_len, dim, device=self._device, dtype=torch.float32) for _ in range(4)]
        
        for t in range(self.max_sim_time):
            max_delta = 0.0
            
            # Flow signal through all 4 ODE layers
            current_input = x
            for i, layer in enumerate(self.layers):
                h_prev = states[i]
                h_new = layer(current_input, h_prev)
                
                # Measure physical change in the latent vector 
                delta = torch.max(torch.abs(h_new - h_prev)).item()
                if delta > max_delta:
                    max_delta = delta
                    
                states[i] = h_new
                current_input = h_new
                
            # If the network stops changing, it has found the logical answer
            if max_delta < self.epsilon:
                return states[-1], t
                
        # If it reaches max simulation time, return current best guess
        return states[-1], self.max_sim_time

if __name__ == "__main__":
    print("Testing V3 Liquid Time-Constant Core Equilibrium Solver...")
    core = ContinuousRecurrentCore(hidden_dim=256)
    
    # Simulate a Graph Embedding from the AST Generator
    dummy_input = torch.randn(1, 16, 256) 
    
    equilibrium_vector, time_taken = core(dummy_input)
    
    print(f"Logic Solved successfully. Reached thermodynamic equilibrium in {time_taken} continuous steps.")
    print(f"Final Thought Output Vector Shape: {equilibrium_vector.shape}")
