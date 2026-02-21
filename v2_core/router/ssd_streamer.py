"""
SSD Micro-MoE Streamer
======================
Simulates pulling 50MB "Micro-Experts" from the local NVMe SSD into VRAM
only when needed. Uses `safetensors` for zero-copy memory mapping, bypassing 
system RAM overhead and allowing an 8GB GPU to access 100GB+ of weights.
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from safetensors.torch import save_file, load_file, safe_open

EXPERTS_DIR = Path(__file__).parent.parent / "experts"
EXPERTS_DIR.mkdir(parents=True, exist_ok=True)

class MicroExpert(nn.Module):
    """
    A 50MB mathematical block representing a specific domain.
    E.g., "Python syntax expert", "Quantum physics equations".
    """
    def __init__(self, dim: int = 1024, ffn_dim: int = 4096):
        super().__init__()
        # Roughly corresponds to a single transformer FFN block
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class SSDStreamer:
    """Manages the memory-mapped streaming of Micro-Experts from NVMe to VRAM."""
    
    def __init__(self, device: str = "cpu"):
        # We test on CPU first, but in deployment this is "cuda" or "directml"
        self.device = torch.device(device)
        self.expert_files = {}
        self._scan_experts()
        
    def _scan_experts(self):
        """Index all available expert safetensors files on disk."""
        if not EXPERTS_DIR.exists():
            return
            
        for f in EXPERTS_DIR.glob("*.safetensors"):
            expert_name = f.stem
            self.expert_files[expert_name] = f
            
        logger.info(f"SSDStreamer found {len(self.expert_files)} available Micro-Experts.")
            
    def load_expert_to_vram(self, expert_name: str) -> nn.Module:
        """
        Hot-swaps an expert from SSD directly into PyTorch tensors on the target device.
        Uses Safetensors memory mapping for near-instant zero-copy loading.
        """
        if expert_name not in self.expert_files:
            raise ValueError(f"Expert {expert_name} not found on SSD.")
            
        file_path = self.expert_files[expert_name]
        
        # Instantiate an empty scaffolding
        expert_module = MicroExpert()
        
        # Instead of `torch.load` which deserializes entirely into RAM first,
        # safetensors instantly memory-maps the file.
        tensors = {}
        with safe_open(file_path, framework="pt", device=str(self.device)) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
                
        # Inject the weights into the module
        expert_module.load_state_dict(tensors)
        expert_module.to(self.device)
        return expert_module


# --- Utility Functions for Testing ---

def generate_mock_expert(name: str):
    """Generates a random 50MB expert and saves it to SSD."""
    logger.info(f"Synthesizing Micro-Expert: {name} (50MB)...")
    expert = MicroExpert()
    
    file_path = EXPERTS_DIR / f"{name}.safetensors"
    save_file(expert.state_dict(), file_path)
    logger.debug(f"Saved {name} to {file_path}")


if __name__ == "__main__":
    print("\n--- Testing PCIe NVMe SSD Streaming ---")
    
    # 1. Create a few fake experts on disk (~50MB each)
    experts_to_make = ["expert_python", "expert_react", "expert_quantum"]
    for e in experts_to_make:
        if not (EXPERTS_DIR / f"{e}.safetensors").exists():
            generate_mock_expert(e)
            
    # 2. Test the streaming latency
    streamer = SSDStreamer(device="cpu") # using CPU for generic testing
    
    print("\nSimulating real-time dynamic expert routing...")
    
    for _ in range(3):
        for expert_name in experts_to_make:
            # Measure time to hot-swap 50MB into operational memory
            start_time = time.perf_counter()
            
            active_expert = streamer.load_expert_to_vram(expert_name)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            print(f"Hot-swapped [{expert_name}] from NVMe -> Memory in {latency_ms:.2f} ms")
            
            # Simulate a quick forward pass
            dummy_data = torch.randn(1, 1024, device=streamer.device)
            _ = active_expert(dummy_data)
            
            # Flush (in a real PyTorch script, we just let garbage collection kill the reference
            # or explicitly use del active_expert to free VRAM instantly).
            del active_expert
