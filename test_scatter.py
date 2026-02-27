import torch
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from v4_core.utils.device import get_device

device = get_device()
print(f"Device: {device}")

print("Testing scatter_add_")
dest = torch.zeros(1, 100279).to(device)
src = torch.randn(1, 20).to(device)
idx = torch.randint(0, 100000, (1, 20)).to(device)

dest.scatter_add_(1, idx, src)
print("scatter_add_ finished.")
