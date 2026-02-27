import torch
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from v4_core.utils.device import get_device

device = get_device()
print(f"Device: {device}")

print("Testing large softmax...")
x = torch.randn(1, 100279).to(device)
y = torch.softmax(x, dim=-1)
print("Softmax finished.")
