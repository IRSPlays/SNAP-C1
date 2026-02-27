import torch
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from v4_core.utils.device import get_device

print("Calling get_device()...")
device = get_device()
print(f"Device: {device}")
