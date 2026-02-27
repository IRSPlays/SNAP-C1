import torch
import sys
import os
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from v4_core.architecture.ast_decoder import V4ASTDecoder
from v4_core.utils.device import get_device

device = get_device()
print(f"Device: {device}")

decoder = V4ASTDecoder(concept_dim=1024, bpe_vocab_size=100279)
decoder.eval()

mock_equilibrium = torch.randn(1, 1, 1024).to(device)
mock_context = torch.randn(1, 20, 1024).to(device)
mock_ids = torch.randint(0, 100000, (1, 20)).to(device)

print("Starting generation...")
start_time = time.time()
with torch.no_grad():
    # Decoder stays on CPU, inputs are moved to CPU internally
    output = decoder(mock_equilibrium, mock_context, mock_ids, max_nodes=5)
end_time = time.time()

print(f"Generation took: {end_time - start_time:.4f}s")
print(f"Output: {output}")
