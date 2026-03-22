#!/bin/bash
# V6 WHORMHOLE - RunPod Setup Script (RTX 6000 Ada 48GB)
# ======================================================

set -e

echo "==================================="
echo "V6 WHORMHOLE - RunPod Setup"
echo "==================================="
echo "Target: RTX 6000 Ada 48GB"

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi

# Install PyTorch with CUDA 12.x
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install transformers datasets accelerate peft bitsandbytes trl

# Install V6 requirements
echo ""
echo "Installing V6 requirements..."
pip install rich loguru pyyaml jsonlines rouge-score nltk tqdm sentence-transformers

# Verify CUDA
echo ""
echo "Verifying CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Test GPU
echo ""
echo "Testing GPU..."
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Clone or update repository
echo ""
echo "Setting up V6 repository..."
cd /workspace
if [ -d "SNAP-C1" ]; then
    echo "Repository exists, pulling latest..."
    cd SNAP-C1 && git pull
else
    echo "Clone repository..."
    git clone <YOUR_REPO_URL> SNAP-C1
    cd SNAP-C1
fi

# Test V6 imports
echo ""
echo "Testing V6 imports..."
python -c "
import sys
sys.path.insert(0, '.')
from v6_core.architecture import (
    V6ResonanceModel, build_v6_local, build_v6_small,
    build_v6_rtx6000, build_v6_large
)
from v6_core.architecture.plastic_weights import PlasticLinear
from v6_core.architecture.agentic_coding import CodingAgent, SelfImprovingWrapper
print('V6 imports OK!')
print('Available models:')
print('  - build_v6_small: ~60M params')
print('  - build_v6_local: ~138M params')
print('  - build_v6_rtx6000: ~800M params (optimized for 48GB)')
print('  - build_v6_large: ~1B params (for A100 80GB)')
"

# Run GPU test
echo ""
echo "Running GPU test..."
python v6_core/training/test_gpu.py

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Training options:"
echo "  1. Quick test: python v6_core/training/test_gpu.py"
echo "  2. Full training: python v6_core/training/v6_full_train.py"
echo "  3. Self-learning: python v6_core/training/v6_self_learning.py"
echo ""
echo "For continuous self-learning on coding tasks:"
echo "  python v6_core/training/v6_self_learning.py --continuous"
echo ""
