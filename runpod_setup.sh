#!/bin/bash
# ============================================================
#  SNAP-C1 V4 — RunPod Parallelized Training
#  Target: NVIDIA RTX 6000 Ada (48GB VRAM)
# ============================================================
# 
# USAGE:
#   1. Create a RunPod GPU Pod with "PyTorch 2.x" template
#   2. Clone your repo:  git clone https://github.com/IRSPlays/SNAP-C1.git
#   3. cd SNAP-C1
#   4. chmod +x runpod_setup.sh && ./runpod_setup.sh
#
# PARALLELIZATION ENABLED:
#   - Batch size 8 (process 8 chunks simultaneously)
#   - AMP FP16 (2x Tensor Core speedup, auto-detected)
#   - torch.compile (kernel fusion, auto-detected)
#   - 4 DataLoader workers (async CPU prefetching)
# ============================================================

set -e

echo "========================================"
echo "  SNAP-C1 V4 RunPod Setup (CUDA)"
echo "========================================"

# Install Python dependencies
pip install --quiet torch torchvision torchaudio
pip install --quiet tiktoken safetensors loguru chromadb onnxruntime numpy tqdm

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found!'; print(f'CUDA OK: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

echo ""
echo "========================================"
echo "  Step 1: Generate Training Dataset"
echo "========================================"
python v4_core/data/v4_general_dataset_builder.py --target_dir ./v3_core --output v4_core/data/v4_test_dataset.json

echo ""
echo "========================================"
echo "  Step 2: Launch Parallelized Training"
echo "  (100 Epochs, Batch=8, 4 Workers)"
echo "========================================"
python v4_core/training/v4_ddp_trainer.py \
    --epochs 100 \
    --batch_size 8 \
    --workers 4

echo ""
echo "========================================"
echo "  Training Complete! Weights saved to:"
echo "  v4_core/snapshot_v4_hyper_router.pt"
echo "========================================"
