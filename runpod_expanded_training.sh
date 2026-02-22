#!/bin/bash
# ============================================================
#  SNAP-C1 V4 — Expanded Training Pipeline for RunPod
#  Target: NVIDIA RTX 6000 Ada (48GB VRAM)
# ============================================================
#
# This script:
#   1. Clones 4 major Python repos (Django, Flask, scikit-learn, requests)
#   2. Generates training data from each (target: 500+ chunks)
#   3. Merges all datasets into one large training set
#   4. Trains 500 epochs with max_loops=100 for deeper ODE convergence
#   5. Runs SWE-Bench benchmark on the improved model
#
# USAGE:
#   chmod +x runpod_expanded_training.sh
#   ./runpod_expanded_training.sh
# ============================================================

set -e

echo "========================================================"
echo "  SNAP-C1 V4 — Expanded Training Pipeline"
echo "========================================================"

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Install deps
pip install --quiet tiktoken safetensors loguru chromadb onnxruntime numpy tqdm datasets

# ============================================================
#  STEP 1: Clone Major Python Repos
# ============================================================
echo ""
echo "========================================================"
echo "  Step 1: Cloning Major Python Repositories"
echo "========================================================"

REPOS_DIR="/workspace/training_repos"
mkdir -p $REPOS_DIR

# Clone repos (shallow clone for speed — we only need the Python source)
declare -A REPOS=(
    ["django"]="https://github.com/django/django.git"
    ["flask"]="https://github.com/pallets/flask.git"
    ["scikit-learn"]="https://github.com/scikit-learn/scikit-learn.git"
    ["requests"]="https://github.com/psf/requests.git"
)

for name in "${!REPOS[@]}"; do
    if [ ! -d "$REPOS_DIR/$name" ]; then
        echo "  Cloning $name..."
        git clone --depth 1 --quiet "${REPOS[$name]}" "$REPOS_DIR/$name"
    else
        echo "  $name already cloned, skipping."
    fi
done

echo "  All repos ready."

# ============================================================
#  STEP 2: Generate Dataset from Each Repo
# ============================================================
echo ""
echo "========================================================"
echo "  Step 2: Generating Training Data from Real Repos"
echo "========================================================"

cd /workspace/SNAP-C1

# Generate dataset from each repo's Python source
for name in django flask scikit-learn requests; do
    echo ""
    echo "  --- Processing: $name ---"
    python v4_core/data/v4_general_dataset_builder.py \
        --target_dir "$REPOS_DIR/$name" \
        --output "v4_dataset_${name}.json" 2>&1 | tail -3
done

# Also include our own codebase
echo "  --- Processing: SNAP-C1 (self) ---"
python v4_core/data/v4_general_dataset_builder.py \
    --target_dir "./v3_core" \
    --output "v4_dataset_self.json" 2>&1 | tail -3

# ============================================================
#  STEP 3: Merge All Datasets
# ============================================================
echo ""
echo "========================================================"
echo "  Step 3: Merging All Datasets"
echo "========================================================"

python -c "
import json, glob, os

merged = []
for f in glob.glob('v4_core/data/v4_dataset_*.json'):
    with open(f) as fh:
        data = json.load(fh)
        print(f'  {os.path.basename(f):30s} → {len(data):5d} chunks')
        merged.extend(data)

# Also include original test dataset
test_path = 'v4_core/data/v4_test_dataset.json'
if os.path.exists(test_path):
    with open(test_path) as fh:
        data = json.load(fh)
        print(f'  {\"v4_test_dataset.json\":30s} → {len(data):5d} chunks')
        merged.extend(data)

out_path = 'v4_core/data/v4_expanded_dataset.json'
with open(out_path, 'w') as fh:
    json.dump(merged, fh, indent=2)

print(f'\n  TOTAL: {len(merged)} training chunks → {out_path}')
"

# ============================================================
#  STEP 4: Train with Expanded Data (500 Epochs, max_loops=100)
# ============================================================
echo ""
echo "========================================================"
echo "  Step 4: Training V4 (500 Epochs, Expanded Dataset)"
echo "========================================================"

python v4_core/training/v4_ddp_trainer.py \
    --epochs 500 \
    --batch_size 16 \
    --workers 4 \
    --dataset v4_core/data/v4_expanded_dataset.json \
    --max_loops 100

# ============================================================
#  STEP 5: Re-evaluate on SWE-Bench
# ============================================================
echo ""
echo "========================================================"
echo "  Step 5: SWE-Bench Re-Benchmark"
echo "========================================================"

python v4_core/evaluation/v4_swe_bench.py \
    --weights v4_core/snapshot_v4_hyper_router.pt \
    --max_instances 50 \
    --batch_size 8

echo ""
echo "========================================================"
echo "  Pipeline Complete! Check swe_bench_report.json"
echo "========================================================"
