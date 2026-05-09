#!/bin/bash
set -e
echo "=== GPU Check ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')"

echo "=== Setting up workspace ==="
cd /workspace
if [ ! -d "SNAP-C1" ]; then
    git clone https://github.com/IRSPlays/SNAP-C1.git
fi
cd SNAP-C1
git pull origin main

echo "=== Running integration test ==="
python3 integration_test.py 2>&1 | tail -5

echo ""
echo "=== Setup complete ==="
echo "Ready to train: cd /workspace/SNAP-C1 && python3 -m cortex.train --generate-synthetic 50000"
echo "                python3 -m cortex.train --pretrain"
