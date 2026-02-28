#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SNAP-C1 DigitalOcean H200 Setup Script
# Run this once after SSH-ing into your DO GPU Droplet:
#   bash setup_do_h200.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "======================================"
echo " SNAP-C1 H200 Environment Setup"
echo "======================================"

# ── 1. System deps ─────────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git wget curl python3-pip python3-venv nvtop screen

# ── 2. Python env ──────────────────────────────────────────────────────────
echo "[2/7] Creating Python virtual environment..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip -q

# ── 3. PyTorch (CUDA 12.4 — matches DO H200 driver) ───────────────────────
echo "[3/7] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# ── 4. Training deps ───────────────────────────────────────────────────────
echo "[4/7] Installing training dependencies..."
pip install bitsandbytes tiktoken transformers accelerate -q
pip install numpy tqdm psutil -q

# ── 5. Clone repo ──────────────────────────────────────────────────────────
echo "[5/7] Cloning SNAP-C1 repo..."
mkdir -p /workspace
cd /workspace
if [ -d "SNAP-C1" ]; then
    echo "  Repo already exists, pulling latest..."
    cd SNAP-C1 && git pull origin main
else
    git clone https://github.com/IRSPlays/SNAP-C1.git
    cd SNAP-C1
fi

# ── 6. Download training data ──────────────────────────────────────────────
echo "[6/7] Downloading Python training corpora..."
mkdir -p /workspace/training_data
cd /workspace/training_data

REPOS=(
    "python/cpython"
    "django/django"
    "pallets/flask"
    "psf/requests"
    "pallets/click"
    "pallets/jinja"
    "scrapy/scrapy"
    "encode/httpx"
    "tiangolo/fastapi"
    "pydantic/pydantic"
    "pytest-dev/pytest"
    "numpy/numpy"
    "ansible/ansible"
    "psf/black"
    "encode/django-rest-framework"
    "matplotlib/matplotlib"
    "pandas-dev/pandas"
    "saltstack/salt"
    "scikit-learn/scikit-learn"
    "scipy/scipy"
    "sqlalchemy/sqlalchemy"
    "encode/starlette"
    "sympy/sympy"
    "huggingface/transformers"
    "python/mypy"
    "twisted/twisted"
)

for REPO in "${REPOS[@]}"; do
    NAME=$(echo $REPO | cut -d'/' -f2)
    if [ -d "$NAME" ]; then
        echo "  $NAME already cloned, skipping"
    else
        echo "  Cloning $NAME..."
        git clone --depth=1 "https://github.com/$REPO.git" "$NAME" 2>/dev/null || echo "  WARNING: Failed to clone $NAME"
    fi
done

# ── 7. Write launch script ─────────────────────────────────────────────────
echo "[7/7] Writing launch scripts..."

cat > /workspace/train_8b.sh << 'LAUNCH'
#!/bin/bash
source /workspace/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/SNAP-C1
nohup python3 -u v5_core/training/v5_pretrain.py \
  --scale 8b \
  --data_dir /workspace/training_data \
  --epochs 3 \
  --batch_size 32 \
  --seq_len 4096 \
  --lr 1.5e-4 \
  --save_every 200 \
  --checkpoint_dir /workspace/SNAP-C1/v5_core/checkpoints \
  > /workspace/pretrain_8b_log.txt 2>&1 &
echo "Training PID: $!"
echo "Monitor: tail -f /workspace/pretrain_8b_log.txt"
LAUNCH

# Resume-from-4b version (if you transfer the RunPod checkpoint)
cat > /workspace/train_8b_resume4b.sh << 'LAUNCH'
#!/bin/bash
# First scp the 4b checkpoint from RunPod:
#   scp -i ~/.ssh/id_ed25519 -P <port> root@<runpod-ip>:/workspace/SNAP-C1/v5_core/checkpoints/v5_pretrain_latest.pt /workspace/SNAP-C1/v5_core/checkpoints/
source /workspace/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/SNAP-C1
nohup python3 -u v5_core/training/v5_pretrain.py \
  --scale 8b \
  --data_dir /workspace/training_data \
  --epochs 3 \
  --batch_size 32 \
  --seq_len 4096 \
  --lr 1.5e-4 \
  --save_every 200 \
  --checkpoint_dir /workspace/SNAP-C1/v5_core/checkpoints \
  > /workspace/pretrain_8b_log.txt 2>&1 &
echo "Training PID: $!"
LAUNCH

chmod +x /workspace/train_8b.sh /workspace/train_8b_resume4b.sh

echo ""
echo "======================================"
echo " Setup complete!"
echo "======================================"
echo ""
echo " GPU check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo " PyTorch CUDA check:"
source /workspace/venv/bin/activate
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
echo ""
echo " To start 8B training:"
echo "   screen -S train"
echo "   bash /workspace/train_8b.sh"
echo "   tail -f /workspace/pretrain_8b_log.txt"
echo ""
echo " Cost estimate at \$3.44/hr:"
echo "   8B pretrain 3 epochs ~8hrs = ~\$27"
echo "   Instruct tune ~2hrs         = ~\$7"
echo "   Total: ~\$34 of your \$200"
echo "======================================"
