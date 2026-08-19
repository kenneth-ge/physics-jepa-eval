#!/usr/bin/env bash
# Cosmos 3 environment (own venv). Cosmos3OmniPipeline is only in diffusers
# git main. Weights are NOT gated (OpenMDW1.1) — no HF token needed.
# Cosmos3-Super is 64B (~120GB bf16 download); Cosmos3-Nano (16B) is the
# lighter fallback via COSMOS_MODEL_ID.
set -euo pipefail

BASE=/data/cosmos
mkdir -p "$BASE"
if [ ! -d "$BASE/venv" ]; then
  python3.10 -m venv "$BASE/venv"
fi
# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
pip install -q -U pip
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    transformers accelerate av pillow "imageio[ffmpeg]" numpy

echo "=== import check ==="
python -c "import torch, diffusers; print('torch', torch.__version__, 'diffusers', diffusers.__version__)"
python -c "from diffusers import Cosmos3OmniPipeline; print('Cosmos3OmniPipeline OK')"
echo "COSMOS_STAGE1_DONE"
