#!/usr/bin/env bash
# Qwen3-VL environment (own venv: transformers >= 4.57, qwen-vl-utils).
# Weights are Apache-2.0 and NOT gated; from_pretrained downloads on first use.
set -euo pipefail

BASE=/data/qwen
mkdir -p "$BASE"
if [ ! -d "$BASE/venv" ]; then
  python3.10 -m venv "$BASE/venv"
fi
# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
pip install -q -U pip
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q "git+https://github.com/huggingface/transformers" accelerate \
    "qwen-vl-utils[decord]==0.0.14" pillow "imageio[ffmpeg]" numpy

echo "=== import check ==="
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
python -c "from transformers import AutoModelForImageTextToText; print('qwen classes OK')"
echo "QWEN_STAGE1_DONE"
