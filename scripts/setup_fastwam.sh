#!/usr/bin/env bash
# Stand up the FastWAM environment on the box (separate venv: torch 2.7.1,
# transformers 4.49 — incompatible with the V-JEPA env). Stage 1 only:
# venv + clone + install + import check. Downloads (12GB checkpoint, Wan base,
# backbone preprocess) are stage 2, run after imports are confirmed.
set -euo pipefail

BASE=/data/fastwam
REPO="$BASE/FastWAM"

mkdir -p "$BASE"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/yuantianyuan01/FastWAM "$REPO"
fi

if [ ! -d "$BASE/venv" ]; then
  python3.10 -m venv "$BASE/venv"
fi
# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
pip install -q -U pip
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
cd "$REPO"
pip install -q -e .
pip install -q "imageio[ffmpeg]" pillow

echo "=== import check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import fastwam; print('fastwam import OK')"
echo "STAGE1_DONE"
