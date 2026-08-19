#!/usr/bin/env bash
# FastWAM stage 2: downloads + backbone preprocess. Run after stage 1.
# ~22GB of downloads (12GB finetuned checkpoint + Wan2.2 base via DiffSynth).
set -euo pipefail

BASE=/data/fastwam
REPO="$BASE/FastWAM"
# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
cd "$REPO"
export DIFFSYNTH_MODEL_BASE_PATH="$REPO/checkpoints"
# The Wan2.2 base only resolves on ModelScope (the HF converted-safetensors
# repo 404s); slow (~250kB/s) but reliable, and a managed job won't time out.
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_HOME=/data/hf

# DiffSynth's preprocess AND the FastWAM runtime (via transformers) both pin
# huggingface-hub<1.0, so keep the venv there and use the 0.x `huggingface-cli`
# for the download (the `hf` CLI only exists in hub>=1.0, which we must avoid).
pip install -q "huggingface_hub[cli]<1.0" modelscope

echo "=== downloading finetuned checkpoint (12GB) ==="
huggingface-cli download yuanty/fastwam \
    libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
    --local-dir "$REPO/checkpoints/fastwam_release"

BACKBONE="$REPO/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [ -f "$BACKBONE" ]; then
  echo "=== backbone already present, skipping preprocess ==="
else
  echo "=== preprocessing ActionDiT backbone (downloads Wan2.2 base) ==="
  python scripts/preprocess_action_dit_backbone.py \
      --model-config configs/model/fastwam.yaml \
      --output "$BACKBONE"
fi

# Guarantee the venv is left at hub<1.0 for the runtime, regardless of whether
# any dependency above nudged it up.
pip install -q "huggingface_hub<1.0"

echo "FASTWAM_STAGE2_DONE"
