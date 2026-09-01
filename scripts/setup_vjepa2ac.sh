#!/usr/bin/env bash
# One-time setup for V-JEPA 2-AC (action-conditioned). Idempotent; everything
# lands on the /data volume so later jobs skip straight to encoding.
#
# NOTE: `torch.hub.load('facebookresearch/vjepa2', ...)` is BROKEN upstream —
# src/hub/backbones.py has VJEPA_BASE_URL hardcoded to http://localhost:8300
# ("for testing"), so the hub path tries to download from localhost. Meta's own
# advice (issue #18) is to wget the checkpoint, which is what we do here.
set -euo pipefail

ROOT="${VJEPA2AC_ROOT:-/data/vjepa2ac}"
CKPT="$ROOT/vjepa2-ac-vitg.pt"
REPO="$ROOT/vjepa2"
URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
SIZE=11760743310   # bytes, verified

mkdir -p "$ROOT"

if [ ! -d "$REPO/.git" ]; then
  echo "=== cloning facebookresearch/vjepa2 ==="
  git clone --depth 1 https://github.com/facebookresearch/vjepa2 "$REPO"
else
  echo "=== vjepa2 repo present ==="
fi

if [ -f "$CKPT" ] && [ "$(stat -c%s "$CKPT")" = "$SIZE" ]; then
  echo "=== AC checkpoint present ($SIZE bytes) ==="
else
  echo "=== downloading AC checkpoint (11.8GB) ==="
  wget -c -O "$CKPT" "$URL"
  got=$(stat -c%s "$CKPT")
  [ "$got" = "$SIZE" ] || { echo "SIZE MISMATCH: got $got want $SIZE"; exit 1; }
fi

echo "VJEPA2AC_ROOT=$ROOT"
echo "VJEPA2AC_CKPT=$CKPT"
echo "VJEPA2AC_REPO=$REPO"
