"""Encode the same clip twice with one encoder and assert identical vectors.

Usage: python scripts/check_encoder_determinism.py <encoder_name> <video_path>
"""

import sys

import numpy as np

from evals.encoders import get_encoder


def main():
    name, path = sys.argv[1], sys.argv[2]
    enc = get_encoder(name)
    a, b = enc.encode(path), enc.encode(path)
    d = float(np.abs(a - b).max())
    print(f"{name} deterministic: {d == 0.0}  max|diff|: {d}")
    assert d == 0.0, f"{name} encode is non-deterministic (max|diff| {d})"


if __name__ == "__main__":
    main()
