"""Encoder that reads precomputed vectors from disk.

Lets a model that lives in a DIFFERENT environment (its own torch/deps)
contribute to the same eval: that model runs `evals.precompute` in its venv
and writes `<name>__<tag>.npy` next to each `<name>.mp4`; then `measure`
consumes them here via the `saved:<tag>` encoder, in any environment.
"""

import pathlib

import numpy as np

from .base import Encoder


class SavedEncoder(Encoder):
    def __init__(self, tag):
        self.tag = tag
        self.name = f"saved:{tag}"

    def encode(self, video_path):
        p = pathlib.Path(video_path)
        vec = p.parent / f"{p.stem}__{self.tag}.npy"
        if not vec.exists():
            raise FileNotFoundError(
                f"no precomputed vector {vec}; run "
                f"`python -m evals.precompute --root <dir> --encoder <in-env> "
                f"--tag {self.tag}` in that model's environment first")
        return np.load(vec)
