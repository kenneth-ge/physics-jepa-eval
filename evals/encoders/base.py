"""The black-box encoder interface every model must satisfy.

An encoder turns a video file into a SINGLE 1-D vector representing the
clip's last second. The eval harness only ever sees this vector, so any
model (V-JEPA, a video CLIP, a custom probe, ...) can be swapped in by
implementing `encode` and registering it in evals/encoders/__init__.py.
"""

import numpy as np


class Encoder:
    #: short stable identifier, used in CLI flags and result tables
    name = "encoder"

    def encode(self, video_path) -> np.ndarray:
        """Return a 1-D float vector for the video's last second."""
        raise NotImplementedError
