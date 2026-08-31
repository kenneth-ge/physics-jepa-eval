"""Video I/O for eval clips (imageio imported lazily: rendering machines
have it; --check runs don't need it)."""

import numpy as np


def save_video(frames, path, fps):
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    # All-intra encoding (every frame a keyframe): H.264 inter prediction
    # otherwise leaves shared-history compression residue in the final frames
    # of clips whose earlier content matches (A,B vs C), which a raw-pixel /
    # pixel-faithful readout can score on even when the raw renders are
    # bit-identical. With -g 1, identical raw frames encode bit-identically
    # (~3.5x file size; clips this small don't care).
    imageio.mimsave(path, np.stack(frames), fps=fps, output_params=["-g", "1"])


def load_video(path):
    """Returns (frames (T,H,W,3) uint8, fps)."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(path)
    fps = reader.get_meta_data()["fps"]
    frames = np.stack([f for f in reader])
    reader.close()
    return frames, fps
