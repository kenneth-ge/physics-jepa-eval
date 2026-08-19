"""Video I/O for eval clips (imageio imported lazily: rendering machines
have it; --check runs don't need it)."""

import numpy as np


def save_video(frames, path, fps):
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, np.stack(frames), fps=fps)


def load_video(path):
    """Returns (frames (T,H,W,3) uint8, fps)."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(path)
    fps = reader.get_meta_data()["fps"]
    frames = np.stack([f for f in reader])
    reader.close()
    return frames, fps
