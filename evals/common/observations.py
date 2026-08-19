"""Write a scene variant's observations to disk.

The primary camera is always saved as <name>.mp4 (back-compatible: V-JEPA and
`measure` read this). Extra cameras go to <name>_<cam>.mp4, and per-frame
state (proprioception-like signal, e.g. object poses) to <name>_state.npz.
A model's encoder then consumes whichever files it needs.
"""

import numpy as np

from .cameras import PRIMARY
from .video import save_video


def write_observations(out_dir, name, cam_frames, fps, state=None, times=None,
                       state_key="state"):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_video(cam_frames[PRIMARY], out_dir / f"{name}.mp4", fps)
    for cam, frames in cam_frames.items():
        if cam == PRIMARY:
            continue
        save_video(frames, out_dir / f"{name}_{cam}.mp4", fps)
    if state is not None:
        payload = {state_key: np.asarray(state)}
        if times is not None:
            payload["times"] = np.asarray(times)
        np.savez(out_dir / f"{name}_state.npz", **payload)
