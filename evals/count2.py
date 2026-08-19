"""count2 eval family: nested/additive counting.

Unlike basic_counting (three disjoint layouts), here the scenes are nested by
construction — cubes are only ever ADDED, never moved:

  A: X cubes;
  B: X + 1 cubes (A's cubes plus one more, same positions);
  C: 2X cubes  (B's cubes plus X-1 more, same positions).

So A ⊂ B ⊂ C, with counts X, X+1, 2X. The invariant (A,B closer than C) isolates
sensitivity to the NUMBER of cubes with position held constant: A→B adds 1 cube,
B→C adds X-1. Sweep over X = 2..10 (9 rungs); X=1 is excluded because there
2X = X+1 = 2, i.e. B and C would be identical.

Renders <root>/seed{S}/eval_{X:02d}/{A,B,C}.mp4 so `evals.aggregate_counting`
reads it directly. Scenes are static; `--rig stereo` also writes the right cam.
"""

import mujoco
import numpy as np

from .basic_counting import sample_poses
from .common import cameras
from .common.observations import write_observations
from .common.sim import render_static_rig
from .common.xml_scene import scene_xml, fr

CUBE_HALF = 0.09
CUBE_RGBA = (0.2, 0.4, 0.9, 1)


def cube_geom(name, pose):
    x, y, yaw = pose
    return (f'<geom name="{name}" type="box" '
            f'size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" '
            f'pos="{x:.4f} {y:.4f} {CUBE_HALF}" euler="0 0 {yaw:.2f}" '
            f'material="obj" rgba="{fr(CUBE_RGBA)}"/>')


def build_scenes(x, seed):
    """Nested cube layouts for count X: A=pool[:X], B=pool[:X+1], C=pool[:2X]."""
    rng = np.random.default_rng([seed, x])
    pool = sample_poses(2 * x, rng)
    return {"A": pool[:x], "B": pool[:x + 1], "C": pool[:2 * x]}


def render_scene(out_dir, name, poses, rig, fps, size, duration):
    geoms = "".join(cube_geom(f"cube{i}", p) for i, p in enumerate(poses))
    model = mujoco.MjModel.from_xml_string(scene_xml(
        geoms=geoms, extra_cameras_xml=cameras.extra_cameras_xml(rig)))
    cam_frames = render_static_rig(model, mujoco.MjData(model),
                                   cam_names=cameras.rig_cameras(rig),
                                   duration=duration, fps=fps, size=size)
    write_observations(out_dir, name, cam_frames, fps)
