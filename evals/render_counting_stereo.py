"""Render the counting sweep with a multi-camera rig (no measuring).

`sweep_counting` renders the primary camera only. Models that need multiple
views (e.g. FastWAM's 2-camera input) require a second camera, so this script
re-renders the same seeded scenes with a rig, writing `<name>.mp4` (primary)
plus `<name>_<cam>.mp4` for each extra camera. Scenes are static and generated
by the exact same `build_scenes`/`sample_poses` stream as `sweep_counting`, so
the primary frames match the ones V-JEPA/Qwen already encoded.

Written to a SEPARATE root by default so it never disturbs the mono renders /
precomputed vectors under `/data/videos/counting_sweep`.

Usage (in the FastWAM env):
  python -m evals.render_counting_stereo \
      --out-root /data/videos/counting_sweep_stereo --seeds 8 --rig stereo
"""

import argparse
import pathlib

import mujoco

from .basic_counting import scene_geoms
from .common import cameras
from .common.objects import prism_mesh_asset
from .common.observations import write_observations
from .common.sim import render_static
from .common.xml_scene import scene_xml
from .sweep_counting import build_scenes


def render_scene_rig(poses, out_dir, name, duration, fps, size, rig):
    model = mujoco.MjModel.from_xml_string(scene_xml(
        assets=prism_mesh_asset(), geoms=scene_geoms(poses),
        extra_cameras_xml=cameras.extra_cameras_xml(rig)))
    data = mujoco.MjData(model)
    cam_frames = {c: render_static(model, data, duration=duration, fps=fps,
                                   size=size, camera=c)
                  for c in cameras.rig_cameras(rig)}
    write_observations(out_dir, name, cam_frames, fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--counts", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--c-mode", choices=("fresh", "union"), default="fresh")
    parser.add_argument("--rig", default="stereo")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    p = parser.parse_args()

    for seed in range(p.seeds):
        for x in p.counts:
            scenes = build_scenes(x, seed, p.c_mode)
            out = p.out_root / f"seed{seed}" / f"eval_{x:02d}"
            out.mkdir(parents=True, exist_ok=True)
            for name, poses in scenes.items():
                render_scene_rig(poses, out, name, p.duration, p.fps, p.size, p.rig)
        print(f"seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
