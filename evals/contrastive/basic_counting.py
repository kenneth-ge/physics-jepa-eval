"""basic_counting eval family: static scenes of triangular prisms.

Eval X (X = 1..10):
  A: X prisms at positions P_1..P_X
  B: X prisms at positions Q_1..Q_X
  C: set by --c-mode:
     fresh (default): X+1 prisms at their own independent random positions —
       all three layouts are equally disjoint, and the only systematic
       {A,B} vs C difference is the COUNT, off by one.
     union: the exact union of A's and B's prisms (2X). Measured 2026-08-18:
       fails 10/10 under both readouts, because A is a subset of C — any
       overlap-faithful similarity ranks A closer to C than to B. Kept for
       comparison against fresh (overlap sensitivity vs count sensitivity).

Positions are seeded per eval number: A and B split one non-overlapping
2X pool; fresh-mode C draws its own pool from the same rng stream.

Invariant: A,B (same count, different layout) closer to each other than
either is to C (different count).

Scenes are static: one rendered frame tiled to the clip length.

Usage: python -m evals.basic_counting --out-root <dir>   (--check: no render)
Outputs <dir>/eval_01 .. eval_10, each with A/B/C.mp4.
"""

import itertools

import mujoco
import numpy as np

from ..common.family import family_cli
from ..common.objects import prism_mesh_asset, prism_geom
from ..common.sim import render_static
from ..common.video import save_video
from ..common.xml_scene import scene_xml

EVAL_IDS = list(range(1, 11))

X_RANGE = (-1.25, 1.25)
Y_RANGE = (-0.45, 0.50)
MIN_SEPARATION = 0.30


def sample_poses(n, rng, max_tries=20000):
    """n non-overlapping (x, y, yaw_deg) poses inside the camera's view."""
    poses = []
    for _ in range(max_tries):
        if len(poses) == n:
            break
        x = rng.uniform(*X_RANGE)
        y = rng.uniform(*Y_RANGE)
        if all(np.hypot(x - px, y - py) >= MIN_SEPARATION for px, py, _ in poses):
            poses.append((x, y, rng.uniform(0, 360)))
    else:
        raise RuntimeError(f"could not place {n} prisms with separation {MIN_SEPARATION}")
    return poses


def scene_geoms(poses):
    return "".join(prism_geom(f"prism{i}", (x, y), yaw)
                   for i, (x, y, yaw) in enumerate(poses))


def render_scene(poses, path, duration, fps, size):
    model = mujoco.MjModel.from_xml_string(
        scene_xml(assets=prism_mesh_asset(), geoms=scene_geoms(poses)))
    frames = render_static(model, mujoco.MjData(model),
                           duration=duration, fps=fps, size=size)
    save_video(frames, path, fps)


def build_scenes(x, seed_base, c_mode):
    """Poses for eval X. A and B split one non-overlapping 2X pool; fresh-mode
    C draws its own X+1 pool from the same rng stream."""
    rng = np.random.default_rng(seed_base + x)
    pool = sample_poses(2 * x, rng)
    scenes = {"A": pool[:x], "B": pool[x:]}
    scenes["C"] = pool if c_mode == "union" else sample_poses(x + 1, rng)
    return scenes


def generate(x, out_dir, args):
    scenes = build_scenes(x, args.seed_base, args.c_mode)
    min_sep = min((np.hypot(a[0] - b[0], a[1] - b[1])
                   for scene in scenes.values()
                   for a, b in itertools.combinations(scene, 2)),
                  default=np.inf)
    print(f"eval_{x:02d}: A={x} B={x} C={len(scenes['C'])} prisms "
          f"(c-mode={args.c_mode}), min within-scene separation {min_sep:.3f}m")
    failures = [] if min_sep >= MIN_SEPARATION - 1e-9 else [
        f"within-scene separation {min_sep:.3f} < {MIN_SEPARATION}"]
    if out_dir is not None:
        for name, poses in scenes.items():
            render_scene(poses, out_dir / f"{name}.mp4", args.duration, args.fps, args.size)
    return failures


def add_args(parser):
    parser.add_argument("--c-mode", choices=("fresh", "union"), default="fresh")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)


def main():
    family_cli(name="basic_counting", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda x: f"eval_{x:02d}", description=__doc__, add_args=add_args)


if __name__ == "__main__":
    main()
