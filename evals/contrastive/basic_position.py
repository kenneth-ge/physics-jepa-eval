"""basic_position eval family: one pyramid, three positions.

Each test case renders the SAME pyramid (identical shape/color) three times,
moving only where it sits on the floor:

  A: pyramid at position X;
  B: pyramid at X + delta   (a small-to-medium radius away);
  C: pyramid at 5 * delta from X (a far position, random direction).

Invariant: A,B (near positions) closer to each other than either is to C
(far position). This holds geometrically by construction (|A-B|=delta,
|A-C|=5*delta, |B-C|>=4*delta), so a pass means the embedding tracks position.
30 cases vary X, delta, and the offset directions; every position is kept
inside the camera's view.

Scenes are static: one rendered frame tiled to the clip length. `--rig stereo`
also writes the right camera (<name>_right.mp4) for two-view models.

Usage: python -m evals.basic_position --out-root <dir>   (--check: no render)
Outputs <dir>/case_00 .. case_29, each with A/B/C.mp4.
"""

import mujoco
import numpy as np

from ..common import cameras
from ..common.family import family_cli
from ..common.objects import pyramid_mesh_asset, pyramid_geom
from ..common.observations import write_observations
from ..common.sim import render_static_rig
from ..common.xml_scene import scene_xml

EVAL_IDS = list(range(30))
SEED_BASE = 5300

# In-view floor region for the pyramid centroid (matches the fixed camera).
POS_BOX = dict(x=1.15, y=0.42)
PYR_HALF = 0.11          # pyramid base half-width (footprint margin)
PYR_HEIGHT = 0.20
DELTA_RANGE = (0.10, 0.30)   # A->B radius (small to medium)
C_FACTOR = 5.0               # A->C distance = C_FACTOR * delta


def _in_view(p):
    return abs(p[0]) <= POS_BOX["x"] - PYR_HALF and abs(p[1]) <= POS_BOX["y"] - PYR_HALF


def _unit(rng):
    th = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(th), np.sin(th)])


def case_positions(i):
    """(A, B, C) floor positions and delta for case i, all guaranteed in view.
    Resamples with a shrinking delta until every position fits the frame."""
    rng = np.random.default_rng([SEED_BASE, i])
    for k in range(800):
        delta = float(rng.uniform(*DELTA_RANGE)) * (0.97 ** k)
        X = np.array([rng.uniform(-POS_BOX["x"], POS_BOX["x"]),
                      rng.uniform(-POS_BOX["y"], POS_BOX["y"])])
        B = X + delta * _unit(rng)
        C = X + C_FACTOR * delta * _unit(rng)
        if _in_view(X) and _in_view(B) and _in_view(C):
            return {"A": X, "B": B, "C": C}, delta
    raise RuntimeError(f"case {i}: could not place all positions in view")


def check_contract(pos, delta):
    d_ab = float(np.linalg.norm(pos["A"] - pos["B"]))
    d_ac = float(np.linalg.norm(pos["A"] - pos["C"]))
    d_bc = float(np.linalg.norm(pos["B"] - pos["C"]))
    failures = []
    if not (d_ab < d_ac and d_ab < d_bc):
        failures.append(f"positions: AB={d_ab:.3f} not < AC={d_ac:.3f}, BC={d_bc:.3f}")
    for n, p in pos.items():
        if not _in_view(p):
            failures.append(f"{n}: out of view at {np.round(p, 3)}")
    return failures, (d_ab, d_ac, d_bc)


def generate(i, out_dir, args):
    pos, delta = case_positions(i)
    failures, (d_ab, d_ac, d_bc) = check_contract(pos, delta)
    print(f"case_{i:02d}: delta={delta:.3f} A={np.round(pos['A'], 2)} "
          f"B={np.round(pos['B'], 2)} C={np.round(pos['C'], 2)} "
          f"| dist AB={d_ab:.3f} AC={d_ac:.3f} BC={d_bc:.3f}")
    if out_dir is None:
        return failures

    cam_names = cameras.rig_cameras(args.rig)
    assets = pyramid_mesh_asset(half=PYR_HALF, height=PYR_HEIGHT)
    for name in ("A", "B", "C"):
        model = mujoco.MjModel.from_xml_string(scene_xml(
            assets=assets, geoms=pyramid_geom(f"pyr_{name}", pos[name]),
            extra_cameras_xml=cameras.extra_cameras_xml(args.rig)))
        cam_frames = render_static_rig(
            model, mujoco.MjData(model), cam_names=cam_names,
            duration=args.duration, fps=args.fps, size=args.size)
        write_observations(out_dir, name, cam_frames, args.fps)
    return failures


def add_args(parser):
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)


def main():
    family_cli(name="basic_position", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda i: f"case_{i:02d}", description=__doc__,
               add_args=add_args)


if __name__ == "__main__":
    main()
