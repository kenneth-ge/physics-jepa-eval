"""basic_color eval family: one object, three colors.

Each test case renders the SAME object (same shape, same pose) three times,
changing only its color:

  A, B: two hues a SHORT arc apart on the HSV color wheel (perceptually close);
  C:    the roughly OPPOSITE hue (~180 deg away on the wheel).

Saturation and value are held equal across A/B/C within a case, so the only
signal is hue-wheel distance. Invariant: A,B (near hues) closer to each other
than either is to C (far hue). 30 cases vary the base hue, the A/B separation,
the sign of that separation, and the shared saturation/value.

Scenes are static: one rendered frame tiled to the clip length. `--rig stereo`
also writes the right camera (<name>_right.mp4) for two-view models.

Usage: python -m evals.basic_color --out-root <dir>   (--check: no render)
Outputs <dir>/case_00 .. case_29, each with A/B/C.mp4.
"""

import colorsys

import mujoco
import numpy as np

from ..common import cameras
from ..common.family import family_cli
from ..common.observations import write_observations
from ..common.sim import render_static_rig
from ..common.xml_scene import scene_xml, fr

EVAL_IDS = list(range(30))
SEED_BASE = 4200

# A/B hue separation (fraction of the wheel; 1.0 == 360 deg). ~14-40 deg.
AB_SEP = (0.04, 0.11)
# C offset from the base hue: ~half the wheel, lightly jittered so C is not
# always exactly antipodal.
C_JITTER = 0.05
# Saturation / value stay high so the hue reads clearly and equally for all.
SV_RANGE = (0.65, 1.0)

# A single boxy object centered on the floor, large enough that its color
# dominates the frame. Only rgba changes between A/B/C.
BOX_HALF = 0.26


def _hue_dist(a, b):
    """Circular distance between two hues expressed as wheel fractions."""
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def case_hsv(i):
    """(h, s, v) for A, B, C of case i. s and v shared; only hue differs."""
    rng = np.random.default_rng([SEED_BASE, i])
    h = float(rng.uniform(0, 1))
    s = float(rng.uniform(*SV_RANGE))
    v = float(rng.uniform(*SV_RANGE))
    sep = float(rng.uniform(*AB_SEP)) * float(rng.choice([-1.0, 1.0]))
    cj = float(rng.uniform(-C_JITTER, C_JITTER))
    return {
        "A": (h % 1.0, s, v),
        "B": ((h + sep) % 1.0, s, v),
        "C": ((h + 0.5 + cj) % 1.0, s, v),
    }


def rgba(hsv):
    r, g, b = colorsys.hsv_to_rgb(*hsv)
    return (r, g, b, 1.0)


def box_geom(color):
    return (f'<geom name="obj" type="box" size="{BOX_HALF} {BOX_HALF} {BOX_HALF}" '
            f'pos="0 0 {BOX_HALF}" material="obj" rgba="{fr(color)}"/>')


def check_contract(hsvs):
    """The ground truth the invariant should reflect: A,B nearer in hue than
    either is to C."""
    d_ab = _hue_dist(hsvs["A"][0], hsvs["B"][0])
    d_ac = _hue_dist(hsvs["A"][0], hsvs["C"][0])
    d_bc = _hue_dist(hsvs["B"][0], hsvs["C"][0])
    failures = []
    if not (d_ab < d_ac and d_ab < d_bc):
        failures.append(f"hue wheel: AB={d_ab:.3f} not < AC={d_ac:.3f}, BC={d_bc:.3f}")
    return failures, (d_ab, d_ac, d_bc)


def generate(i, out_dir, args):
    hsvs = case_hsv(i)
    failures, (d_ab, d_ac, d_bc) = check_contract(hsvs)
    hues = {n: round(hsvs[n][0], 3) for n in "ABC"}
    print(f"case_{i:02d}: hues {hues} s={hsvs['A'][1]:.2f} v={hsvs['A'][2]:.2f} "
          f"| hue-dist AB={d_ab:.3f} AC={d_ac:.3f} BC={d_bc:.3f}")
    if out_dir is None:
        return failures

    cam_names = cameras.rig_cameras(args.rig)
    for name in ("A", "B", "C"):
        model = mujoco.MjModel.from_xml_string(scene_xml(
            geoms=box_geom(rgba(hsvs[name])),
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
    family_cli(name="basic_color", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda i: f"case_{i:02d}", description=__doc__,
               add_args=add_args)


if __name__ == "__main__":
    main()
