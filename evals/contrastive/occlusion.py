"""occlusion eval family (v2): a back face revealed only mid-spin.

A kinematic cube (mocap) spins about the vertical axis. It starts blue-front
with a RED plate on its BACK face (facing away, occluded); the red swings into
view at the half-turn and is hidden again by the end, so every clip STARTS and
ENDS on the blue side.

v2 closes the static end-frame leak found in v1 (RESULTS_contrastive.md, job
15391: fastwam z_raw 30/30 from a single frame). Two v1 bugs:
  1. The plate protruded 13mm past the back face, so its red rim was visible
     from the angled `right` stereo camera in EVERY frame, including the last
     (~40 px even at a perfect 360). v2 insets the plate flush (outer surface
     2mm proud: above depth-buffer precision, sub-pixel from every rig camera).
  2. Clips stopped one frame short of the full turn (A/C ended ~356 deg, and
     A vs B end poses differed ~1 deg). v2 quantises T to whole frames and
     includes the endpoint, so A/B/C all end EXACTLY blue-front.
With both fixes the A/C final frames are pixel-identical on every rig camera
(verified across all 30 cases), so the red-vs-no-red signal is history-only.

  A: blue cube with a RED back face, rotates 360 degrees (red revealed once).
  B: same red-backed cube, rotates 720 degrees at the SAME speed (red revealed
     twice; its clip is twice as long); ends blue-front like A.
  C: all-blue cube (the back plate is blue too), rotates 360 degrees like A —
     no red ever appears.

The final frame is blue-front for all three, so the red-vs-no-red signal lives
in the whole-clip HISTORY: the encoders read the last-second tokens but attend
over the entire clip (bidirectional), so A,B (a red-backed cube was seen) should
separate from C (never any red). Invariant holds if the representation carries
the occluded-then-revealed face rather than only the final frame / rotation count.

30 cases vary speed, spin direction, cube size and position. `--rig stereo` also
writes the right camera.

Usage: python -m evals.occlusion --out-root <dir>   (--check: no render)
"""

import math

import mujoco
import numpy as np

from ..common import cameras
from ..common.family import family_cli
from ..common.observations import write_observations
from ..common.sim import render_cams
from ..common.xml_scene import scene_xml, fr

EVAL_IDS = list(range(30))
SEED_BASE = 7700
BLUE = (0.2, 0.4, 0.9, 1)
RED = (0.85, 0.15, 0.15, 1)
PSI0 = 0.0      # start (and, after full turns, end) blue-front: red plate on
                # the +y back face, occluded; revealed only at the mid-spin.


def cube_body(half, plate_rgba, center):
    p = half * 0.98
    # Flush plate: outer surface 2mm proud of the back face. v1's +0.007
    # offset protruded the rim past the silhouette, leaking plate colour to
    # the angled cameras even when facing away. Scanned 0.3/1/2/3/5mm:
    # <=0.3mm z-fights (plate vanishes in 4/30 cases), >=5mm leaks again;
    # 1-3mm give zero end-frame diff px on both rig cameras in all 30 cases.
    off = half - 0.006 + 0.002
    cx, cy, cz = center
    return (f'<body name="cube" mocap="true" pos="{cx:.3f} {cy:.3f} {cz:.3f}">'
            f'<geom type="box" size="{half} {half} {half}" rgba="{fr(BLUE)}" '
            f'contype="0" conaffinity="0"/>'
            f'<geom type="box" size="{p} 0.006 {p}" pos="0 {off:.3f} 0" '
            f'rgba="{fr(plate_rgba)}" contype="0" conaffinity="0"/></body>')


def _quat_z(angle_deg):
    a = math.radians(angle_deg) / 2.0
    return np.array([math.cos(a), 0.0, 0.0, math.sin(a)])


def build_case(i, fps=25):
    rng = np.random.default_rng([SEED_BASE, i])
    # A/C duration; B is 2T (same speed). Quantised to a whole number of
    # frames so the rendered clips (endpoint included) end EXACTLY on the
    # full turn — v1 stopped one frame short (~356 deg, and A vs B differed).
    T = round(float(rng.uniform(1.6, 2.8)) * fps) / fps
    direction = float(rng.choice([-1.0, 1.0]))
    half = float(rng.uniform(0.15, 0.22))
    center = (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.15, 0.15)), 0.45)
    # (name, duration, plate colour): A/C spin 360 over T, B spins 720 over 2T.
    variants = {
        "A": (T, RED),
        "B": (2.0 * T, RED),
        "C": (T, BLUE),
    }
    return dict(T=T, direction=direction, half=half, center=center), variants


def render_variant(out_dir, name, dur, plate, params, args):
    cam_names = cameras.rig_cameras(args.rig)
    model = mujoco.MjModel.from_xml_string(scene_xml(
        bodies=cube_body(params["half"], plate, params["center"]),
        extra_cameras_xml=cameras.extra_cameras_xml(args.rig)))
    omega = 360.0 / params["T"]               # deg/s, shared by A/B/C
    # Endpoint INCLUDED: T is a whole number of frames, so the final frame
    # lands exactly on the full turn (blue-front) for A, B and C alike.
    times = np.arange(int(round(dur * args.fps)) + 1) / args.fps

    def advance(d, i):
        a = PSI0 + params["direction"] * omega * times[i]
        d.mocap_quat[0] = _quat_z(a)
        mujoco.mj_forward(model, d)

    cam_frames, states = render_cams(
        model, mujoco.MjData(model), cam_names=cam_names, n_frames=len(times),
        advance=advance, size=args.size,
        state_fn=lambda d: d.mocap_quat[0].copy())
    write_observations(out_dir, name, cam_frames, args.fps,
                       state=states, times=times, state_key="cube_quat")


def generate(i, out_dir, args):
    params, variants = build_case(i, args.fps)
    print(f"case_{i:02d}: T={params['T']:.2f}s (A/C 360°, B 720° @ same speed) "
          f"dir={params['direction']:+.0f} half={params['half']:.2f} "
          f"center={tuple(round(c, 2) for c in params['center'])}")
    if out_dir is not None:
        for name, (dur, plate) in variants.items():
            render_variant(out_dir, name, dur, plate, params, args)
    return []


def add_args(parser):
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)


def main():
    family_cli(name="occlusion", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda i: f"case_{i:02d}", description=__doc__,
               add_args=add_args)


if __name__ == "__main__":
    main()
