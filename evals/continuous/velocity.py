"""velocity sweep family: one ball rolls at N_POINTS speeds, theta = speed.

Each case is one ball on flat ground rolling along x at a constant speed
(low rolling resistance + matched spin, as in the contrastive roll family).
The ladder sweeps speed v_00 < .. < v_15; every rung rolls for the SAME
duration and ENDS at the same point E (start = E - v*T), so the final
position is controlled out and theta is carried by motion: how fast the ball
translates through the window, and where it sits at each instant BEFORE the
common endpoint. A position-only representation of the final frame sees all
rungs alike; ranking the ladder needs speed (position-over-time).

Cases vary the roll direction, the endpoint E, the lateral offset and the
ball's hue. Grading is the standard nn ladder adjacency (`nn_cos`/`nn_l1` in
evals.continuous.measure): for every interior rung its two embedding nearest
neighbors should be the adjacent speeds.

Contract (--check, simulates without rendering): speeds strictly increasing
and evenly spaced; every trajectory stays inside the frame bound; each rung
ends within END_TOL of E still moving at ~its nominal speed (low-friction
rolling keeps decay small).

Usage: python -m evals.continuous.velocity --out-root <dir>   (--check: no render)
Outputs <dir>/case_00 .. case_09, each with v_00..v_15.mp4, manifest.json and
grid.png (first-frame contact sheet: the ladder of start positions).
"""

import colorsys

import mujoco
import numpy as np

from ..common import cameras
from ..common.family import family_cli
from ..common.manifest import write_manifest
from ..common.observations import write_observations
from ..common.sim import BallTracker, render_cams, rolling_spin, rollout
from ..common.video import save_video
from ..common.xml_scene import scene_xml
from .grid import COLS as GRID_COLS, save_grid

EVAL_IDS = list(range(10))
SEED_BASE = 7500

N_POINTS = 16
V_RANGE = (0.10, 0.70)  # m/s, 0.04 m/s rungs; slowest rung still clearly rolls
BALL_R = 0.10
XLIM = 1.00             # endpoints kept within +/- this
FRAME_X = 1.25          # hard in-frame bound (contract, as in contrastive roll)
E_JITTER = 0.12         # endpoint jitter beyond the centred sweep segment
Y_OFF = 0.15
MU = (1.0, 0.005, 0.0005)   # low roll resistance => ~constant speed
DAMPING = 0.0
END_TOL = 0.12          # |final_x - E| tolerance (friction decay undershoot)
SPEED_TOL = 0.25        # final_speed within this fraction of nominal v


def sweep_speeds():
    return np.linspace(V_RANGE[0], V_RANGE[1], N_POINTS)


def build_case(i, args):
    rng = np.random.default_rng([SEED_BASE, i])
    sign = float(rng.choice([-1.0, 1.0]))
    # Fastest rung sweeps a segment of length v_max*T; centre it (plus jitter)
    # so every start point and the shared endpoint stay inside +/- XLIM.
    span = V_RANGE[1] * args.duration
    E = sign * (span / 2.0 + float(rng.uniform(-E_JITTER, E_JITTER)))
    y0 = float(rng.uniform(-Y_OFF, Y_OFF))
    hue = float(rng.uniform(0.0, 1.0))
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.85)
    return dict(E=E, sign=sign, y0=y0, hue=hue, rgba=(*rgb, 1.0))


def ball_body(rgba):
    return (f'<body name="ball" pos="0 0 1">'
            f'<joint name="ball" type="free" damping="{DAMPING}"/>'
            f'<geom name="ball" type="sphere" size="{BALL_R}" mass="0.1" '
            f'rgba="{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} 1" condim="6" '
            f'friction="{MU[0]} {MU[1]} {MU[2]}"/></body>')


def simulate(model, x0, y0, vx, dur, fps, size,
             out_dir=None, name=None, cam_names=(cameras.PRIMARY,)):
    data = mujoco.MjData(model)
    data.qpos[0:3] = (x0, y0, BALL_R + 0.001)
    data.qvel[0:3] = (vx, 0.0, 0.0)
    data.qvel[3:6] = rolling_spin((vx, 0.0, 0.0), BALL_R)
    tracker = BallTracker(model)
    render = out_dir is not None

    if not render or tuple(cam_names) == (cameras.PRIMARY,):
        frames = rollout(model, data, duration=dur, fps=fps, size=size,
                         post_step=tracker.post_step, render=render)
        if render:
            save_video(frames, out_dir / f"{name}.mp4", fps)
            return tracker.summary(dur), frames[0]
        return tracker.summary(dur), None

    spf = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_frames = int(dur / model.opt.timestep) // spf

    def advance(d_, _i):
        for _ in range(spf):
            mujoco.mj_step(model, d_)
            tracker.post_step(d_)

    cam_frames, _ = render_cams(model, data, cam_names=list(cam_names),
                                n_frames=n_frames, advance=advance, size=size)
    write_observations(out_dir, name, cam_frames, fps)
    return tracker.summary(dur), cam_frames[cameras.PRIMARY][0]


def check_contract(case, results, args):
    f = []
    vs = sweep_speeds()
    if not (np.all(np.diff(vs) > 0) and np.allclose(np.diff(vs), vs[1] - vs[0])):
        f.append("speeds not strictly increasing / evenly spaced")
    for k, (v, r) in enumerate(zip(vs, results)):
        x0 = case["E"] - case["sign"] * v * args.duration
        if abs(x0) > XLIM or abs(case["E"]) > XLIM:
            f.append(f"v_{k:02d}: endpoint outside +/-{XLIM} "
                     f"(x0={x0:+.2f}, E={case['E']:+.2f})")
        if r["min_x"] < -FRAME_X or r["max_x"] > FRAME_X:
            f.append(f"v_{k:02d}: leaves frame "
                     f"(x in [{r['min_x']:.2f}, {r['max_x']:.2f}])")
        if abs(r["final_pos"][0] - case["E"]) > END_TOL:
            f.append(f"v_{k:02d}: end {r['final_pos'][0]:+.3f} != "
                     f"E={case['E']:+.3f}")
        if abs(r["final_speed"] - v) > SPEED_TOL * v:
            f.append(f"v_{k:02d}: final speed {r['final_speed']:.3f} != "
                     f"nominal {v:.3f}")
    return f


def generate(i, out_dir, args):
    case = build_case(i, args)
    model = mujoco.MjModel.from_xml_string(scene_xml(
        bodies=ball_body(case["rgba"]), ground_friction=MU,
        extra_cameras_xml=cameras.extra_cameras_xml(args.rig)))
    cam_names = cameras.rig_cameras(args.rig)

    vs = sweep_speeds()
    results, clips, stills, labels = [], [], [], []
    for k, v in enumerate(vs):
        x0 = case["E"] - case["sign"] * v * args.duration
        res, still = simulate(model, x0, case["y0"], case["sign"] * float(v),
                              args.duration, args.fps, args.size,
                              out_dir=out_dir, name=f"v_{k:02d}",
                              cam_names=cam_names)
        results.append(res)
        clips.append({"file": f"v_{k:02d}.mp4", "theta": float(v)})
        if still is not None:
            stills.append(still)
            labels.append(f"{k:02d}  v={v:.2f}")

    failures = check_contract(case, results, args)
    print(f"case_{i:02d}: E={case['E']:+.2f} dir={case['sign']:+.0f} "
          f"y0={case['y0']:+.2f} v={vs[0]:.2f}..{vs[-1]:.2f} | "
          f"end x {results[0]['final_pos'][0]:+.2f}/{results[-1]['final_pos'][0]:+.2f} "
          f"end speed {results[0]['final_speed']:.2f}/{results[-1]['final_speed']:.2f}")
    if out_dir is None:
        return failures

    write_manifest(out_dir, "roll_speed_mps", clips,
                   endpoint=case["E"], direction=case["sign"],
                   duration=args.duration)
    if not args.no_grid:
        rows = save_grid(stills, labels, out_dir / "grid.png")
        print(f"  wrote grid.png ({rows}x{GRID_COLS})")
    return failures


def add_args(parser):
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--no-grid", action="store_true",
                        help="skip the per-case grid.png contact sheet")


def main():
    family_cli(name="velocity", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda i: f"case_{i:02d}", description=__doc__,
               add_args=add_args)


if __name__ == "__main__":
    main()
