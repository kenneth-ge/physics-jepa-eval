"""roll eval family: motion vs. static at a controlled position.

One ball on flat ground, filmed for a fixed window; the last second is measured.
Let E = X + delta be the common end point.

  A: ball rolls from E-delta  to E (travels delta,  still moving at the end).
  B: ball rolls from E-2*delta to E (travels 2*delta at the SAME speed, so its
     clip is twice as long); ends at E moving at the same speed as A.
  C: ball stays still at E for the whole clip.

All three END at the same position E, so position is controlled out. A and B are
rolling through E at the same speed (their last second is identical, only the
history differs); C is stationary at E. Invariant (A,B closer than C) therefore
holds iff the representation encodes MOTION — a static-vs-moving distinction at a
fixed location. Both readouts are reported (raw / mean).

30 cases vary E, the signed step delta, and the lateral offset; every trajectory
is kept in frame. `--rig stereo` also writes the right camera.

Usage: python -m evals.roll --out-root <dir>   (--check: no render)
"""

import mujoco
import numpy as np

from ..common import cameras
from ..common.family import family_cli
from ..common.observations import write_observations
from ..common.sim import BallTracker, rolling_spin, rollout, render_cams
from ..common.video import save_video
from ..common.xml_scene import scene_xml

EVAL_IDS = list(range(30))
SEED_BASE = 6400

BALL_R = 0.10
XLIM = 1.00             # endpoints kept within +/- this
FRAME_X = 1.25          # hard in-frame bound (contract)
D_RANGE = (0.40, 0.70)  # |delta| (A's roll distance; B rolls 2*delta)
Y_OFF = 0.15
MU = (1.0, 0.005, 0.0005)   # low roll resistance => ~constant speed
DAMPING = 0.0
MOVE_SPEED_MIN = 0.10   # A/B must still be moving this fast at the end


def ball_body():
    return (f'<body name="ball" pos="0 0 1">'
            f'<joint name="ball" type="free" damping="{DAMPING}"/>'
            f'<geom name="ball" type="sphere" size="{BALL_R}" mass="0.1" '
            f'rgba="0.85 0.15 0.15 1" condim="6" '
            f'friction="{MU[0]} {MU[1]} {MU[2]}"/></body>')


def build_case(i, args):
    rng = np.random.default_rng([SEED_BASE, i])
    d = float(rng.uniform(*D_RANGE)) * float(rng.choice([-1.0, 1.0]))
    lo, hi = max(-XLIM, -XLIM + 2 * d), min(XLIM, XLIM + 2 * d)
    E = float(rng.uniform(lo, hi))            # common end point (= X + delta)
    y0 = float(rng.uniform(-Y_OFF, Y_OFF))
    v = d / args.duration                     # shared speed of A and B
    z0 = BALL_R + 0.001
    return {
        # (start_x, roll speed, duration, moving?)
        "A": dict(x0=E - d,     vx=v,   dur=args.duration,       moving=True),
        "B": dict(x0=E - 2 * d, vx=v,   dur=2.0 * args.duration, moving=True),
        "C": dict(x0=E,         vx=0.0, dur=args.duration,       moving=False),
    }, dict(d=d, E=E, y0=y0, v=v), z0


def simulate(model, x0, y0, z0, vx, dur, moving, fps, size,
             out_dir=None, name=None, cam_names=(cameras.PRIMARY,)):
    data = mujoco.MjData(model)
    data.qpos[0:3] = (x0, y0, z0)
    if moving:
        data.qvel[0:3] = (vx, 0.0, 0.0)
        data.qvel[3:6] = rolling_spin((vx, 0.0, 0.0), BALL_R)
    tracker = BallTracker(model)
    render = out_dir is not None

    if not render or tuple(cam_names) == (cameras.PRIMARY,):
        frames = rollout(model, data, duration=dur, fps=fps, size=size,
                         post_step=tracker.post_step, render=render)
        if render:
            save_video(frames, out_dir / f"{name}.mp4", fps)
        return tracker.summary(dur)

    spf = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_frames = int(dur / model.opt.timestep) // spf

    def advance(d_, _i):
        for _ in range(spf):
            mujoco.mj_step(model, d_)
            tracker.post_step(d_)

    cam_frames, _ = render_cams(model, data, cam_names=list(cam_names),
                               n_frames=n_frames, advance=advance, size=size)
    write_observations(out_dir, name, cam_frames, fps)
    return tracker.summary(dur)


def check_contract(res, meta):
    f = []
    a, b, c = res["A"], res["B"], res["C"]
    if a["final_speed"] < MOVE_SPEED_MIN:
        f.append(f"A not moving at end (speed={a['final_speed']:.3f})")
    if b["final_speed"] < MOVE_SPEED_MIN:
        f.append(f"B not moving at end (speed={b['final_speed']:.3f})")
    if c["final_speed"] > 0.02:
        f.append(f"C not still (speed={c['final_speed']:.3f})")
    # all three share the end point E
    for n, r in res.items():
        if abs(r["final_pos"][0] - meta["E"]) > 0.12:
            f.append(f"{n} end {r['final_pos'][0]:.3f} != E={meta['E']:.3f}")
        if r["min_x"] < -FRAME_X or r["max_x"] > FRAME_X:
            f.append(f"{n}: leaves frame (x in [{r['min_x']:.2f}, {r['max_x']:.2f}])")
    return f


def generate(i, out_dir, args):
    scenes, meta, z0 = build_case(i, args)
    model = mujoco.MjModel.from_xml_string(scene_xml(
        bodies=ball_body(), ground_friction=MU,
        extra_cameras_xml=cameras.extra_cameras_xml(args.rig)))
    cam_names = cameras.rig_cameras(args.rig)
    res = {}
    for name, ic in scenes.items():
        res[name] = simulate(model, ic["x0"], meta["y0"], z0, ic["vx"],
                             ic["dur"], ic["moving"], args.fps, args.size,
                             out_dir=out_dir, name=name, cam_names=cam_names)
    failures = check_contract(res, meta)
    print(f"case_{i:02d}: E={meta['E']:+.2f} d={meta['d']:+.2f} v={meta['v']:+.2f} | "
          f"A_end={res['A']['final_pos'][0]:+.2f}({res['A']['final_speed']:.2f}) "
          f"B_end={res['B']['final_pos'][0]:+.2f}({res['B']['final_speed']:.2f}) "
          f"C_end={res['C']['final_pos'][0]:+.2f}({res['C']['final_speed']:.2f})")
    return failures


def add_args(parser):
    parser.add_argument("--duration", type=float, default=2.0)  # A/C; B is 2x
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)


def main():
    family_cli(name="roll", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda i: f"case_{i:02d}", description=__doc__,
               add_args=add_args)


if __name__ == "__main__":
    main()
