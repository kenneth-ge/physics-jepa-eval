"""bounce eval family: sensitivity to restitution, PHYSICS, signal in the window.

A real (free-jointed) ball is dropped onto a flat elastic floor, timed so its
FIRST impact and rebound happen during the measured last second. The rebound
height is set by the coefficient of restitution, so the discriminating motion is
right where the encoders read it (the end of the clip) — including single-frame
models, which see the rebound height directly.

  A: restitution r1, dropped at x = c.
  B: IDENTICAL to A (same x, same r1) but its clip has a STILL_PREFIX-second
     prefix where the ball is held at the drop height, i.e. it falls 0.5s later.
  C: restitution r2 = factor*r1, dropped at x = c.

All three share the drop point x = c (position is controlled out). B's measured
last second is identical to A's bounce (the prefix is excluded from the last-
second window) — so A,B are a matched pair differing only in history; A vs C
differ only in restitution (the rebound HEIGHT in the last second). Both readouts
are reported (raw / mean).

Restitution is set via the contact's direct stiffness/damping (solref = -K -B),
with B from an empirical drop-test calibration. We SWEEP r2 = factor*r1 over
0.5x .. 2x (r1=0.45 so factor 2.0 -> r2~0.9, factor 1.0 is the control C == A).
Render/measure via `evals.render_bounce` + `evals.aggregate_bounce`.
"""

import mujoco
import numpy as np

from ..common import cameras
from ..common.observations import write_observations
from ..common.sim import render_cams
from ..common.xml_scene import scene_xml

# --- restitution calibration (measured at K=1e5, timestep 5e-4, flat ground) --
K_STIFF = 1.0e5
_CAL_B = np.array([3, 6, 10, 15, 22, 30, 45, 65, 90, 130, 200], float)
_CAL_E = np.array([0.987, 0.971, 0.951, 0.927, 0.894, 0.857, 0.796, 0.723,
                   0.641, 0.536, 0.397], float)


def damping_for_restitution(e):
    return float(np.interp(e, _CAL_E[::-1], _CAL_B[::-1]))


# --- sweep / geometry -------------------------------------------------------
R1_BASE = 0.45           # base restitution; factor 2.0 -> r2 ~= 0.9 (physical max)
FACTORS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.75, 2.0]
R_CLAMP = (0.20, 0.92)

BALL_R = 0.09
H_FALL = 0.78            # drop height -> first impact ~0.4s, rebound in last sec
DURATION = 1.0          # fall + first rebound (measured window)
STILL_PREFIX = 0.5      # B: seconds the ball is held at drop height before falling
FPS = 50                # finer temporal sampling for the fast bounce
TIMESTEP = 0.0005
C_RANGE = 0.30          # shared drop-point sampled in [-C_RANGE, C_RANGE]
Y_OFF = 0.06


def ball_body(e):
    B = damping_for_restitution(e)
    return (f'<body name="ball" pos="0 0 1"><freejoint/>'
            f'<geom type="sphere" size="{BALL_R}" mass="0.1" priority="1" '
            f'rgba="0.85 0.15 0.15 1" condim="3" friction="0.5 0.005 0.0001" '
            f'solref="-{K_STIFF:.0f} -{B:.2f}" solimp="0.95 0.99 0.001"/></body>')


def build_model(e, rig):
    model = mujoco.MjModel.from_xml_string(scene_xml(
        bodies=ball_body(e), extra_cameras_xml=cameras.extra_cameras_xml(rig),
        timestep=TIMESTEP))
    return model


def build_case(seed, factor, r1=R1_BASE):
    rng = np.random.default_rng([seed, int(round(factor * 1000))])
    r2 = float(np.clip(factor * r1, *R_CLAMP))
    c = float(rng.uniform(-C_RANGE, C_RANGE))
    y0 = float(rng.uniform(-Y_OFF, Y_OFF))
    return dict(r1=r1, r2=r2, c=c, y0=y0)


def _render_one(out_dir, meta, e, x, name, rig, fps, size, duration, still):
    model = build_model(e, rig)
    data = mujoco.MjData(model)
    data.qpos[0:3] = (x, meta["y0"], H_FALL + BALL_R)   # dropped from rest
    cam_names = cameras.rig_cameras(rig)
    spf = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_still = int(round(still * fps))                   # held-still prefix frames
    n_frames = n_still + int(round(duration * fps))

    def advance(d, i):
        if i < n_still:
            mujoco.mj_forward(model, d)                 # held at drop height
        else:
            if i == n_still:
                d.qacc_warmstart[:] = 0                 # fresh start == A's drop
            for _ in range(spf):
                mujoco.mj_step(model, d)

    cam_frames, _ = render_cams(model, data, cam_names=cam_names,
                               n_frames=n_frames, advance=advance, size=size)
    if out_dir is not None:
        write_observations(out_dir, name, cam_frames, fps)
    return data.qpos[0:3].copy(), float(np.linalg.norm(data.qvel[0:3]))


def render_case(out_dir, meta, *, rig="mono", fps=FPS, size=256, duration=DURATION):
    # A,B,C share x=c; B is A with a still prefix (falls STILL_PREFIX s later).
    variants = {"A": (meta["r1"], meta["c"], 0.0),
                "B": (meta["r1"], meta["c"], STILL_PREFIX),
                "C": (meta["r2"], meta["c"], 0.0)}
    return {name: _render_one(out_dir, meta, e, x, name, rig, fps, size,
                              duration, still)
            for name, (e, x, still) in variants.items()}
