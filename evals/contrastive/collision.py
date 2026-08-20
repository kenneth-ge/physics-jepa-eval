"""collision eval family: sensitivity to MASS via momentum transfer.

A red ball rolls in from the LEFT and strikes a stationary blue target on the
RIGHT. The collision outcome — how much the incoming ball rebounds vs how far the
target is driven — is set by the mass RATIO, so mass is legible only through the
dynamics: both balls have the same radius, only the target's density changes.

  A: incoming ball reaches the target with impact momentum p; short lead-in.
  B: the SAME collision as A (same impact momentum p, same target mass), but the
     clip starts earlier and further left, so we watch the incoming ball roll and
     shed momentum before impact — a matched pair differing only in how much
     pre-impact history is shown (cf. bounce's still prefix / occlusion 360-720).
     The incoming ball is launched faster so that, after more rolling friction, it
     still arrives with momentum p.
  C: the SAME as A (same lead-in, same impact momentum) but the target is much
     more massive, so the incoming ball rebounds and the target barely moves.

Impact momentum is matched by calibrating each launch speed (bisection over the
free roll to the impact point). Sweep factor = m_target_C / m_target_A over
0.5x..2x (factor 1.00 = control, C == A). Both readouts reported. Render/measure
via `evals.contrastive.render_collision` + `evals.contrastive.aggregate_bounce`.
"""

import mujoco
import numpy as np

from ..common import cameras
from ..common.observations import write_observations
from ..common.sim import render_cams
from ..common.xml_scene import fr, scene_xml

FACTORS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.75, 2.0]

BALL_R = 0.11
M_IN = 0.15                       # incoming ball mass (fixed)
X_TGT = 0.80                      # target rest x (right side)
IMPACT_X = X_TGT - 2 * BALL_R     # incoming centre x at first contact
D_A = 0.45                        # A/C lead-in distance
D_B = 1.30                        # B lead-in distance (longer roll)
V_IMP = 1.8                       # matched impact speed (m/s)
T_AFTER = 0.7                     # aftermath seconds rendered after impact
FPS = 50
TIMESTEP = 0.0005
ROLL_FRICTION = (1.0, 0.02, 0.04)  # slide, spin, roll -> gradual momentum loss
STOP_SPEED = 0.02


def ball_body(name, x, m, rgba):
    return (f'<body name="{name}" pos="{x:.4f} 0 {BALL_R}"><freejoint/>'
            f'<geom type="sphere" size="{BALL_R}" mass="{m}" rgba="{rgba}" '
            f'priority="1" condim="6" friction="{fr(ROLL_FRICTION)}" '
            f'solref="-40000 -50" solimp="0.9 0.95 0.001"/></body>')


def build_model(m_tgt, x_in, rig, x_tgt=X_TGT):
    bodies = (ball_body("incoming", x_in, M_IN, "0.85 0.15 0.15 1")
              + ball_body("target", x_tgt, m_tgt, "0.15 0.35 0.9 1"))
    return mujoco.MjModel.from_xml_string(scene_xml(
        bodies=bodies, extra_cameras_xml=cameras.extra_cameras_xml(rig),
        timestep=TIMESTEP, ground_friction=ROLL_FRICTION))


def _launch(data, v0):
    data.qvel[0] = v0                 # +x linear velocity
    data.qvel[4] = v0 / BALL_R        # rolling-without-slipping spin about +y


def _roll_to(x_in, v0, x_stop, max_t=3.0):
    """Free-roll the incoming ball (target parked far away); return (speed, time)
    when its centre first reaches x_stop, or None if it stops short."""
    model = build_model(1.0, x_in, "mono", x_tgt=20.0)
    data = mujoco.MjData(model)
    _launch(data, v0)
    while data.time < max_t:
        mujoco.mj_step(model, data)
        if data.qpos[0] >= x_stop:
            return float(np.linalg.norm(data.qvel[0:3])), float(data.time)
        if np.linalg.norm(data.qvel[0:3]) < STOP_SPEED:
            return None
    return None


def _calibrate(x_in, iters=20):
    """Launch speed so the incoming ball arrives at IMPACT_X with speed V_IMP."""
    lo, hi = V_IMP, V_IMP + 6.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = _roll_to(x_in, mid, IMPACT_X)
        if res is None or res[0] < V_IMP:
            lo = mid
        else:
            hi = mid
    speed, t_roll = _roll_to(x_in, hi, IMPACT_X)
    return hi, t_roll


def build_case(seed, factor):
    rng = np.random.default_rng([seed, int(round(factor * 1000))])
    base_ratio = 0.5 if rng.random() < 0.5 else 2.0  # target 2x lighter/heavier
    m_tgt = M_IN * base_ratio
    x_A = IMPACT_X - (D_A + float(rng.uniform(-0.05, 0.05)))
    x_B = IMPACT_X - (D_B + float(rng.uniform(-0.10, 0.10)))
    return dict(x_A=x_A, x_B=x_B, m_tgt=m_tgt, m_tgt2=factor * m_tgt)


def _render_one(out_dir, x_in, v0, m_tgt, t_roll, name, rig, fps, size):
    model = build_model(m_tgt, x_in, rig)
    data = mujoco.MjData(model)
    _launch(data, v0)
    cam_names = cameras.rig_cameras(rig)
    spf = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_frames = int(round((t_roll + T_AFTER) * fps))

    def advance(d, i):
        if i == 0:
            mujoco.mj_forward(model, d)     # frame 0 = launch pose
        else:
            for _ in range(spf):
                mujoco.mj_step(model, d)

    cam_frames, _ = render_cams(model, data, cam_names=cam_names,
                               n_frames=n_frames, advance=advance, size=size)
    if out_dir is not None:
        write_observations(out_dir, name, cam_frames, fps)


def render_case(out_dir, meta, *, rig="mono", fps=FPS, size=256):
    # calibrate the two lead-ins once (pre-impact roll is target-mass independent)
    v0_A, tr_A = _calibrate(meta["x_A"])
    v0_B, tr_B = _calibrate(meta["x_B"])
    _render_one(out_dir, meta["x_A"], v0_A, meta["m_tgt"], tr_A, "A", rig, fps, size)
    _render_one(out_dir, meta["x_B"], v0_B, meta["m_tgt"], tr_B, "B", rig, fps, size)
    _render_one(out_dir, meta["x_A"], v0_A, meta["m_tgt2"], tr_A, "C", rig, fps, size)
