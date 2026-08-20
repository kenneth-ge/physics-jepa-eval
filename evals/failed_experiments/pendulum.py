"""pendulum eval family: sensitivity to MASS via damped-swing dynamics.

A ball on a rigid rod swings under gravity with JOINT DAMPING. The damping torque
is a fixed property of the joint, but the swing inertia scales with mass, so the
damping ratio goes as ~1/mass: a heavier ball loses less amplitude per swing.
Mass is therefore legible ONLY through the dynamics — the ball's visual size is
held constant, only its density changes. (An ideal, undamped pendulum's period is
mass-independent, so the damping is what makes this a mass probe at all.)

  A: mass m, released from rest at the LEFT extreme, filmed for a half-swing so
     it ends on the RIGHT.
  B: mass m (same as A), released from rest at the RIGHT extreme, filmed for a
     full swing (right -> left -> right) so it also ends on the RIGHT — a matched
     pair whose only innocuous difference from A is one extra half-swing of
     history (cf. occlusion's 360 vs 720).
  C: mass factor*m, released from rest at the LEFT extreme, filmed for a half-
     swing (same as A) so it ends on the RIGHT — differs from A only in mass.

Both readouts (raw/mean) reported. Sweep factor = m_C / m over 0.5x..2x
(factor 1.00 = control, C == A). The damping coefficient is fixed per case (set
from the BASE mass), so only C's inertia changes. Render/measure via
`evals.contrastive.render_pendulum` + `evals.contrastive.aggregate_bounce`
(shared seed*/factor_* layout).
"""

import math

import mujoco
import numpy as np

from ..common import cameras
from ..common.observations import write_observations
from ..common.sim import render_cams
from ..common.xml_scene import scene_xml

G = 9.81
FACTORS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.75, 2.0]

BALL_R = 0.11            # fixed visual size for every mass (density varies)
ROD_R = 0.02
PIVOT_Z = 1.35
FPS = 50
TIMESTEP = 0.001
ZETA_BASE = 0.22        # damping ratio at the base mass (fixes the joint damping)


def damping_for(m, L):
    """Joint damping giving ZETA_BASE at mass m, length L (fixed per case)."""
    w = math.sqrt(G / L)
    return 2.0 * ZETA_BASE * m * L * L * w


def half_period(L, theta0):
    """Half swing time incl. the leading finite-amplitude correction."""
    w = math.sqrt(G / L)
    return (math.pi / w) * (1.0 + theta0 ** 2 / 16.0)


def pend_body(m, b, L):
    return (f'<body name="pend" pos="0 0 {PIVOT_Z}">'
            f'<joint name="hinge" type="hinge" axis="0 1 0" damping="{b:.5f}"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 {-L}" size="{ROD_R}" '
            f'mass="0.001" rgba="0.3 0.3 0.35 1"/>'
            f'<geom type="sphere" pos="0 0 {-L}" size="{BALL_R}" mass="{m}" '
            f'rgba="0.85 0.15 0.15 1"/></body>')


def build_model(m, b, L, rig):
    return mujoco.MjModel.from_xml_string(scene_xml(
        bodies=pend_body(m, b, L), extra_cameras_xml=cameras.extra_cameras_xml(rig),
        timestep=TIMESTEP))


def build_case(seed, factor):
    # per-seed pendulum so seeds differ; C = factor * base mass.
    rng = np.random.default_rng([seed, int(round(factor * 1000))])
    L = float(rng.uniform(0.80, 1.00))
    theta0 = float(rng.uniform(0.70, 0.95))   # release amplitude (rad)
    m = float(rng.uniform(0.10, 0.20))
    return dict(L=L, theta0=theta0, m=m, m2=factor * m)


def _render_one(out_dir, meta, m, theta0_signed, duration, name, rig, fps, size):
    # damping is fixed from the BASE mass (joint property); only C's mass changes.
    b = damping_for(meta["m"], meta["L"])
    model = build_model(m, b, meta["L"], rig)
    data = mujoco.MjData(model)
    adr = model.joint("hinge").qposadr[0]
    data.qpos[adr] = theta0_signed            # released from rest
    cam_names = cameras.rig_cameras(rig)
    spf = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_frames = int(round(duration * fps))

    def advance(d, i):
        if i == 0:
            mujoco.mj_forward(model, d)        # frame 0 = release pose
        else:
            for _ in range(spf):
                mujoco.mj_step(model, d)

    cam_frames, _ = render_cams(model, data, cam_names=cam_names,
                               n_frames=n_frames, advance=advance, size=size)
    if out_dir is not None:
        write_observations(out_dir, name, cam_frames, fps)
    return float(data.qpos[adr])


def render_case(out_dir, meta, *, rig="mono", fps=FPS, size=256):
    # +theta0 => ball on the LEFT (x = -L sin θ); the swing ends on the RIGHT.
    th = meta["theta0"]
    half = half_period(meta["L"], th)
    variants = {"A": (meta["m"], +th, half),        # left  -> right (half swing)
                "B": (meta["m"], -th, 2.0 * half),  # right -> left -> right (full)
                "C": (meta["m2"], +th, half)}       # left  -> right, heavier
    return {name: _render_one(out_dir, meta, m, th_s, dur, name, rig, fps, size)
            for name, (m, th_s, dur) in variants.items()}
