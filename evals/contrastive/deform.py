"""deform eval family: sensitivity to material stiffness (kinematic).

A ball arcs down at an angle onto a cube, the cube squashes and springs back,
and the ball rebounds and plunges into a deep void (falls forever, disappears).
The cube's squash amplitude and wobble encode its STIFFNESS; it recovers to its
original shape, so the END IMAGE is identical for all three (original cube + ball
gone). The stiffness signal therefore lives in the deformation history.

  A: cube stiffness k1.
  B: IDENTICAL to A (same k1) but its clip has a still prefix (the ball is held
     before it falls), so it's the matched pair differing only in history.
  C: cube stiffness k2 = factor*k1 (significantly different) -> squashes
     differently, but recovers to the same shape and the ball still ends in the
     void, so the last frame matches A/B.

Kinematic (mocap cube with animated geom_size + mocap ball); MuJoCo's elasticity
plugin isn't available in this env, so the squash is scripted (a damped
oscillation whose amplitude ~ 1/k and frequency ~ sqrt(k)). Sweep factor = k2/k1.
Render via `evals.contrastive.render_deform`.
"""

import math

import mujoco
import numpy as np

from ..common import cameras
from ..common.observations import write_observations
from ..common.sim import render_cams
from ..common.xml_scene import DEFAULT_CAMERA, fr

# --- sweep ------------------------------------------------------------------
K1_BASE = 1.0
FACTORS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.75, 2.0]

# --- geometry / timing ------------------------------------------------------
BALL_R = 0.09
CUBE_S = 0.15               # cube half-size (undeformed)
CUBE_X = -0.42             # cube sits here on the flat
HOLE_X = 0.52             # deep void here
HR = 1.3                  # heightfield half-extent
R_HOLE = 0.16
DEEP = 2.0               # void depth (out of frame -> falls forever)
NGRID = 181
Z_OFF = -DEEP            # hfield: flat top at z=0, void floor at -DEEP

DURATION = 1.6           # A/C clip length
STILL_PREFIX = 0.5       # B: ball held this long before the sequence
FPS = 50
T_IMPACT = 0.45          # ball hits the cube top
T_HOLE = 1.0             # ball reaches the void mouth
START = (CUBE_X - 0.75, 0.0, 0.95)   # ball launch (upper side)

# squash model: amplitude ~ softness (1/k), frequency ~ sqrt(k)
AMP_REF = 0.42
OMEGA_REF = 26.0
DECAY = 6.0
REB_REF = 0.22          # rebound-arc apex gain (stiffer ball bounce = higher)


def _squash_params(k):
    amp = float(np.clip(AMP_REF / k, 0.0, 0.72))
    omega = OMEGA_REF * math.sqrt(k)
    apex = float(np.clip(REB_REF * k, 0.08, 0.45))
    return amp, omega, apex


def cube_size(t, te_imp, amp, omega):
    """(sx,sy,sz) of the cube at time t; squashes at te_imp, recovers."""
    if t < te_imp:
        return CUBE_S, CUBE_S, CUBE_S
    u = t - te_imp
    env = amp * math.exp(-DECAY * u) * math.cos(omega * u)   # +env = compressed
    sz = CUBE_S * (1.0 - env)
    sxy = CUBE_S * (1.0 + 0.45 * env)
    return sxy, sxy, sz


def ball_pos(t, apex):
    """Ball: arc onto the cube top, rebound to the void mouth, plunge in."""
    top = 2 * CUBE_S + BALL_R
    cube_top = np.array([CUBE_X, 0.0, top])
    hole_top = np.array([HOLE_X, 0.0, BALL_R])
    if t <= T_IMPACT:                      # approach: parabola START -> cube top
        s = t / T_IMPACT
        xy = np.array(START) * (1 - s) + cube_top * s
        arc = 0.35 * math.sin(math.pi * s) * 0  # keep approach a clean descent
        return np.array([xy[0], 0.0, START[2] * (1 - s) + top * s + arc])
    if t <= T_HOLE:                        # rebound: cube top -> hole mouth
        s = (t - T_IMPACT) / (T_HOLE - T_IMPACT)
        x = cube_top[0] * (1 - s) + hole_top[0] * s
        z = cube_top[2] * (1 - s) + hole_top[2] * s + apex * math.sin(math.pi * s) * 4
        return np.array([x, 0.0, z])
    s = min((t - T_HOLE) / (DURATION - T_HOLE), 1.0)   # plunge into the void
    ease = s * s
    return np.array([HOLE_X, 0.0, BALL_R + (Z_OFF + BALL_R - BALL_R) * ease])


_XML = """
<mujoco model="deform">
  <option timestep="0.002"/>
  <visual>
    <global offwidth="1024" offheight="1024"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <hfield name="ground" nrow="{n}" ncol="{n}" size="{HR} {HR} {zmax} 0.05"/>
    <material name="ground" rgba="0.72 0.72 0.76 1" specular="0.2" shininess="0.3"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <camera name="fixed" pos="{cpos}" xyaxes="{cxy}" fovy="{fovy}"/>
    {extra_cameras}
    <geom name="ground" type="hfield" hfield="ground" material="ground"
          pos="0 0 {zoff}" contype="0" conaffinity="0"/>
    <body name="cube" mocap="true" pos="{cube_x} 0 {cube_s}">
      <geom name="cube" type="box" size="{cube_s} {cube_s} {cube_s}"
            rgba="0.2 0.5 0.9 1" contype="0" conaffinity="0"/>
    </body>
    <body name="ball" mocap="true" pos="0 0 1">
      <geom type="sphere" size="{r}" rgba="0.85 0.15 0.15 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""


def build_model(rig):
    xml = _XML.format(
        n=NGRID, HR=HR, zmax=DEEP, zoff=Z_OFF, cube_x=CUBE_X, cube_s=CUBE_S,
        r=BALL_R, cpos=fr(DEFAULT_CAMERA["pos"]), cxy=fr(DEFAULT_CAMERA["xyaxes"]),
        fovy=DEFAULT_CAMERA["fovy"], extra_cameras=cameras.extra_cameras_xml(rig))
    m = mujoco.MjModel.from_xml_string(xml)
    xs = np.linspace(-HR, HR, NGRID)
    Xg, Yg = np.meshgrid(xs, xs)
    rho = np.sqrt((Xg - HOLE_X) ** 2 + (Yg) ** 2)
    frac = np.clip(rho / R_HOLE, 0.0, 1.0)          # 0 at hole centre -> deep
    m.hfield_data[:] = (frac * frac * (3 - 2 * frac)).ravel()
    return m


def build_case(seed, factor, k1=K1_BASE):
    return dict(k1=k1, k2=factor * k1)


def _render_one(out_dir, k, name, still, rig, fps, size, duration):
    model = build_model(rig)
    data = mujoco.MjData(model)
    cube_gid = model.geom("cube").id
    amp, omega, apex = _squash_params(k)
    cam_names = cameras.rig_cameras(rig)
    n_still = int(round(still * fps))
    times = np.arange(n_still + int(round(duration * fps))) / fps
    te_imp = T_IMPACT + still            # impact shifted by the still prefix

    def advance(d, i):
        t = times[i]
        teff = max(t - still, 0.0)
        sx, sy, sz = cube_size(t, te_imp, amp, omega)
        model.geom_size[cube_gid] = (sx, sy, sz)
        d.mocap_pos[model.body("cube").mocapid[0]] = (CUBE_X, 0.0, sz)
        d.mocap_pos[model.body("ball").mocapid[0]] = ball_pos(teff, apex)
        mujoco.mj_forward(model, d)

    cam_frames, _ = render_cams(model, data, cam_names=cam_names,
                               n_frames=len(times), advance=advance, size=size)
    if out_dir is not None:
        write_observations(out_dir, name, cam_frames, fps)


def render_case(out_dir, meta, *, rig="mono", fps=FPS, size=256, duration=DURATION):
    variants = {"A": (meta["k1"], 0.0),
                "B": (meta["k1"], STILL_PREFIX),
                "C": (meta["k2"], 0.0)}
    for name, (k, still) in variants.items():
        _render_one(out_dir, k, name, still, rig, fps, size, duration)
