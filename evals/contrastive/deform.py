"""deform eval family: sensitivity to material stiffness (kinematic).

A ball arcs down at an angle onto a cube; the cube squashes and springs back
(amplitude ~ 1/stiffness, wobble frequency ~ sqrt(stiffness)); the ball rebounds
toward a deep void. The cube's squash encodes its STIFFNESS.

Two variants (choose via render_deform --variant):
  history  : the cube fully recovers and the ball plunges into the void, so the
             END IMAGE is identical across A/B/C — the stiffness signal lives only
             in the deformation history (like bounce_void / occlusion).
  adjusted : the impact happens late and recovery is slow, so the cube is still
             visibly deformed in the LAST SECOND — the end image differs by
             stiffness and the signal is in the readout window (measurable by all
             models, incl. single-frame ones).

  A: cube stiffness k1.
  B: same k1 as A but with a still prefix (ball held before it falls) — matched
     pair differing only in history.
  C: cube stiffness k2 = factor*k1 (significantly different).

Kinematic (mocap cube with animated geom_size + mocap ball); MuJoCo's elasticity
plugin isn't available in this env, so the squash is scripted. Sweep factor=k2/k1.
Render via `evals.contrastive.render_deform`.
"""

import math

import mujoco
import numpy as np

from ..common import cameras
from ..common.observations import write_observations
from ..common.sim import render_cams
from ..common.xml_scene import DEFAULT_CAMERA, fr

K1_BASE = 1.0
FACTORS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.75, 2.0]

BALL_R = 0.09
CUBE_S = 0.15
CUBE_X = -0.42
HOLE_X = 0.52
HR = 1.3
R_HOLE = 0.16
DEEP = 2.0
NGRID = 181
Z_OFF = -DEEP
FPS = 50
START = (CUBE_X - 0.75, 0.0, 0.95)

AMP_REF = 0.42
OMEGA_REF = 26.0
REB_REF = 0.22

# variant timing: history = recover + plunge (same end image);
# adjusted = late impact, slow recovery, ball still mid-arc (signal in window).
VARIANTS = {
    "history":  dict(t_impact=0.45, t_hole=1.00, duration=1.6, decay=6.0,
                     plunge=True,  still=0.5),
    "adjusted": dict(t_impact=0.50, t_hole=1.60, duration=1.0, decay=2.0,
                     plunge=False, still=0.5),
}


def _squash_params(k):
    amp = float(np.clip(AMP_REF / k, 0.0, 0.72))
    omega = OMEGA_REF * math.sqrt(k)
    apex = float(np.clip(REB_REF * k, 0.08, 0.45))
    return amp, omega, apex


def cube_size(t, te_imp, amp, omega, decay):
    if t < te_imp:
        return CUBE_S, CUBE_S, CUBE_S
    u = t - te_imp
    env = amp * math.exp(-decay * u) * math.cos(omega * u)
    return CUBE_S * (1 + 0.45 * env), CUBE_S * (1 + 0.45 * env), CUBE_S * (1 - env)


def ball_pos(t, apex, p):
    top = 2 * CUBE_S + BALL_R
    cube_top = np.array([CUBE_X, 0.0, top])
    hole_top = np.array([HOLE_X, 0.0, BALL_R])
    if t <= p["t_impact"]:
        s = t / p["t_impact"]
        return np.array([START[0] * (1 - s) + CUBE_X * s, 0.0,
                         START[2] * (1 - s) + top * s])
    if t <= p["t_hole"]:
        s = (t - p["t_impact"]) / (p["t_hole"] - p["t_impact"])
        x = cube_top[0] * (1 - s) + hole_top[0] * s
        z = cube_top[2] * (1 - s) + hole_top[2] * s + apex * math.sin(math.pi * s) * 4
        return np.array([x, 0.0, z])
    s = min((t - p["t_hole"]) / max(p["duration"] - p["t_hole"], 1e-3), 1.0)
    return np.array([HOLE_X, 0.0, BALL_R + (Z_OFF) * (s * s)])


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
    rho = np.sqrt((Xg - HOLE_X) ** 2 + Yg ** 2)
    frac = np.clip(rho / R_HOLE, 0.0, 1.0)
    m.hfield_data[:] = (frac * frac * (3 - 2 * frac)).ravel()
    return m


def build_case(seed, factor, k1=None):
    # per-seed base stiffness so the seeds aren't identical; C = factor * k1.
    rng = np.random.default_rng([seed, int(round(factor * 1000))])
    base = k1 if k1 is not None else float(rng.uniform(0.7, 1.4))
    return dict(k1=base, k2=factor * base)


def _render_one(out_dir, k, name, still, p, rig, fps, size):
    model = build_model(rig)
    data = mujoco.MjData(model)
    cube_gid = model.geom("cube").id
    cube_mid = model.body("cube").mocapid[0]
    ball_mid = model.body("ball").mocapid[0]
    amp, omega, apex = _squash_params(k)
    cam_names = cameras.rig_cameras(rig)
    n_still = int(round(still * fps))
    times = np.arange(n_still + int(round(p["duration"] * fps))) / fps
    te_imp = p["t_impact"] + still

    def advance(d, i):
        t = times[i]
        teff = max(t - still, 0.0)
        sx, sy, sz = cube_size(t, te_imp, amp, omega, p["decay"])
        model.geom_size[cube_gid] = (sx, sy, sz)
        d.mocap_pos[cube_mid] = (CUBE_X, 0.0, sz)
        d.mocap_pos[ball_mid] = ball_pos(teff, apex, p)
        mujoco.mj_forward(model, d)

    cam_frames, _ = render_cams(model, data, cam_names=cam_names,
                               n_frames=len(times), advance=advance, size=size)
    if out_dir is not None:
        write_observations(out_dir, name, cam_frames, fps)


def render_case(out_dir, meta, *, variant="history", rig="mono", fps=FPS, size=256):
    p = VARIANTS[variant]
    variants = {"A": (meta["k1"], 0.0),
                "B": (meta["k1"], p["still"]),
                "C": (meta["k2"], 0.0)}
    for name, (k, still) in variants.items():
        _render_one(out_dir, k, name, still, p, rig, fps, size)
