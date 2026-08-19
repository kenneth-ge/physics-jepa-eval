"""bounce_void: the ORIGINAL void design (signal lives in the bounce HISTORY).

Same restitution sweep as `bounce`, but the ball bounces on a gently-sloped
rubber-sheet (restitution history = the signal), funnels inward, and PLUNGES
into a deep ball-sized cutout (~2.5 m, out of frame, absorbing floor) — so the
measured last second is an EMPTY sheet for every variant. The restitution
difference is therefore only recoverable by a model whose last-second latents
attend back to the earlier bounce frames — i.e. whole-clip encoders
(V-JEPA2, Qwen). Cosmos (last 5 frames) and FastWAM (last frame) never receive
the bounce, so this variant is run with V-JEPA2 + Qwen ONLY.

Compare against `bounce` (the retimed version, where the rebound is in the last
second and all four models can see it). Render/measure via
`evals.render_bounce_void` + `evals.aggregate_bounce`.
"""

import mujoco
import numpy as np

from ..common import cameras
from ..common.observations import write_observations
from ..common.sim import render_cams
from ..common.xml_scene import DEFAULT_CAMERA, fr

K_STIFF = 1.0e5
_CAL_B = np.array([3, 6, 10, 15, 22, 30, 45, 65, 90, 130, 200], float)
_CAL_E = np.array([0.987, 0.971, 0.951, 0.927, 0.894, 0.857, 0.796, 0.723,
                   0.641, 0.536, 0.397], float)


def damping_for_restitution(e):
    return float(np.interp(e, _CAL_E[::-1], _CAL_B[::-1]))


R1_BASE = 0.45
FACTORS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.75, 2.0]
R_CLAMP = (0.20, 0.92)

BALL_R = 0.09
HR = 1.15
R_HOLE = 0.20            # deep central cutout radius
D_RIM = 0.55             # sheet depth at the hole rim (funnel steepens into it)
P_FUNNEL = 2.0           # gentle far out, steep near the hole
DEEP = 2.5               # cutout depth (out of frame; the "void")
NGRID = 221
H0 = 0.24
V_APPROACH = 0.10
DURATION = 3.0
FPS = 25
TIMESTEP = 0.0005
OFF_RANGE = (0.55, 0.98)
MIN_XZ_SEP = 0.45
Y_OFF = 0.08

WELL_DEPTH = DEEP
Z_OFF = -DEEP


def _profile01(rho):
    frac = np.clip((HR - rho) / (HR - R_HOLE), 0.0, 1.0)
    outer = D_RIM * frac ** P_FUNNEL
    depth = np.where(rho < R_HOLE, DEEP, outer)
    return np.clip(1.0 - depth / DEEP, 0.0, 1.0)


_XML = """
<mujoco model="bounce_void">
  <option timestep="{ts}"/>
  <visual>
    <global offwidth="1024" offheight="1024"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <hfield name="well" nrow="{n}" ncol="{n}" size="{HR} {HR} {zmax} 0.05"/>
    <material name="ground" rgba="0.72 0.72 0.76 1" specular="0.2" shininess="0.3"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <camera name="fixed" pos="{cpos}" xyaxes="{cxy}" fovy="{fovy}"/>
    {extra_cameras}
    <geom name="well" type="hfield" hfield="well" material="ground"
          pos="0 0 {zoff}" condim="6" friction="1 0.03 0.006"/>
    <geom name="void" type="cylinder" size="{hole_r} 0.01" pos="0 0 {floorz}"
          rgba="0.05 0.05 0.06 1" condim="3" friction="1 0.3 0.2" priority="2"
          solref="-{K:.0f} -400" solimp="0.95 0.99 0.001"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="{r}" mass="0.1" priority="1"
            rgba="0.85 0.15 0.15 1" condim="6" friction="1 0.03 0.006"
            solref="-{K:.0f} -{B:.2f}" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>"""


def build_model(e, rig):
    B = damping_for_restitution(e)
    xml = _XML.format(
        ts=TIMESTEP, n=NGRID, HR=HR, zmax=WELL_DEPTH, zoff=Z_OFF,
        floorz=Z_OFF + 0.01, hole_r=R_HOLE - BALL_R * 0.3,
        cpos=fr(DEFAULT_CAMERA["pos"]), cxy=fr(DEFAULT_CAMERA["xyaxes"]),
        fovy=DEFAULT_CAMERA["fovy"],
        extra_cameras=cameras.extra_cameras_xml(rig), r=BALL_R, K=K_STIFF, B=B)
    model = mujoco.MjModel.from_xml_string(xml)
    xs = np.linspace(-HR, HR, NGRID)
    Xg, Yg = np.meshgrid(xs, xs)
    model.hfield_data[:] = _profile01(np.sqrt(Xg**2 + Yg**2)).ravel()
    return model


def _surface_z(x):
    return Z_OFF + WELL_DEPTH * float(_profile01(np.array(abs(x))))


def build_case(seed, factor, r1=R1_BASE):
    rng = np.random.default_rng([seed, int(round(factor * 1000))])
    r2 = float(np.clip(factor * r1, *R_CLAMP))

    def off():
        return float(rng.choice([-1.0, 1.0]) * rng.uniform(*OFF_RANGE))

    X = off()
    Z = off()
    while abs(X - Z) < MIN_XZ_SEP:
        Z = off()
    y0 = float(rng.uniform(-Y_OFF, Y_OFF))
    return dict(r1=r1, r2=r2, X=X, Z=Z, y0=y0)


def _render_one(out_dir, meta, e, start_x, name, rig, fps, size, duration):
    model = build_model(e, rig)
    data = mujoco.MjData(model)
    data.qpos[0:3] = (start_x, meta["y0"], _surface_z(start_x) + BALL_R + H0)
    data.qvel[0] = -np.sign(start_x) * V_APPROACH
    cam_names = cameras.rig_cameras(rig)
    spf = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_frames = int(duration / model.opt.timestep) // spf

    def advance(d, _i):
        for _ in range(spf):
            mujoco.mj_step(model, d)

    cam_frames, _ = render_cams(model, data, cam_names=cam_names,
                               n_frames=n_frames, advance=advance, size=size)
    if out_dir is not None:
        write_observations(out_dir, name, cam_frames, fps)
    return data.qpos[0:3].copy(), float(np.linalg.norm(data.qvel[0:3]))


def render_case(out_dir, meta, *, rig="mono", fps=FPS, size=256, duration=DURATION):
    variants = {"A": (meta["r1"], meta["X"]),
                "B": (meta["r1"], meta["Z"]),
                "C": (meta["r2"], meta["X"])}
    return {name: _render_one(out_dir, meta, e, x, name, rig, fps, size, duration)
            for name, (e, x) in variants.items()}
