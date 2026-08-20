"""deform eval family: sensitivity to material stiffness.

A real (free-jointed) ball is thrown in at an angle, bounces off a cube, and
falls into a deep void under gravity. The cube SQUASHES on impact and springs
back (amplitude ~ 1/stiffness, wobble frequency ~ sqrt(stiffness)); that squash
is the STIFFNESS cue.

The ball is genuine physics (real bounce + real gravity plunge into the void) —
only the cube's deformation is scripted, because MuJoCo's elasticity plugin isn't
available in this env. The ball bounces off an invisible rigid pad at the cube's
position, and the visible cube (a non-colliding mocap geom) squashes, keyed to the
ball's ACTUAL detected impact. Because the squash never touches the ball, the ball
trajectory is identical across A/B/C — the ONLY difference is the cube's shape.

Two variants (choose via render_deform --variant):
  history  : long clip — the ball plunges fully into the void and the cube
             recovers, so the END IMAGE is identical across A/B/C; the stiffness
             signal lives only in the (differently-timed) deformation history.
  adjusted : short clip — the cube is still visibly deformed in the LAST SECOND,
             so the end image differs by stiffness (measurable by single-frame
             models too).

  A: cube stiffness k1.
  B: same k1 as A but with a still prefix (ball held before release) — matched
     pair differing only in history.
  C: cube stiffness k2 = factor*k1 (significantly different).

Sweep factor = k2/k1. Render via `evals.contrastive.render_deform`; aggregate
with `evals.contrastive.aggregate_bounce` (shared seed*/factor_* layout).
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
CUBE_S = 0.15            # cube half-size (nominal); top at 2*CUBE_S
CUBE_X = -0.30
BALL_X0 = CUBE_X - 0.42  # thrown in from up-left
BALL_Z0 = 1.02
VX0 = 1.1                # inbound horizontal speed (lands on the cube top)
HOLE_X = 0.66            # void centre = ballistic landing point (tuned)
R_HOLE = 0.30
HR = 1.6
DEEP = 2.5
NGRID = 201
Z_OFF = -DEEP            # hfield base so the flat top sits at z=0
FPS = 50
TIMESTEP = 0.0005

# lively bounce off the (rigid) cube pad — e ~ 0.63 at this K (bounce calib)
K_STIFF = 1.0e5
B_STIFF = 90.0

# squash envelope (per unit stiffness)
AMP_REF = 0.42
OMEGA_REF = 26.0

VARIANTS = {
    "history":  dict(duration=1.9, decay=6.0, still=0.5),  # ball fully plunges
    "adjusted": dict(duration=1.05, decay=2.0, still=0.5),  # squash still in window
}


def _squash_params(k):
    amp = float(np.clip(AMP_REF / k, 0.0, 0.72))
    omega = OMEGA_REF * math.sqrt(k)
    return amp, omega


def cube_size(dt, amp, omega, decay):
    """Visible cube half-sizes at dt seconds after impact (dt<0 => nominal)."""
    if dt < 0.0:
        return CUBE_S, CUBE_S, CUBE_S
    env = amp * math.exp(-decay * dt) * math.cos(omega * dt)
    return CUBE_S * (1 + 0.45 * env), CUBE_S * (1 + 0.45 * env), CUBE_S * (1 - env)


_XML = """
<mujoco model="deform">
  <option timestep="{ts}" integrator="implicitfast"/>
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
          pos="0 0 {zoff}" condim="3" friction="1 0.01 0.01"/>
    <geom name="pad" type="box" size="{cs} {cs} {cs}" pos="{cx} 0 {cs}"
          rgba="0 0 0 0" condim="3" friction="0.4 0.01 0.001"/>
    <body name="cube" mocap="true" pos="{cx} 0 {cs}">
      <geom name="viscube" type="box" size="{cs} {cs} {cs}" rgba="0.2 0.5 0.9 1"
            contype="0" conaffinity="0"/>
    </body>
    <body name="ball" pos="{bx} 0 {bz}">
      <freejoint/>
      <geom name="ball" type="sphere" size="{r}" mass="0.1" priority="1"
            rgba="0.85 0.15 0.15 1" condim="3" friction="0.4 0.005 0.0001"
            solref="-{K:.0f} -{B:.2f}" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>"""


def build_model(rig):
    xml = _XML.format(
        ts=TIMESTEP, n=NGRID, HR=HR, zmax=DEEP, zoff=Z_OFF, cs=CUBE_S, cx=CUBE_X,
        bx=BALL_X0, bz=BALL_Z0, r=BALL_R, K=K_STIFF, B=B_STIFF,
        cpos=fr(DEFAULT_CAMERA["pos"]), cxy=fr(DEFAULT_CAMERA["xyaxes"]),
        fovy=DEFAULT_CAMERA["fovy"], extra_cameras=cameras.extra_cameras_xml(rig))
    m = mujoco.MjModel.from_xml_string(xml)
    xs = np.linspace(-HR, HR, NGRID)
    Xg, Yg = np.meshgrid(xs, xs)
    rho = np.sqrt((Xg - HOLE_X) ** 2 + Yg ** 2)
    frac = np.clip(rho / R_HOLE, 0.0, 1.0)
    m.hfield_data[:] = (frac * frac * (3 - 2 * frac)).ravel()   # 1 flat, 0 void
    return m


def build_case(seed, factor, k1=None):
    # per-seed base stiffness so the seeds aren't identical; C = factor * k1.
    rng = np.random.default_rng([seed, int(round(factor * 1000))])
    base = k1 if k1 is not None else float(rng.uniform(0.7, 1.4))
    return dict(k1=base, k2=factor * base)


def _static_frame(rig, size):
    """One frame of the held release pose (ball at the throw point, cube nominal)."""
    model = build_model(rig)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_names = cameras.rig_cameras(rig)
    cam_frames, _ = render_cams(model, data, cam_names=cam_names, n_frames=1,
                               advance=lambda d, i: None, size=size)
    return {c: f[0] for c, f in cam_frames.items()}


def _sim_frames(k, p, rig, fps, size):
    """Physics rollout for stiffness k; squash keyed to the ACTUAL ball impact.
    Returns ({cam: [frames]}, final_ball_pos)."""
    model = build_model(rig)
    data = mujoco.MjData(model)
    data.qvel[0] = VX0                                   # angled inbound throw
    vis_gid = model.geom("viscube").id
    ball_gid = model.geom("ball").id
    pad_gid = model.geom("pad").id
    cube_mid = model.body("cube").mocapid[0]
    amp, omega = _squash_params(k)
    cam_names = cameras.rig_cameras(rig)
    spf = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_frames = int(round(p["duration"] * fps))
    state = {"t_impact": None}

    def hit_pad(d):
        for c in d.contact[:d.ncon]:
            g = (c.geom1, c.geom2)
            if ball_gid in g and pad_gid in g:
                return True
        return False

    def advance(d, i):
        for _ in range(spf):
            mujoco.mj_step(model, d)
            if state["t_impact"] is None and hit_pad(d):
                state["t_impact"] = d.time
        dt = -1.0 if state["t_impact"] is None else d.time - state["t_impact"]
        sx, sy, sz = cube_size(dt, amp, omega, p["decay"])
        model.geom_size[vis_gid] = (sx, sy, sz)
        d.mocap_pos[cube_mid] = (CUBE_X, 0.0, sz)        # bottom stays on ground

    cam_frames, _ = render_cams(model, data, cam_names=cam_names,
                               n_frames=n_frames, advance=advance, size=size)
    return cam_frames, data.qpos[0:3].copy()


def render_case(out_dir, meta, *, variant="history", rig="mono", fps=FPS, size=256):
    # A,B share stiffness k1 (B = A + a still prefix, so its motion is IDENTICAL,
    # not a re-sim — the edge bounce is chaotic); C has stiffness k2.
    p = VARIANTS[variant]
    n_still = int(round(p["still"] * fps))
    still = _static_frame(rig, size)
    framesA, posA = _sim_frames(meta["k1"], p, rig, fps, size)
    framesC, posC = _sim_frames(meta["k2"], p, rig, fps, size)
    framesB = {c: [still[c]] * n_still + framesA[c] for c in framesA}
    if out_dir is not None:
        write_observations(out_dir, "A", framesA, fps)
        write_observations(out_dir, "B", framesB, fps)
        write_observations(out_dir, "C", framesC, fps)
    return {"A": posA, "B": posA, "C": posC}
