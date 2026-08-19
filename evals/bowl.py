"""Bowl eval family: does the final-state embedding separate {A,B} from C?

A: ball rolls into the bowl over its RIGHT rim, settles at the bottom.
B: mirror of A — rolls in over the LEFT rim, settles at the bottom.
C by scenario:
  1: starts ON the right rim with outward velocity, rolls a short distance
     onto the flat, stopping near the bowl.
  2: starts at the bowl BOTTOM and escapes out over the right rim
     (shallower bowl + higher shared speed).
  3: crater bowl (raised rim) + high-friction apron; C starts on the rim
     crest, tips over the outer wall, stops right next to the bowl. The
     apron is required: C's exit path is A's entry path, so C recovers any
     energy A/B need to get in.
  4: scenario 3's A/B, but C simply RESTS at the bowl center for the whole
     clip — all three scenes end identically; only history differs.

Same ball, same bowl, same |initial speed| within a scenario; layout,
camera and lighting are mirror-symmetric about x=0. The ground is a
heightfield (parabolic depression, optionally with a raised rim) because
MuJoCo convexifies meshes — a concave bowl mesh would not collide.

--check runs physics only (CPU, no rendering) and validates the contract:
every scene ends with the ball at rest >1 s; A and B end at the same
position. Exits nonzero on violation.

Usage: python -m evals.bowl --out-root <dir>   (--only 3, --check, ...)
"""

import types

import mujoco
import numpy as np

from .common import cameras
from .common.family import family_cli
from .common.observations import write_observations
from .common.sim import BallTracker, rolling_spin, rollout, render_cams
from .common.video import save_video

# Fixed geometry/material constants shared by every scenario.
CONSTANTS = dict(bowl_r=0.4, ball_r=0.08, mu_slide=1.0, mu_spin=0.01,
                 flat_h=0.12, hx=1.7)

XML = """
<mujoco model="bowl">
  <option timestep="0.002"/>
  <visual>
    <global offwidth="512" offheight="512"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{hx} {hy} {zmax} 0.05"/>
    <material name="ground" rgba="0.72 0.72 0.76 1" specular="0.2" shininess="0.3"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <camera name="fixed" pos="0 -2.8 1.1" xyaxes="1 0 0 0 0.366 0.93" fovy="52"/>
    {extra_cameras}
    <geom name="terrain" type="hfield" hfield="terrain" material="ground"
          condim="6" friction="{mu_slide} {mu_spin} {mu_roll}"/>
    {extra_geoms}
    <body name="ball" pos="0 0 1">
      <joint name="ball" type="free" damping="{damping}"/>
      <geom name="ball" type="sphere" size="{ball_r}" mass="0.1"
            rgba="0.85 0.15 0.15 1" condim="6"
            friction="{mu_slide} {mu_spin} {mu_roll}"/>
    </body>
  </worldbody>
</mujoco>
"""

# Tuned per-scenario parameters — see the guide notes in the README before
# touching these; friction/speed/geometry are tightly coupled.
SCENARIOS = {
    1: dict(speed=0.4, start_offset=0.65, bowl_d=0.10, c_start="rim",
            c_max_x=0.95, rim_h=0.0, rim_w=0.0, mu_apron=None,
            mu_roll=0.001, damping=0.0005),
    2: dict(speed=1.1, start_offset=1.4, bowl_d=0.03, c_start="bottom",
            c_max_x=None, rim_h=0.0, rim_w=0.0, mu_apron=None,
            mu_roll=0.001, damping=0.0005),
    3: dict(speed=0.9, start_offset=0.55, bowl_d=0.08, c_start="rim",
            c_max_x=0.75, rim_h=0.02, rim_w=0.06, mu_apron=0.02,
            # less-slick interior: halves settling oscillation while keeping
            # A/B centering ~6 mm; above 0.002 balls stall on the outer wall.
            mu_roll=0.002, damping=0.001),
    4: dict(speed=0.9, start_offset=0.55, bowl_d=0.08, c_start="center",
            c_max_x=None, rim_h=0.02, rim_w=0.06, mu_apron=0.02,
            mu_roll=0.002, damping=0.001),
}


def build_model(p, extra_cameras=""):
    nrow, ncol = 61, 341
    zmax = p.flat_h + p.rim_h
    extra = ""
    if p.mu_apron is not None:
        # High-friction apron: two boxes (symmetric in x) whose tops sit
        # flush with (0.5 mm above) the flat ground.
        x0 = p.bowl_r + p.rim_w + 0.01
        half_len = (p.hx - x0) / 2
        cx = (p.hx + x0) / 2
        half_h = (p.flat_h + 0.0005) / 2
        for tag, sign in (("R", 1), ("L", -1)):
            extra += (f'<geom name="apron{tag}" type="box" '
                      f'size="{half_len:.4f} 0.6 {half_h:.4f}" '
                      f'pos="{sign * cx:.4f} 0 {half_h:.4f}" material="ground" '
                      f'condim="6" friction="{p.mu_slide} {p.mu_spin} {p.mu_apron}"/>')
    model = mujoco.MjModel.from_xml_string(XML.format(
        nrow=nrow, ncol=ncol, hx=p.hx, hy=0.6, zmax=zmax,
        mu_slide=p.mu_slide, mu_spin=p.mu_spin, mu_roll=p.mu_roll,
        damping=p.damping, ball_r=p.ball_r, extra_geoms=extra,
        extra_cameras=extra_cameras))
    # Flat ground at flat_h with a parabolic bowl (depth d, radius R) at
    # x=0 — symmetric in x by construction. Parabolic, not cosine: nonzero
    # slope at the rim gives a sharp edge instead of a wide flattening lip.
    # With rim_h > 0 a crater lip rises at rho=R, then a straight outer
    # wall drops back to ground level over rim_w.
    xs = np.linspace(-p.hx, p.hx, ncol)
    ys = np.linspace(-0.6, 0.6, nrow)
    X, Y = np.meshgrid(xs, ys)
    rho = np.sqrt(X**2 + Y**2)
    z = np.zeros_like(rho)
    inside = rho <= p.bowl_r
    z[inside] = -p.bowl_d + (p.bowl_d + p.rim_h) * (rho[inside] / p.bowl_r) ** 2
    if p.rim_w > 0:
        wall = (rho > p.bowl_r) & (rho <= p.bowl_r + p.rim_w)
        z[wall] = p.rim_h * (1.0 - (rho[wall] - p.bowl_r) / p.rim_w)
    model.hfield_data[:] = ((p.flat_h + z) / zmax).ravel()
    return model


def scene_initial_conditions(p):
    z_ab = p.flat_h + p.ball_r + (0.001 if p.mu_apron is not None else 0.0)
    z_rim = p.flat_h + p.rim_h + p.ball_r
    z_bowl = p.flat_h - p.bowl_d + p.ball_r
    c_ic = {
        "rim": dict(x0=+p.bowl_r, vx0=+p.speed, z0=z_rim),
        "bottom": dict(x0=0.0, vx0=+p.speed, z0=z_bowl),
        "center": dict(x0=0.0, vx0=0.0, z0=z_bowl),
    }[p.c_start]
    return {
        "A": dict(x0=+p.start_offset, vx0=-p.speed, z0=z_ab),
        "B": dict(x0=-p.start_offset, vx0=+p.speed, z0=z_ab),
        "C": c_ic,
    }


def simulate(model, p, x0, vx0, z0, out_dir=None, name=None,
             cam_names=(cameras.PRIMARY,)):
    data = mujoco.MjData(model)
    data.qpos[0:3] = (x0, 0.0, z0)
    data.qvel[0:3] = (vx0, 0.0, 0.0)
    data.qvel[3:6] = rolling_spin((vx0, 0.0, 0.0), p.ball_r)
    tracker = BallTracker(model)
    render = out_dir is not None

    if not render or tuple(cam_names) == (cameras.PRIMARY,):
        # Mono path: byte-identical to the original single-camera render.
        frames = rollout(model, data, duration=p.duration, fps=p.fps,
                         size=p.size, post_step=tracker.post_step, render=render)
        if render:
            save_video(frames, out_dir / f"{name}.mp4", p.fps)
        return tracker.summary(p.duration)

    # Multi-camera rig: same physics, rendered from every camera per frame.
    spf = max(1, round(1.0 / (p.fps * model.opt.timestep)))
    n_frames = int(p.duration / model.opt.timestep) // spf

    def advance(d, _i):
        for _ in range(spf):
            mujoco.mj_step(model, d)
            tracker.post_step(d)

    cam_frames, _ = render_cams(model, data, cam_names=list(cam_names),
                               n_frames=n_frames, advance=advance, size=p.size)
    write_observations(out_dir, name, cam_frames, p.fps)
    return tracker.summary(p.duration)


def check_contract(p, results):
    rim = p.bowl_r + p.rim_w + p.ball_r
    failures = []
    for name in "AB":
        r = results[name]
        if abs(r["final_pos"][0]) > 0.02 or abs(r["final_pos"][1]) > 0.02:
            failures.append(f"{name}: did not settle at bowl bottom (x={r['final_pos'][0]:+.4f})")
        if r["stopped_dur"] < 1.05:
            failures.append(f"{name}: stopped only {r['stopped_dur']:.2f}s (< 1.05s)")
    if results["A"]["min_x"] < -rim:
        failures.append(f"A: escaped out the far rim (min_x={results['A']['min_x']:+.3f})")
    if results["B"]["max_x"] > rim:
        failures.append(f"B: escaped out the far rim (max_x={results['B']['max_x']:+.3f})")
    delta_ab = float(np.linalg.norm(results["A"]["final_pos"] - results["B"]["final_pos"]))
    if delta_ab > 0.02:
        failures.append(f"A/B final positions differ by {delta_ab:.4f}m (> 0.02)")

    rc = results["C"]
    if p.c_start == "center":
        if abs(rc["max_x"]) > 0.03 or abs(rc["min_x"]) > 0.03:
            failures.append(f"C: moved from the center (x-range [{rc['min_x']:+.3f}, {rc['max_x']:+.3f}])")
        if rc["stopped_dur"] < p.duration - 0.5:
            failures.append(f"C: not at rest for the whole clip (stopped {rc['stopped_dur']:.2f}s)")
    else:
        if rc["final_pos"][0] < rim:
            failures.append(f"C: did not end outside the bowl (final x={rc['final_pos'][0]:+.4f} < {rim:.2f})")
        if p.c_start == "rim" and rc["min_x"] < p.bowl_r - 0.05:
            failures.append(f"C: fell back into the bowl (min_x={rc['min_x']:+.3f})")
        if p.c_max_x is not None and rc["final_pos"][0] > p.c_max_x:
            failures.append(f"C: stopped too far from the bowl (x={rc['final_pos'][0]:+.4f} > {p.c_max_x})")
        if rc["final_pos"][0] > p.hx - p.ball_r:
            failures.append(f"C: rolled off the terrain (final x={rc['final_pos'][0]:+.4f})")
        if rc["stopped_dur"] < 1.05:
            failures.append(f"C: stopped only {rc['stopped_dur']:.2f}s (< 1.05s)")
    return delta_ab, failures


def scenario_params(scenario, args):
    """Merge fixed constants + the scenario's tuned dict + runtime args into
    one namespace consumed by build_model/simulate/check_contract."""
    p = types.SimpleNamespace(**CONSTANTS, **SCENARIOS[scenario],
                              duration=args.duration, fps=args.fps, size=args.size)
    # flat_h must exceed bowl_d (hfield heights cannot go below 0)
    assert p.flat_h > p.bowl_d, f"scenario {scenario}: flat_h <= bowl_d"
    return p


def generate(scenario, out_dir, args):
    p = scenario_params(scenario, args)
    rig = getattr(args, "rig", "mono")
    model = build_model(p, extra_cameras=cameras.extra_cameras_xml(rig))
    cam_names = tuple(cameras.rig_cameras(rig))
    results = {}
    for name, ic in scene_initial_conditions(p).items():
        results[name] = simulate(model, p, out_dir=out_dir, name=name,
                                 cam_names=cam_names, **ic)
        r = results[name]
        print(f"{name}: final x={r['final_pos'][0]:+.4f} y={r['final_pos'][1]:+.4f} "
              f"z={r['final_pos'][2]:.4f} speed={r['final_speed']:.4f} "
              f"stopped={r['stopped_dur']:.2f}s x-range=[{r['min_x']:+.3f}, {r['max_x']:+.3f}]")
    delta_ab, failures = check_contract(p, results)
    print(f"A/B final position delta: {delta_ab*1000:.1f} mm")
    return failures


def add_args(parser):
    parser.add_argument("--duration", type=float, default=9.5)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)


def main():
    family_cli(name="bowl", eval_ids=sorted(SCENARIOS), generate=generate,
               subdir=lambda s: f"scenario_{s}", description=__doc__, add_args=add_args)


if __name__ == "__main__":
    main()
