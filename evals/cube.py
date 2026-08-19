"""Cube eval family: a kinematic cube (mocap, collision-free — MuJoCo only
renders) travels along a path and holds its final pose. The last second is
the measured window, so the invariant depends only on where the cube ENDS:

  A and B end at the same point Y (identical last second, different paths);
  C ends at a different point (different last second).

Path types (5 evals each unless noted):
  bezier_arc  : eval 0 is the original semicircle pair (A ccw / B cw, same
                X and Y; C = A translated); evals 1–4 are random beziers.
  line        : straight lines (A and B reach Y from different directions).
  polyline    : random straight-line sequences (3–5 legs).
  bezier_chain: chaotic chains of 2–3 cubic bezier segments.
  orbit       : circular arcs in a random plane.
  spiral      : orbits with growing radius and axial drift.

Every path is pinned to its endpoint by translation, so A/B share Y exactly.
A camera-projection check guarantees the whole cube stays in frame for every
frame (verified in --check, no rendering needed) — paths that would leave
the frame are resampled with a shrinking extent.

Usage: python -m evals.cube --out-root <dir>   (--only line_2, --check, ...)
"""

import math

import mujoco
import numpy as np

from .common import cameras
from .common.family import family_cli
from .common.observations import write_observations
from .common.sim import render_cams
from .common.xml_scene import scene_xml, DEFAULT_CAMERA

# Endpoint sampling box (inner) and per-path local extent (outer margin).
END_BOX = dict(x=0.55, y=0.30, z_lo=0.45, z_hi=1.05)
SPREAD0 = 0.55          # initial path extent around its endpoint
MIN_C_OFFSET = 0.5      # C's endpoint must be this far from Y
N_VIS_SAMPLES = 48      # trajectory samples for the visibility check
FLOOR_CLEAR = 0.03

EVAL_SPECS = [("bezier_arc_0", "arc", 0)]
EVAL_SPECS += [(f"bezier_{i}", "bezier", i) for i in range(1, 5)]
for _kind in ("line", "polyline", "bezier_chain", "orbit", "spiral"):
    EVAL_SPECS += [(f"{_kind}_{i}", _kind, i) for i in range(5)]
EVAL_IDS = [e[0] for e in EVAL_SPECS]
SPEC = {e[0]: (e[1], e[2]) for e in EVAL_SPECS}


# ---------------------------------------------------------------- camera --

def _camera_basis():
    xy = DEFAULT_CAMERA["xyaxes"]
    r = np.array(xy[:3], float); r /= np.linalg.norm(r)
    u = np.array(xy[3:], float); u /= np.linalg.norm(u)
    f = -np.cross(r, u); f /= np.linalg.norm(f)   # camera looks along -z_cam
    return np.array(DEFAULT_CAMERA["pos"], float), r, u, f


_EYE, _R, _U, _F = _camera_basis()
_TAN = math.tan(math.radians(DEFAULT_CAMERA["fovy"]) / 2)  # aspect 1 -> fovx=fovy
_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                     for sz in (-1, 1)], float)


def in_frame(points, half, margin=0.02):
    """True iff all 8 cube corners at every point project inside the frame
    (with margin) and stay above the floor."""
    for p in points:
        if p[2] < half + FLOOR_CLEAR:
            return False
        for c in _CORNERS:
            rel = p + c * half - _EYE
            d = _F @ rel
            if d <= 0.05:
                return False
            if abs((_R @ rel) / (d * _TAN)) > 1 - margin:
                return False
            if abs((_U @ rel) / (d * _TAN)) > 1 - margin:
                return False
    return True


# ---------------------------------------------------- path shape builders --
# Each returns shape(u), u in [0,1], with shape(1) == origin (endpoint), so
# adding the desired endpoint pins the path there.

def _rand(rng, spread):
    return rng.uniform(-1, 1, 3) * np.array([spread, spread * 0.6, spread * 0.7])


def _rand_dir(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _bezier(pts, u):
    pts = [np.asarray(p, float) for p in pts]
    while len(pts) > 1:
        pts = [(1 - u) * pts[i] + u * pts[i + 1] for i in range(len(pts) - 1)]
    return pts[0]


def _polyline(pts, u):
    n = len(pts) - 1
    seg = min(int(u * n), n - 1)
    local = u * n - seg
    return (1 - local) * np.asarray(pts[seg], float) + local * np.asarray(pts[seg + 1], float)


def g_line(rng, spread):
    start = _rand(rng, spread)
    return lambda u: (1 - u) * start


def g_bezier(rng, spread):
    pts = [_rand(rng, spread) for _ in range(3)] + [np.zeros(3)]
    return lambda u: _bezier(pts, u)


def g_polyline(rng, spread):
    k = int(rng.integers(3, 6))
    pts = [_rand(rng, spread) for _ in range(k)] + [np.zeros(3)]
    return lambda u: _polyline(pts, u)


def g_bezier_chain(rng, spread):
    k = int(rng.integers(2, 4))
    knots = [_rand(rng, spread) for _ in range(k)] + [np.zeros(3)]
    segs = [(knots[i], _rand(rng, spread), _rand(rng, spread), knots[i + 1])
            for i in range(k)]
    def shape(u):
        seg = min(int(u * k), k - 1)
        return _bezier(segs[seg], u * k - seg)
    return shape


def g_orbit(rng, spread):
    e1, e2 = _rand_dir(rng), _rand_dir(rng)
    e2 = e2 - e1 * (e1 @ e2); e2 /= np.linalg.norm(e2)
    radius = spread * rng.uniform(0.5, 0.9)
    th0 = rng.uniform(0, 2 * math.pi)
    th1 = th0 + rng.choice([-1, 1]) * rng.uniform(0.8, 1.6) * math.pi
    center = -radius * (math.cos(th1) * e1 + math.sin(th1) * e2)
    def shape(u):
        th = th0 + (th1 - th0) * u
        return center + radius * (math.cos(th) * e1 + math.sin(th) * e2)
    return shape


def g_spiral(rng, spread):
    e1, e2 = _rand_dir(rng), _rand_dir(rng)
    e2 = e2 - e1 * (e1 @ e2); e2 /= np.linalg.norm(e2)
    axis = np.cross(e1, e2)
    r0, r1 = spread * 0.25, spread * rng.uniform(0.6, 0.9)
    th0 = rng.uniform(0, 2 * math.pi)
    turns = rng.choice([-1, 1]) * rng.uniform(1.0, 2.0) * math.pi
    drift = spread * rng.uniform(-0.4, 0.4)
    def raw(u):
        th = th0 + turns * u
        r = r0 + (r1 - r0) * u
        return r * (math.cos(th) * e1 + math.sin(th) * e2) + drift * u * axis
    end = raw(1.0)
    return lambda u: raw(u) - end   # pin shape(1) to origin


GENERATORS = {"line": g_line, "bezier": g_bezier, "polyline": g_polyline,
              "bezier_chain": g_bezier_chain, "orbit": g_orbit, "spiral": g_spiral}


# ------------------------------------------------------------ assembly ----

def _timed(shape, end, move_time, hold_time):
    def fn(t):
        s = min(t / move_time, 1.0)
        u = 3 * s ** 2 - 2 * s ** 3     # smoothstep ease
        return np.asarray(shape(u)) + np.asarray(end, float)
    return fn


def sample_endpoint(rng):
    return np.array([rng.uniform(-END_BOX["x"], END_BOX["x"]),
                     rng.uniform(-END_BOX["y"], END_BOX["y"]),
                     rng.uniform(END_BOX["z_lo"], END_BOX["z_hi"])])


def build_visible(kind, rng, end, args):
    """A path of `kind` ending at `end`, guaranteed in-frame; resample with a
    shrinking extent until the visibility check passes."""
    ts = np.linspace(0, args.move_time, N_VIS_SAMPLES)
    for k in range(400):
        shape = GENERATORS[kind](rng, SPREAD0 * (0.93 ** k))
        fn = _timed(shape, end, args.move_time, args.hold_time)
        # generate with extra buffer so paths never crowd the frame edge
        if in_frame([fn(t) for t in ts], args.cube_half, margin=0.05):
            return fn
    return _timed(lambda u: np.zeros(3), end, args.move_time, args.hold_time)


def arc_trajectory(name, t, args):
    """The original semicircle pair: A ccw / B cw share X and Y; C = A shifted."""
    cx = args.offset_x if name == "C" else 0.0
    direction = -1 if name == "B" else +1
    s = np.clip(t / args.move_time, 0.0, 1.0)
    u = 3 * s ** 2 - 2 * s ** 3
    theta = np.pi + direction * np.pi * u
    return np.array([cx + args.radius * np.cos(theta), 0.0,
                     args.zc + args.radius * np.sin(theta)])


def build_eval(kind, seed, args):
    if kind == "arc":
        return {n: (lambda t, n=n: arc_trajectory(n, t, args)) for n in "ABC"}
    Y = sample_endpoint(np.random.default_rng([seed, 10]))
    rng_c = np.random.default_rng([seed, 11])
    Yc = sample_endpoint(rng_c)
    while np.linalg.norm(Yc - Y) < MIN_C_OFFSET:
        Yc = sample_endpoint(rng_c)
    return {
        "A": build_visible(kind, np.random.default_rng([seed, 0]), Y, args),
        "B": build_visible(kind, np.random.default_rng([seed, 1]), Y, args),
        "C": build_visible(kind, np.random.default_rng([seed, 2]), Yc, args),
    }


def generate(eval_id, out_dir, args):
    kind, seed = SPEC[eval_id]
    trajs = build_eval(kind, seed, args)
    duration = args.move_time + args.hold_time
    times = np.arange(int(round(duration * args.fps))) / args.fps

    ends, failures = {}, []
    for name, fn in trajs.items():
        pts = [fn(t) for t in np.linspace(0, duration, N_VIS_SAMPLES)]
        ends[name] = fn(duration)
        if not in_frame(pts, args.cube_half):
            failures.append(f"{name}: leaves the frame")
        print(f"{name}: end {np.round(ends[name], 3)}")
    if np.linalg.norm(ends["A"] - ends["B"]) > 1e-6:
        failures.append("A and B must share the end state")
    if np.linalg.norm(ends["A"] - ends["C"]) < 0.1:
        failures.append("C must end elsewhere")
    if out_dir is None:
        return failures

    cam_names = cameras.rig_cameras(args.rig)
    model = mujoco.MjModel.from_xml_string(scene_xml(
        bodies=cube_body(args.cube_half),
        extra_cameras_xml=cameras.extra_cameras_xml(args.rig)))
    for name, fn in trajs.items():
        data = mujoco.MjData(model)

        def advance(data, i, fn=fn):
            data.mocap_pos[0] = fn(times[i])
            mujoco.mj_forward(model, data)

        cam_frames, states = render_cams(
            model, data, cam_names=cam_names, n_frames=len(times),
            advance=advance, size=args.size,
            state_fn=lambda d: d.mocap_pos[0].copy())
        write_observations(out_dir, name, cam_frames, args.fps,
                           state=states, times=times, state_key="cube_pos")
    return failures


def cube_body(half):
    return (f'<body name="cube" mocap="true" pos="0 0 0.7">'
            f'<geom type="box" size="{half} {half} {half}" rgba="0.2 0.4 0.9 1" '
            f'contype="0" conaffinity="0"/></body>')


def add_args(parser):
    parser.add_argument("--radius", type=float, default=0.5, help="arc eval radius")
    parser.add_argument("--zc", type=float, default=0.7, help="arc eval X/Y height")
    parser.add_argument("--offset-x", type=float, default=0.4, help="arc eval C shift")
    parser.add_argument("--move-time", type=float, default=3.2)
    parser.add_argument("--hold-time", type=float, default=1.3)
    parser.add_argument("--cube-half", type=float, default=0.09)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)


def main():
    family_cli(name="cube", eval_ids=EVAL_IDS, generate=generate,
               subdir=str, description=__doc__, add_args=add_args)


if __name__ == "__main__":
    main()
