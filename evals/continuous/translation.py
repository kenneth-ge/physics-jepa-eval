"""translation sweep family: camera trucks left -> right past a cube field.

Each case builds ONE static scene — several cubes of different colors, sizes,
positions and z-rotations — and renders it from N_POINTS camera positions
swept horizontally (pure translation: same orientation, height and distance,
only the camera x changes; with --rig stereo the right camera translates in
lockstep). theta = camera x. Every clip is a still of the same world from a
slightly different viewpoint, so viewpoint translation is the ONLY thing that
varies along the ladder.

The target invariant is local: for every interior sweep point, its two nearest
neighbors in embedding space should be the clips rendered right before and
right after it (`nn_cos` / `nn_l1` in evals.continuous.measure), with the
global Spearman rho(|dx|, distance) reported alongside.

Contract (--check): cameras strictly increasing and evenly spaced (adjacent
points are theta-NN by construction), cubes inside the shared view of all
sweep positions, cubes non-overlapping, hues pairwise distinct.

Usage: python -m evals.continuous.translation --out-root <dir>   (--check: no render)
Outputs <dir>/case_00 .. case_09, each with t_00..t_15.mp4 + manifest.json.
"""

import colorsys

import mujoco
import numpy as np

from ..common import cameras
from ..common.family import family_cli
from ..common.manifest import write_manifest
from ..common.observations import write_observations
from ..common.sim import render_static_rig
from ..common.xml_scene import DEFAULT_CAMERA, scene_xml, fr

EVAL_IDS = list(range(10))
SEED_BASE = 7100

N_POINTS = 16
CAM_X = 1.2                  # sweep endpoints: x in [-CAM_X, +CAM_X]
N_CUBES = (6, 10)            # cubes per case (inclusive range)
CUBE_HALF = (0.09, 0.13)
POS_BOX = dict(x=0.72, y=0.42)   # keeps cubes in view from BOTH sweep ends
MIN_SEP = 0.36               # min cube center distance (no overlap/contact)
MIN_HUE_GAP = 0.06           # min circular hue distance between any two cubes


def sweep_xs():
    return np.linspace(-CAM_X, CAM_X, N_POINTS)


def _hue_dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def case_scene(i):
    """Cubes for case i: list of dicts(pos, half, yaw_deg, rgba)."""
    rng = np.random.default_rng([SEED_BASE, i])
    n = int(rng.integers(N_CUBES[0], N_CUBES[1] + 1))
    positions = []
    for _ in range(4000):
        if len(positions) == n:
            break
        p = np.array([rng.uniform(-POS_BOX["x"], POS_BOX["x"]),
                      rng.uniform(-POS_BOX["y"], POS_BOX["y"])])
        if all(np.linalg.norm(p - q) >= MIN_SEP for q in positions):
            positions.append(p)
    if len(positions) < n:
        n = len(positions)   # dense draw; contract still checks n >= N_CUBES[0]
    hues = ((np.arange(n) + rng.uniform(0, 1)) / n
            + rng.uniform(-0.25, 0.25, n) * MIN_HUE_GAP) % 1.0
    rng.shuffle(hues)
    cubes = []
    for k in range(n):
        half = float(rng.uniform(*CUBE_HALF))
        r, g, b = colorsys.hsv_to_rgb(float(hues[k]),
                                      float(rng.uniform(0.85, 1.0)),
                                      float(rng.uniform(0.85, 1.0)))
        cubes.append(dict(pos=positions[k], half=half,
                          yaw=float(rng.uniform(0, 90)),
                          rgba=(r, g, b, 1.0)))
    return cubes


def check_contract(cubes):
    failures = []
    if len(cubes) < N_CUBES[0]:
        failures.append(f"only placed {len(cubes)} cubes (< {N_CUBES[0]})")
    for k, c in enumerate(cubes):
        if (abs(c["pos"][0]) > POS_BOX["x"] or abs(c["pos"][1]) > POS_BOX["y"]):
            failures.append(f"cube {k} out of box at {np.round(c['pos'], 3)}")
        for m in range(k + 1, len(cubes)):
            if np.linalg.norm(c["pos"] - cubes[m]["pos"]) < MIN_SEP:
                failures.append(f"cubes {k},{m} closer than MIN_SEP")
            if _hue_dist(colorsys.rgb_to_hsv(*c["rgba"][:3])[0],
                         colorsys.rgb_to_hsv(*cubes[m]["rgba"][:3])[0]) < MIN_HUE_GAP:
                failures.append(f"cubes {k},{m} hues too close")
    xs = sweep_xs()
    if not (np.all(np.diff(xs) > 0) and np.allclose(np.diff(xs), xs[1] - xs[0])):
        failures.append("sweep xs not strictly increasing / evenly spaced")
    return failures


def cube_geoms(cubes):
    xml = ""
    for k, c in enumerate(cubes):
        h = c["half"]
        xml += (f'<geom name="cube_{k}" type="box" size="{h} {h} {h}" '
                f'pos="{fr(list(c["pos"]) + [h])}" euler="0 0 {c["yaw"]:.1f}" '
                f'rgba="{fr(np.round(c["rgba"], 3))}"/>')
    return xml


def _truck_cameras_xml(rig, dx):
    """Non-primary rig cameras translated by dx along x (orientation fixed:
    the look-at target shifts with the eye, so every sweep point is a pure
    truck move for the whole rig)."""
    xml = ""
    for name in cameras.RIGS[rig]:
        if name == cameras.PRIMARY:
            continue
        c = cameras._EXTRA[name]
        pos = [c["pos"][0] + dx, c["pos"][1], c["pos"][2]]
        target = [cameras.SCENE_CENTER[0] + dx] + list(cameras.SCENE_CENTER[1:])
        xml += (f'<camera name="{name}" pos="{fr(pos)}" '
                f'xyaxes="{fr(cameras._xyaxes(pos, target))}" fovy="{c["fovy"]}"/>')
    return xml


def generate(i, out_dir, args):
    cubes = case_scene(i)
    failures = check_contract(cubes)
    xs = sweep_xs()
    print(f"case_{i:02d}: {len(cubes)} cubes, sweep x {xs[0]:+.2f}..{xs[-1]:+.2f} "
          f"step {xs[1] - xs[0]:.3f}")
    if out_dir is None:
        return failures

    cam_names = cameras.rig_cameras(args.rig)
    geoms = cube_geoms(cubes)
    clips = []
    for k, dx in enumerate(xs):
        cam = dict(pos=[float(dx), DEFAULT_CAMERA["pos"][1], DEFAULT_CAMERA["pos"][2]])
        model = mujoco.MjModel.from_xml_string(scene_xml(
            geoms=geoms, camera=cam,
            extra_cameras_xml=_truck_cameras_xml(args.rig, float(dx))))
        cam_frames = render_static_rig(
            model, mujoco.MjData(model), cam_names=cam_names,
            duration=args.duration, fps=args.fps, size=args.size)
        write_observations(out_dir, f"t_{k:02d}", cam_frames, args.fps)
        clips.append({"file": f"t_{k:02d}.mp4", "theta": float(dx)})
    write_manifest(out_dir, "camera_x", clips)
    return failures


def add_args(parser):
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)


def main():
    family_cli(name="translation", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda i: f"case_{i:02d}", description=__doc__,
               add_args=add_args)


if __name__ == "__main__":
    main()
