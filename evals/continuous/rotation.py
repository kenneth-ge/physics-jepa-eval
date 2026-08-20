"""rotation sweep family: camera orbits 180 degrees around a cube field.

Each case builds ONE static scene — several cubes of different colors, sizes,
positions and z-rotations — and renders it from N_POINTS camera positions on a
semicircle around the scene centre. The camera keeps a fixed radius, height
and look-at target, so the only thing that changes along the ladder is the
viewing azimuth: theta = azimuth in degrees, sweeping -90 (viewing from the
+x side) through 0 (the familiar front view) to +90 (the -x side). With
--rig stereo the second camera orbits alongside it.

This is the angular counterpart of `translation`: there the camera slides and
the scene shifts across the frame; here the camera circles and the scene turns,
so cubes change their mutual occlusion and show different faces while staying
centred. Grading is identical — for every interior sweep point, its two
nearest neighbors in embedding space should be the clips rendered right before
and right after it (`nn_cos` / `nn_l1` in evals.continuous.measure).

Contract (--check): azimuths strictly increasing and evenly spaced, cubes
non-overlapping with pairwise-distinct hues, and every cube fully inside the
frame at EVERY azimuth (checked numerically via `_view_margin`).

Usage: python -m evals.continuous.rotation --out-root <dir>   (--check: no render)
Outputs <dir>/case_00 .. case_09, each with r_00..r_15.mp4, manifest.json and
grid.png (a 2x8 contact sheet of the orbit; --no-grid skips it).
"""

import mujoco
import numpy as np

from ..common import cameras
from ..common.family import family_cli
from ..common.manifest import write_manifest
from ..common.observations import write_observations
from ..common.sim import render_static_rig
from ..common.xml_scene import scene_xml, fr
from .cubes import CUBE_HALF, check_field, cube_geoms, sample_field
from .grid import COLS as GRID_COLS, save_grid

EVAL_IDS = list(range(10))
SEED_BASE = 7300

N_POINTS = 16
AZ_HALF = 90.0               # sweep azimuth in [-AZ_HALF, +AZ_HALF] degrees
ORBIT_R = 2.8                # camera radius (matches the shared front camera)
ORBIT_Z = 1.1                # camera height
LOOK_AT = (0.0, 0.0, 0.15)   # orbit centre / look-at target
FOVY_DEG = 52.0
N_CUBES = (6, 10)
R_FIELD = 0.70               # cubes live in a disc of this radius
MIN_SEP = 0.32
VIEW_MARGIN = 0.04           # world-units of slack demanded at the frame edge

# The shared 4x4 floor ends inside the frame, and when the camera orbits its
# edge sweeps across the image as a rotating diagonal — a strong azimuth cue
# that has nothing to do with the cubes. MuJoCo renders a plane with a zero
# size component as infinite, which puts a clean horizon there instead.
INFINITE_FLOOR = ('<geom name="floor" type="plane" size="0 0 0.1" '
                  'material="ground" condim="6" friction="1.0 0.01 0.02"/>')


def sweep_azimuths():
    return np.linspace(-AZ_HALF, AZ_HALF, N_POINTS)


def camera_pose(az_deg):
    """Eye position for an azimuth; az=0 reproduces the shared front view."""
    a = np.deg2rad(az_deg)
    return [ORBIT_R * np.sin(a), -ORBIT_R * np.cos(a), ORBIT_Z]


def case_scene(i):
    rng = np.random.default_rng([SEED_BASE, i])
    n = int(rng.integers(N_CUBES[0], N_CUBES[1] + 1))
    return sample_field(rng, n=n, half_range=(R_FIELD, R_FIELD),
                        min_sep=MIN_SEP,
                        in_region=lambda p: np.linalg.norm(p) <= R_FIELD)


def _basis(az_deg):
    pos = np.array(camera_pose(az_deg))
    ax = cameras._xyaxes(pos, LOOK_AT)
    right, up = np.array(ax[:3]), np.array(ax[3:])
    return pos, right, up, np.cross(right, up) * -1.0


def _view_margin(cube, az_deg):
    """Signed slack (world units) between the cube's bounding sphere and the
    nearest frame edge at this azimuth. Negative = clipped."""
    pos, right, up, fwd = _basis(az_deg)
    centre = np.array([cube["pos"][0], cube["pos"][1], cube["half"]])
    radius = cube["half"] * np.sqrt(3.0)      # cube corner from its centre
    v = centre - pos
    depth = float(v @ fwd)
    limit = depth * np.tan(np.deg2rad(FOVY_DEG / 2))
    return min(limit - abs(float(v @ right)) - radius,
               limit - abs(float(v @ up)) - radius)


def check_contract(cubes):
    failures = check_field(cubes, n_min=N_CUBES[0], min_sep=MIN_SEP)
    az = sweep_azimuths()
    if not (np.all(np.diff(az) > 0) and np.allclose(np.diff(az), az[1] - az[0])):
        failures.append("azimuths not strictly increasing / evenly spaced")
    for k, c in enumerate(cubes):
        worst = min(_view_margin(c, a) for a in az)
        if worst < VIEW_MARGIN:
            failures.append(
                f"cube {k} at {np.round(c['pos'], 2)} is {worst:+.3f} from the "
                f"frame edge at some azimuth (need >={VIEW_MARGIN})")
    return failures


def _orbit_cameras_xml(rig, az_deg):
    """Non-primary rig cameras, rotated to the same azimuth so the whole rig
    orbits together (the stereo partner keeps its lateral offset)."""
    xml = ""
    for name in cameras.RIGS[rig]:
        if name == cameras.PRIMARY:
            continue
        c = cameras._EXTRA[name]
        base = np.array(c["pos"], float)
        a = np.deg2rad(az_deg)
        # Same CCW rotation about z that carries (0,-R) to camera_pose(az).
        rot = np.array([[np.cos(a), -np.sin(a)],
                        [np.sin(a), np.cos(a)]])
        xy = rot @ base[:2]
        pos = [float(xy[0]), float(xy[1]), float(base[2])]
        xml += (f'<camera name="{name}" pos="{fr(pos)}" '
                f'xyaxes="{fr(cameras._xyaxes(pos, LOOK_AT))}" '
                f'fovy="{c["fovy"]}"/>')
    return xml


def generate(i, out_dir, args):
    cubes = case_scene(i)
    failures = check_contract(cubes)
    az = sweep_azimuths()
    print(f"case_{i:02d}: {len(cubes)} cubes, azimuth {az[0]:+.0f}..{az[-1]:+.0f} "
          f"step {az[1] - az[0]:.1f} deg")
    if out_dir is None:
        return failures

    cam_names = cameras.rig_cameras(args.rig)
    geoms = cube_geoms(cubes)
    clips, stills, labels = [], [], []
    for k, a in enumerate(az):
        pos = camera_pose(float(a))
        cam = dict(pos=pos, xyaxes=cameras._xyaxes(pos, LOOK_AT),
                   fovy=FOVY_DEG)
        model = mujoco.MjModel.from_xml_string(scene_xml(
            geoms=geoms, camera=cam, floor=INFINITE_FLOOR,
            extra_cameras_xml=_orbit_cameras_xml(args.rig, float(a))))
        cam_frames = render_static_rig(
            model, mujoco.MjData(model), cam_names=cam_names,
            duration=args.duration, fps=args.fps, size=args.size)
        write_observations(out_dir, f"r_{k:02d}", cam_frames, args.fps)
        clips.append({"file": f"r_{k:02d}.mp4", "theta": float(a)})
        stills.append(cam_frames[cameras.PRIMARY][0])
        labels.append(f"{k:02d}  az={a:+.0f}")
    write_manifest(out_dir, "camera_azimuth_deg", clips)
    if not args.no_grid:
        rows = save_grid(stills, labels, out_dir / "grid.png")
        print(f"  wrote grid.png ({rows}x{GRID_COLS})")
    return failures


def add_args(parser):
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--no-grid", action="store_true",
                        help="skip the per-case grid.png contact sheet")


def main():
    family_cli(name="rotation", eval_ids=EVAL_IDS, generate=generate,
               subdir=lambda i: f"case_{i:02d}", description=__doc__,
               add_args=add_args)


if __name__ == "__main__":
    main()
