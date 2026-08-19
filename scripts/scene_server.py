"""Interactive A/B/C scene designer for the V-JEPA 2 evals.

Serves a web UI (single page) where each of the three scenarios A/B/C is a
JSON scene spec. Objects:

- bowl: a TRUE bowl sitting on the ground (walls above ground, wide flat
  lip). MuJoCo convexifies meshes, so the bowl is assembled from convex box
  panels: a vertical outer wall ring, a flat lip ring, parabolic interior
  panels, and a center disc. Arbitrary radius/height/lip/depth/friction.
- ramp: an inclined box (angle, length, width, yaw, friction).
- ball: dynamic sphere (free joint) with initial position/velocity;
  initial spin is set for rolling without slipping.
- cube: kinematic mocap box (no physics interaction) following a
  user-specified curve of {t, pos} waypoints, smoothstep-interpolated,
  holding the last waypoint to the end of the clip.

Endpoints: GET / (UI), POST /render, POST /eval, POST /save, GET /load/...
Run on the GPU box:  MUJOCO_GL=egl python scripts/scene_server.py
Reach it from your machine via VS Code Remote-SSH port forwarding or
`ssh -L 8020:localhost:8020 kenny-dev`, then open http://localhost:8020
"""

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path(os.environ.get("SCENE_WORKDIR", "/data/scenes"))
CURRENT = WORK / "current"

sys.path.insert(0, str(REPO))
from evals.common.objects import bowl_geoms, ramp_geoms          # noqa: E402
from evals.common.xml_scene import DEFAULT_CAMERA, DEFAULT_FRICTION, fr  # noqa: E402


def build_scene(spec):
    duration = float(spec.get("duration", 7.0))
    fps = int(spec.get("fps", 25))
    size = int(spec.get("size", 256))
    cam = {**DEFAULT_CAMERA, **spec.get("camera", {})}
    ground_fric = spec.get("ground", {}).get("friction", [1.0, 0.01, 0.02])

    geoms, balls, cubes = [], [], []
    for i, o in enumerate(spec.get("objects", [])):
        t = o.get("type")
        if t == "bowl":
            geoms += bowl_geoms(o, i)
        elif t == "ramp":
            geoms += ramp_geoms(o, i)
        elif t == "ball":
            balls.append(o)
        elif t == "cube":
            cubes.append(o)
        else:
            raise ValueError(f"unknown object type: {t!r}")

    bodies = []
    for i, b in enumerate(balls):
        r = float(b.get("radius", 0.08))
        rgba = b.get("rgba", [0.85, 0.15, 0.15, 1])
        fric = b.get("friction", DEFAULT_FRICTION)
        pos = b.get("pos", [0, 0, r])
        bodies.append(
            f'<body name="ball{i}" pos="{fr(pos)}">'
            f'<joint name="ball{i}" type="free" damping="{b.get("damping", 0.001)}"/>'
            f'<geom name="ball{i}" type="sphere" size="{r}" mass="{b.get("mass", 0.1)}" '
            f'rgba="{fr(rgba)}" condim="6" friction="{fr(fric)}"/></body>')
    for i, c in enumerate(cubes):
        h = float(c.get("half", 0.09))
        rgba = c.get("rgba", [0.2, 0.4, 0.9, 1])
        start = (c.get("curve") or [{"pos": [0, 0, 0.5]}])[0]["pos"]
        bodies.append(
            f'<body name="cube{i}" mocap="true" pos="{fr(start)}">'
            f'<geom type="box" size="{h} {h} {h}" rgba="{fr(rgba)}" '
            f'contype="0" conaffinity="0"/></body>')

    xml = f"""
<mujoco model="designer">
  <option timestep="0.002"/>
  <visual>
    <global offwidth="1024" offheight="1024"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <material name="ground" rgba="0.72 0.72 0.76 1" specular="0.2" shininess="0.3"/>
    <material name="obj" specular="0.3" shininess="0.4"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <camera name="fixed" pos="{fr(cam['pos'])}" xyaxes="{fr(cam['xyaxes'])}" fovy="{cam['fovy']}"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="ground"
          condim="6" friction="{fr(ground_fric)}"/>
    {''.join(geoms)}
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    return model, balls, cubes, duration, fps, size


def cube_pos_at(curve, t):
    wps = sorted(curve, key=lambda w: w.get("t", 0))
    if t <= wps[0].get("t", 0):
        return np.array(wps[0]["pos"], dtype=float)
    for a, b in zip(wps, wps[1:]):
        ta, tb = a.get("t", 0), b.get("t", 0)
        if t <= tb:
            s = (t - ta) / max(tb - ta, 1e-9)
            u = 3 * s**2 - 2 * s**3
            return (1 - u) * np.array(a["pos"], float) + u * np.array(b["pos"], float)
    return np.array(wps[-1]["pos"], dtype=float)


def simulate(spec, video_path=None):
    model, balls, cubes, duration, fps, size = build_scene(spec)
    data = mujoco.MjData(model)
    for i, b in enumerate(balls):
        adr = model.joint(f"ball{i}").qposadr[0]
        vadr = model.joint(f"ball{i}").dofadr[0]
        v = np.array(b.get("vel", [0, 0, 0]), dtype=float)
        r = float(b.get("radius", 0.08))
        data.qpos[adr:adr + 3] = b.get("pos", [0, 0, r])
        data.qvel[vadr:vadr + 3] = v
        data.qvel[vadr + 3:vadr + 6] = [-v[1] / r, v[0] / r, 0]  # rolling spin

    renderer = None
    frames = []
    if video_path is not None:
        renderer = mujoco.Renderer(model, height=size, width=size)
    steps_per_frame = max(1, round(1.0 / (fps * model.opt.timestep)))
    n_steps = int(duration / model.opt.timestep)

    stats = {f"ball{i}": {"last_move_t": 0.0} for i in range(len(balls))}
    for step in range(n_steps):
        for i, c in enumerate(cubes):
            data.mocap_pos[i] = cube_pos_at(c.get("curve", []), data.time)
        mujoco.mj_step(model, data)
        for i in range(len(balls)):
            vadr = model.joint(f"ball{i}").dofadr[0]
            if np.linalg.norm(data.qvel[vadr:vadr + 3]) > 0.02:
                stats[f"ball{i}"]["last_move_t"] = data.time
        if renderer is not None and step % steps_per_frame == 0:
            renderer.update_scene(data, camera="fixed")
            frames.append(renderer.render())

    for i in range(len(balls)):
        adr = model.joint(f"ball{i}").qposadr[0]
        vadr = model.joint(f"ball{i}").dofadr[0]
        stats[f"ball{i}"].update(
            final_pos=[round(v, 4) for v in data.qpos[adr:adr + 3]],
            final_speed=round(float(np.linalg.norm(data.qvel[vadr:vadr + 3])), 4),
            rest_seconds=round(duration - stats[f"ball{i}"]["last_move_t"], 2))
    if renderer is not None:
        import imageio.v2 as imageio
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, np.stack(frames), fps=fps)
        renderer.close()
    return stats


# ---------------------------------------------------------------- presets --

def bowl_preset():
    # A rolling ball cannot climb the bowl's vertical outer wall, so A/B
    # approach over ramps that rise to lip height (mirror-symmetric).
    def ramp(side):
        return dict(type="ramp", pos=[side * 0.76, 0], yaw_deg=90 + side * 90,
                    length=0.62, width=0.5, angle_deg=12,
                    friction=[1.0, 0.01, 0.002])
    def scene(objs):
        return dict(duration=8.0, fps=25,
                    ground={"friction": [1.0, 0.01, 0.02]},
                    objects=[dict(type="bowl", pos=[0, 0], outer_radius=0.46,
                                  height=0.12, lip_width=0.10, depth=0.10,
                                  friction=[1.0, 0.01, 0.002]),
                             ramp(+1), ramp(-1)] + objs)
    ball = dict(type="ball", radius=0.08, friction=[1.0, 0.01, 0.002])
    return {
        "A": scene([{**ball, "pos": [1.2, 0, 0.08], "vel": [-2.15, 0, 0]}]),
        "B": scene([{**ball, "pos": [-1.2, 0, 0.08], "vel": [2.15, 0, 0]}]),
        "C": scene([{**ball, "pos": [0, 0, 0.10]}]),  # at rest inside the bowl
    }


def cube_preset():
    def arc(cx, direction, n=13):
        pts = []
        for k in range(n):
            s = k / (n - 1)
            th = math.pi + direction * math.pi * s
            pts.append(dict(t=round(3.2 * s, 3),
                            pos=[round(cx + 0.5 * math.cos(th), 4), 0,
                                 round(0.7 + 0.5 * math.sin(th), 4)]))
        return pts
    def scene(curve):
        return dict(duration=4.5, fps=25,
                    objects=[dict(type="cube", half=0.09, curve=curve)])
    return {"A": scene(arc(0.0, +1)), "B": scene(arc(0.0, -1)),
            "C": scene(arc(0.4, +1))}


PRESETS = {"bowl (true bowl on ground)": bowl_preset(), "cube arcs": cube_preset()}

# --------------------------------------------------------------------- ui --

HTML = (pathlib.Path(__file__).parent / "builder.html").read_text()


def make_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    app = FastAPI()

    @app.get("/")
    def index():
        return HTMLResponse(HTML.replace("__PRESETS__", json.dumps(PRESETS)))

    @app.post("/render")
    def render(payload: dict):
        s = payload.get("scenario")
        if s not in ("A", "B", "C"):
            raise HTTPException(400, "scenario must be A, B or C")
        try:
            stats = simulate(payload["scene"], CURRENT / f"{s}.mp4")
        except Exception as e:  # surface scene-spec errors to the UI
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        return {"url": f"/video/{s}.mp4", "stats": stats}

    @app.get("/video/{name}")
    def video(name: str):
        path = CURRENT / pathlib.Path(name).name
        if not path.exists():
            raise HTTPException(404, "not rendered yet")
        return FileResponse(path, media_type="video/mp4")

    @app.post("/eval")
    def eval_run(payload: dict):
        proc = subprocess.run(
            [sys.executable, "-m", "evals.measure", "--videos", str(CURRENT)],
            capture_output=True, text=True, cwd=REPO)
        return {"output": proc.stdout + proc.stderr}

    @app.post("/save")
    def save(payload: dict):
        name = "".join(c for c in payload.get("name", "unnamed") if c.isalnum() or c in "-_")
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / f"{name}.json").write_text(json.dumps(payload["scenes"], indent=1))
        return {"ok": True}

    @app.get("/list")
    def list_saved():
        return sorted(p.stem for p in WORK.glob("*.json"))

    @app.get("/load/{name}")
    def load(name: str):
        path = WORK / (pathlib.Path(name).name + ".json")
        if not path.exists():
            raise HTTPException(404, "unknown scenario")
        return json.loads(path.read_text())

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--selftest", action="store_true",
                        help="build+simulate the presets without rendering")
    args = parser.parse_args()
    if args.selftest:
        for pname, preset in PRESETS.items():
            for s, spec in preset.items():
                stats = simulate(spec, video_path=None)
                print(f"{pname} / {s}: {stats}")
        print("SELFTEST OK")
        return
    import uvicorn
    CURRENT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(make_app(), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
