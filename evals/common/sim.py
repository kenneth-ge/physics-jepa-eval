"""Simulation/render loops shared by eval generators."""

import mujoco
import numpy as np

STOP_SPEED = 0.02  # m/s: below this a ball counts as stopped


def rolling_spin(vel, radius):
    """Angular velocity for rolling without slipping on flat ground."""
    return [-vel[1] / radius, vel[0] / radius, 0.0]


class BallTracker:
    """Records a free-jointed ball's trajectory statistics during rollout."""

    def __init__(self, model, joint_name="ball"):
        self.qadr = model.joint(joint_name).qposadr[0]
        self.vadr = model.joint(joint_name).dofadr[0]
        self.t_last_moving = 0.0
        self.min_x = np.inf
        self.max_x = -np.inf
        self.final_pos = None
        self.final_speed = None

    def post_step(self, data):
        x = data.qpos[self.qadr]
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        speed = float(np.linalg.norm(data.qvel[self.vadr:self.vadr + 3]))
        if speed > STOP_SPEED:
            self.t_last_moving = data.time
        self.final_pos = data.qpos[self.qadr:self.qadr + 3].copy()
        self.final_speed = speed

    def summary(self, duration):
        return {
            "final_pos": self.final_pos,
            "final_speed": self.final_speed,
            "stopped_dur": duration - self.t_last_moving,
            "min_x": float(self.min_x),
            "max_x": float(self.max_x),
        }


def rollout(model, data, *, duration, fps, size=256, camera="fixed",
            pre_step=None, post_step=None, render=True):
    """Step physics for `duration`, optionally rendering at `fps`.

    pre_step(data) runs before each mj_step (e.g. to drive mocap bodies);
    post_step(data) runs after (e.g. trackers). Returns frames or None.
    """
    renderer = mujoco.Renderer(model, height=size, width=size) if render else None
    steps_per_frame = max(1, round(1.0 / (fps * model.opt.timestep)))
    frames = []
    for i in range(int(duration / model.opt.timestep)):
        if pre_step is not None:
            pre_step(data)
        mujoco.mj_step(model, data)
        if post_step is not None:
            post_step(data)
        if renderer is not None and i % steps_per_frame == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    if renderer is not None:
        renderer.close()
        return frames
    return None


def render_cams(model, data, *, cam_names, n_frames, advance, size=256,
                state_fn=None):
    """Render `n_frames` from multiple cameras. `advance(data, i)` prepares
    frame i (step physics, set mocap, forward, ...). Returns
    ({cam: [frames]}, [state per frame])."""
    renderer = mujoco.Renderer(model, height=size, width=size)
    cam_frames = {c: [] for c in cam_names}
    states = []
    for i in range(n_frames):
        advance(data, i)
        for c in cam_names:
            renderer.update_scene(data, camera=c)
            cam_frames[c].append(renderer.render())
        if state_fn is not None:
            states.append(state_fn(data))
    renderer.close()
    return cam_frames, states


def render_static(model, data, *, duration, fps, size=256, camera="fixed"):
    """Render a static scene: one forward pass, one frame, tiled to length."""
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=size, width=size)
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
    renderer.close()
    return [frame] * int(round(duration * fps))


def render_static_rig(model, data, *, cam_names, duration, fps, size=256):
    """Static scene from every camera in a rig: one forward pass, one frame per
    camera, each tiled to length. Returns {cam: [frames]} for write_observations."""
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=size, width=size)
    n = int(round(duration * fps))
    cam_frames = {}
    for c in cam_names:
        renderer.update_scene(data, camera=c)
        cam_frames[c] = [renderer.render()] * n
    renderer.close()
    return cam_frames
