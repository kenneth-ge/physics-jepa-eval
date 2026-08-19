"""Rotation helpers shared by scene builders (MuJoCo is z-up)."""

import math

import mujoco
import numpy as np


def rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_str(R):
    """Quaternion attribute string for a 3x3 rotation matrix."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return " ".join(f"{v:.6f}" for v in q)
