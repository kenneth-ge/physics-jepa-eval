"""Camera rigs for multi-modal observation rendering.

The `fixed` camera is the primary one every eval has always used (and the
one V-JEPA / measure read from A.mp4); extra cameras are added for models
that expect multi-view input (e.g. a robot world-action model). A rig is a
list of camera names to capture; `mono` is back-compatible (fixed only).
"""

import math

import numpy as np

from .xml_scene import DEFAULT_CAMERA, fr

PRIMARY = "fixed"
SCENE_CENTER = (0.0, 0.0, 0.30)

# Extra cameras defined by eye position + look-at target; xyaxes computed.
_EXTRA = {
    "left":  dict(pos=[-1.9, -2.1, 1.05], target=SCENE_CENTER, fovy=52),
    "right": dict(pos=[1.9, -2.1, 1.05], target=SCENE_CENTER, fovy=52),
    "top":   dict(pos=[0.0, -0.35, 2.6], target=SCENE_CENTER, fovy=52),
}

RIGS = {
    "mono": [PRIMARY],
    "stereo": [PRIMARY, "right"],
    "tri": [PRIMARY, "left", "right"],
}


def _xyaxes(pos, target):
    """MuJoCo camera xyaxes (world right + up) for an eye looking at target."""
    forward = np.array(target, float) - np.array(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return list(right) + list(up)


def extra_cameras_xml(rig):
    """<camera> elements for every non-primary camera in the rig."""
    xml = ""
    for name in RIGS[rig]:
        if name == PRIMARY:
            continue
        c = _EXTRA[name]
        xml += (f'<camera name="{name}" pos="{fr(c["pos"])}" '
                f'xyaxes="{fr(_xyaxes(c["pos"], c["target"]))}" fovy="{c["fovy"]}"/>')
    return xml


def rig_cameras(rig):
    return RIGS[rig]
