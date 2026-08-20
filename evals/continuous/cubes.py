"""Shared multicoloured-cube field used by the camera-motion families.

A case is a static scene of N cubes with distinct hues, random sizes and yaw
angles, placed without overlap inside a family-supplied region. The camera
families (translation, rotation) differ only in how they move the camera over
that scene, so the scene sampling lives here.

`translation.py` still carries its own copy of this logic; it will migrate
here once its in-flight re-render lands (changing its sampling now would
desynchronise the scenes from the vectors being computed for it).
"""

import colorsys

import numpy as np

from ..common.xml_scene import fr

MIN_HUE_GAP = 0.06        # min circular hue distance between any two cubes


def hue_dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def sample_field(rng, *, n, half_range, min_sep, in_region, tries=4000):
    """Place up to `n` cubes: dicts of pos (xy), half, yaw, rgba.

    `in_region(p)` decides whether a centre is admissible, so a family can use
    a box, a disc, or anything else. Hues are spread evenly around the wheel
    and then jittered, so every pair clears MIN_HUE_GAP by construction.
    Returns fewer than `n` cubes only if the region is too tight to fit them.
    """
    positions = []
    for _ in range(tries):
        if len(positions) == n:
            break
        p = rng.uniform(-1.0, 1.0, 2)
        p = np.array([p[0] * half_range[0], p[1] * half_range[1]])
        if in_region(p) and all(
                np.linalg.norm(p - q) >= min_sep for q in positions):
            positions.append(p)
    n = len(positions)
    hues = ((np.arange(n) + rng.uniform(0, 1)) / max(n, 1)
            + rng.uniform(-0.25, 0.25, n) * MIN_HUE_GAP) % 1.0
    rng.shuffle(hues)
    cubes = []
    for k in range(n):
        half = float(rng.uniform(*CUBE_HALF))
        r, g, b = colorsys.hsv_to_rgb(float(hues[k]),
                                      float(rng.uniform(0.85, 1.0)),
                                      float(rng.uniform(0.85, 1.0)))
        cubes.append(dict(pos=positions[k], half=half,
                          yaw=float(rng.uniform(0, 90)), rgba=(r, g, b, 1.0)))
    return cubes


CUBE_HALF = (0.07, 0.11)


def cube_geoms(cubes):
    """<geom> elements for a sampled field, each cube resting on the floor."""
    xml = ""
    for k, c in enumerate(cubes):
        h = c["half"]
        xml += (f'<geom name="cube_{k}" type="box" size="{h} {h} {h}" '
                f'pos="{fr(list(c["pos"]) + [h])}" euler="0 0 {c["yaw"]:.1f}" '
                f'rgba="{fr(np.round(c["rgba"], 3))}"/>')
    return xml


def check_field(cubes, *, n_min, min_sep):
    """Overlap + hue-separation failures (framing is the family's business)."""
    failures = []
    if len(cubes) < n_min:
        failures.append(f"only placed {len(cubes)} cubes (< {n_min})")
    for k, c in enumerate(cubes):
        for m in range(k + 1, len(cubes)):
            if np.linalg.norm(c["pos"] - cubes[m]["pos"]) < min_sep:
                failures.append(f"cubes {k},{m} closer than min_sep")
            if hue_dist(colorsys.rgb_to_hsv(*c["rgba"][:3])[0],
                        colorsys.rgb_to_hsv(*cubes[m]["rgba"][:3])[0]) < MIN_HUE_GAP:
                failures.append(f"cubes {k},{m} hues too close")
    return failures
