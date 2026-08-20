"""Per-case clip manifest: which clips exist and what parameter each holds.

Contrastive cases are self-describing (A/B/C.mp4) and skip this. Continuous
families write a manifest.json into each case dir so `continuous.measure`
reads clip -> theta without parsing filenames:

  {"param": "restitution", "clips": [{"file": "theta_00.mp4", "theta": 0.45}, ...]}

Anything family-specific may be added alongside — readers ignore unknown keys.
"""

import json
import pathlib

NAME = "manifest.json"


def write_manifest(case_dir, param, clips, **extra):
    """clips: list of {"file": <name.mp4>, "theta": <float>} dicts."""
    data = {"param": param, "clips": clips, **extra}
    path = pathlib.Path(case_dir) / NAME
    path.write_text(json.dumps(data, indent=1))
    return path


def read_manifest(case_dir):
    return json.loads((pathlib.Path(case_dir) / NAME).read_text())
