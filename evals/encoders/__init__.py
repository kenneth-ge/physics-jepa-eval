"""Encoder registry. Add a model by implementing base.Encoder and adding a
factory entry here; nothing else in the eval harness needs to change.

Factories import their model module lazily so that heavy / env-specific deps
(FastWAM's repo, Cosmos, big VLMs) only load when that encoder is requested.
`saved:<tag>` is special: it reads precomputed vectors written by
`evals.precompute` in another environment (see saved.py).
"""

from .base import Encoder
from .saved import SavedEncoder


def _vjepa(readout):
    from .vjepa2 import VJEPA2Encoder
    return VJEPA2Encoder(readout)


def _fastwam(readout):
    from .fastwam import FastWAMEncoder
    return FastWAMEncoder(readout)


def _qwen3vl(readout):
    from .qwen3vl import Qwen3VLEncoder
    return Qwen3VLEncoder(readout)


def _cosmos(readout):
    from .cosmos import CosmosEncoder
    return CosmosEncoder(readout)


def _pfm(role, readout):
    from .pfm import PFMEncoder
    return PFMEncoder(role, readout)


# name -> zero-arg factory. Each model runs in its own env; vjepa2 shares the
# default env, the rest feed `measure` via `saved:<tag>` (see precompute.py).
_REGISTRY = {
    "vjepa2_raw": lambda: _vjepa("raw"),
    "vjepa2_mean": lambda: _vjepa("mean"),
    "vjepa2_pred_raw": lambda: _vjepa("pred_raw"),
    "vjepa2_pred_mean": lambda: _vjepa("pred_mean"),
    "vjepa2_fut_raw": lambda: _vjepa("fut_raw"),
    "vjepa2_fut_mean": lambda: _vjepa("fut_mean"),
    "vjepa2_next_raw": lambda: _vjepa("next_raw"),
    "vjepa2_next_mean": lambda: _vjepa("next_mean"),
    "fastwam_raw": lambda: _fastwam("raw"),
    "fastwam_mean": lambda: _fastwam("mean"),
    "fastwam_z_raw": lambda: _fastwam("z_raw"),
    "fastwam_z_mean": lambda: _fastwam("z_mean"),
    "fastwam_zf_raw": lambda: _fastwam("zf_raw"),
    "fastwam_zf_mean": lambda: _fastwam("zf_mean"),
    "qwen3vl_raw": lambda: _qwen3vl("raw"),
    "qwen3vl_mean": lambda: _qwen3vl("mean"),
    "cosmos_raw": lambda: _cosmos("raw"),
    "cosmos_mean": lambda: _cosmos("mean"),
    "cosmos_pred_raw": lambda: _cosmos("pred_raw"),
    "cosmos_pred_mean": lambda: _cosmos("pred_mean"),
}

# PFM checkpoint ladder (see pfm.py): pfm_{role}_{raw,mean} for the six
# manifest roles. All share one DINOv3 tokenizer cache in-process.
for _role in ("worst", "low", "middle", "best", "f16", "f4"):
    for _ro in ("raw", "mean"):
        _REGISTRY[f"pfm_{_role}_{_ro}"] = (
            lambda r=_role, o=_ro: _pfm(r, o))

# The canonical V-JEPA pair: the observed last-second state (raw) and the
# NEXT-token forecast past the clip (next_raw), which is the readout that
# exposes dynamics — the observed state alone misses them (roll 0/30 vs 30/30
# for the extrapolated readout). `mean`, `pred_*` and `fut_*` stay registered
# for comparisons but are no longer the default.
DEFAULT_ENCODERS = ["vjepa2_raw", "vjepa2_next_raw"]


def available():
    return sorted(_REGISTRY) + ["saved:<tag>"]


def get_encoder(name):
    if name.startswith("saved:"):
        return SavedEncoder(name.split(":", 1)[1])
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; available: {available()}")
    return _REGISTRY[name]()


def register(name, factory):
    _REGISTRY[name] = factory
