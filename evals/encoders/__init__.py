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


# name -> zero-arg factory. Each model runs in its own env; vjepa2 shares the
# default env, the rest feed `measure` via `saved:<tag>` (see precompute.py).
_REGISTRY = {
    "vjepa2_raw": lambda: _vjepa("raw"),
    "vjepa2_mean": lambda: _vjepa("mean"),
    "fastwam_raw": lambda: _fastwam("raw"),
    "fastwam_mean": lambda: _fastwam("mean"),
    "qwen3vl_raw": lambda: _qwen3vl("raw"),
    "qwen3vl_mean": lambda: _qwen3vl("mean"),
    "cosmos_raw": lambda: _cosmos("raw"),
    "cosmos_mean": lambda: _cosmos("mean"),
}

DEFAULT_ENCODERS = ["vjepa2_raw", "vjepa2_mean"]


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
