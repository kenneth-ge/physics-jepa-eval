"""V-JEPA 2 behind the black-box Encoder interface.

Whole-clip encode (uniform 64-frame subsample), keep the temporal token
positions in the last `seconds`, then reduce to a 1-D vector:
  readout="raw"  -> the kept token grid flattened in order (position-aligned)
  readout="mean" -> tokens averaged over spatial positions (position-blind)

The forward pass for a (model, video, seconds) triple is cached, so running
the raw and mean readouts over the same clips costs one pass, not two.
"""

import functools
import pathlib

import numpy as np

from ..common.video import load_video
from .base import Encoder

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 64   # fpc64 checkpoint
TUBELET = 2       # frames per temporal token position
LAST_TOKENS = 8   # resample the last-second span to this many temporal tokens,
                  # so the raw (flattened) vector has a fixed dim even when
                  # variants differ in duration (e.g. roll/occlusion B = 2x).

_backbone = {}


def _load(model_id):
    if model_id not in _backbone:
        import torch
        from transformers import AutoModel, AutoVideoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoVideoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        _backbone[model_id] = (processor, model, device)
    return _backbone[model_id]


@functools.lru_cache(maxsize=512)
def _last_second_grid(model_id, path_str, seconds):
    """(T', S, D) numpy array of the kept last-second token grid."""
    import torch
    processor, model, device = _load(model_id)
    frames, fps = load_video(pathlib.Path(path_str))
    duration = len(frames) / fps
    idx = np.round(np.linspace(0, len(frames) - 1, NUM_FRAMES)).astype(int)
    inputs = processor(videos=list(frames[idx]), return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    tokens = out.last_hidden_state[0].float().cpu()
    grid = tokens.reshape(NUM_FRAMES // TUBELET, -1, tokens.shape[-1])
    times = (idx[0::TUBELET] + idx[1::TUBELET] + 1) / 2 / fps
    ls = grid[times >= duration - seconds]
    if len(ls) == 0:
        ls = grid[-1:]
    # Resample the last-second span to a fixed temporal length so the raw
    # (flattened) vector has a fixed dim regardless of clip duration.
    sel = np.round(np.linspace(0, len(ls) - 1, LAST_TOKENS)).astype(int)
    return ls[sel].numpy()


@functools.lru_cache(maxsize=64)
def _predicted_grid(model_id, path_str, seconds):
    """(LAST_TOKENS, S, D) PREDICTED last-second token grid: the JEPA
    predictor's output at the last-second token positions, given only the
    pre-last-second tokens as context. CAVEAT: the HF API exposes masks only
    at the predictor interface — the encoder itself attends bidirectionally
    over the whole clip, so context tokens are not strictly future-blind.
    This is a predictor rollout of the end state, not an encoding of it."""
    import torch
    processor, model, device = _load(model_id)
    frames, fps = load_video(pathlib.Path(path_str))
    duration = len(frames) / fps
    idx = np.round(np.linspace(0, len(frames) - 1, NUM_FRAMES)).astype(int)
    inputs = processor(videos=list(frames[idx]), return_tensors="pt").to(device)
    times = (idx[0::TUBELET] + idx[1::TUBELET] + 1) / 2 / fps

    pv = inputs["pixel_values_videos"]
    patch = model.config.patch_size
    S = (pv.shape[-2] // patch) * (pv.shape[-1] // patch)
    T = NUM_FRAMES // TUBELET
    tgt_t = np.flatnonzero(times >= duration - seconds)
    if len(tgt_t) == 0:
        tgt_t = np.array([T - 1])
    ctx_t = np.flatnonzero(times < duration - seconds)
    if len(ctx_t) == 0:
        ctx_t = np.array([0])          # sub-1s clips: minimal context

    def flat(ts):
        i = (np.asarray(ts)[:, None] * S + np.arange(S)[None, :]).ravel()
        return torch.tensor(i, device=device)[None]

    with torch.inference_mode():
        out = model(**inputs, context_mask=[flat(ctx_t)],
                    target_mask=[flat(tgt_t)])
        assert out.last_hidden_state.shape[1] == T * S, out.last_hidden_state.shape
        pred = out.predictor_output.last_hidden_state[0].float().cpu()
    grid = pred.reshape(len(tgt_t), S, -1)
    sel = np.round(np.linspace(0, len(tgt_t) - 1, LAST_TOKENS)).astype(int)
    return grid[sel].numpy()


class VJEPA2Encoder(Encoder):
    """raw/mean = encoder tokens of the observed last second; pred_raw /
    pred_mean = the predictor's FORECAST of the last-second tokens from the
    preceding context (see _predicted_grid caveat)."""

    def __init__(self, readout="raw", model_id=MODEL_ID, seconds=1.0):
        assert readout in ("raw", "mean", "pred_raw", "pred_mean")
        self.readout = readout
        self.model_id = model_id
        self.seconds = seconds
        self.name = f"vjepa2_{readout}"

    def encode(self, video_path):
        fn = _predicted_grid if self.readout.startswith("pred") else _last_second_grid
        grid = fn(self.model_id, str(video_path), self.seconds)
        if self.readout.endswith("mean"):
            return grid.mean(axis=(0, 1)).ravel()   # position-blind
        return grid.ravel()                          # position-aligned
