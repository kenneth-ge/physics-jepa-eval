"""V-JEPA 2-AC (action-conditioned) behind the black-box Encoder interface.

Why this exists: the base V-JEPA 2 predictor was pretrained with multi-block
masks at temporal mask ratio ~1.0 (spatial blocks spanning ALL frames), so it
was never trained to predict later timesteps from earlier ones — our
vjepa2_next/pred readouts are out-of-distribution probes. V-JEPA 2-AC IS
trained in the temporal direction (next-frame latent prediction on DROID), so
it is the fair test of whether that caveat is doing real work.

Two properties that make it a better forecasting probe than base V-JEPA:
- The encoder runs PER FRAME (each frame is duplicated to fill the 2-frame
  tubelet), so ALL temporal information flows through the predictor and the
  rollout is genuinely future-blind — no bidirectional leak.
- The predictor is block-causal: its output at position t is the prediction of
  frame t+1, so one teacher-forced pass yields the forecast of the frame just
  past the clip at the last position. No autoregression needed for horizon 1.

Conditioning (see research notes; all raw SI units, NO normalization anywhere
in the model, config or checkpoint):
- actions [B,T,7] are RELATIVE deltas (xyz metres, euler xyz radians, gripper
  closedness delta). ZERO IS IN-DISTRIBUTION and is the mode: it means "the
  arm did not move", the DROID loader applies no idle filter, and Meta's own
  CEM planner uses a zero-mean action prior and hard-zeros the rotation dims.
- states [B,T,7] are ABSOLUTE end-effector poses in the robot base frame.
  ZERO IS NOT VALID here (EE at the base origin with identity orientation is
  kinematically impossible), so we hold a plausible DROID pose constant. With
  a zero action the pose is unchanged step to step, so a constant state is
  exactly self-consistent with the zero-action sequence.
- gripper state 0.0 = OPEN. The paper's ablation shows an open gripper makes
  the world model keep object positions unchanged, whereas a closed one drags
  objects with the (here absent) arm — open is the right neutral prior.

Caveat to keep in mind when reading results: our MuJoCo scenes have no robot,
no table and no Franka, so the action coordinate frame is undefined (the paper
notes the model infers it from the image and errs when the base is not
visible) and the frames are OOD for a DROID-trained encoder. A zero action is
the only defensible choice precisely because any nonzero one would be
interpreted in an arbitrary, scene-dependent frame. Treat a null result as
informative rather than as evidence about the architecture.

Needs the facebookresearch/vjepa2 repo on sys.path (VJEPA2AC_REPO) and the
11.8GB checkpoint (VJEPA2AC_CKPT); see scripts/setup_vjepa2ac.sh. Runs in its
own env; precompute -> saved:vjepa2ac_*.
"""

import functools
import os
import pathlib
import sys

import numpy as np

from ..common.video import load_video
from .base import Encoder

CKPT = os.environ.get("VJEPA2AC_CKPT", "/data/vjepa2ac/vjepa2-ac-vitg.pt")
REPO = os.environ.get("VJEPA2AC_REPO", "/data/vjepa2ac/vjepa2")

IMG = 256
PATCH = 16
DIM = 1408
TPF = (IMG // PATCH) ** 2      # 256 tokens per frame
MAX_FRAMES = 64                # causal mask supports T <= MAX_FRAMES // 2
FPS = 4.0                      # DROID training rate
MAX_T = MAX_FRAMES // 2

# Zero action = "did not move" (in-distribution, the planner's own no-op).
NULL_ACTION = [0.0] * 7
# A plausible DROID end-effector pose, gripper OPEN; absolute, so NOT zeros.
NULL_STATE = [0.55, 0.0, 0.30, -3.10, 0.0, -1.90, 0.0]

_backbone = {}


def _clean(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v
            for k, v in sd.items()}


def _load():
    if "m" not in _backbone:
        import torch
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from src.models import ac_predictor as vit_ac_predictor
        from src.models import vision_transformer as vit_encoder

        enc = vit_encoder.vit_giant_xformers(
            patch_size=PATCH, img_size=(IMG, IMG), num_frames=MAX_FRAMES,
            tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
            uniform_power=False, use_rope=True)
        pred = vit_ac_predictor.vit_ac_predictor(
            img_size=(IMG, IMG), patch_size=PATCH, num_frames=MAX_FRAMES,
            tubelet_size=2, embed_dim=DIM)

        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        missing, unexpected = enc.load_state_dict(_clean(sd["encoder"]),
                                                  strict=False)
        # strict=False is what the repo does; surface any real mismatch.
        assert not [k for k in missing if "pos_embed" not in k], missing
        pred.load_state_dict(_clean(sd["predictor"]), strict=True)
        del sd

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _backbone["m"] = (enc.to(device).eval(), pred.to(device).eval(), device)
    return _backbone["m"]


def _frames_at_fps(path):
    """Resample the clip to the model's 4 fps, capped at the causal-mask limit
    and keeping the END of the clip (our readouts are last-second based)."""
    frames, fps = load_video(pathlib.Path(path))
    n = max(2, min(MAX_T, int(round(len(frames) / fps * FPS))))
    idx = np.round(np.linspace(0, len(frames) - 1, n)).astype(int)
    return frames[idx]


@functools.lru_cache(maxsize=64)
def _next_frame_grid(path_str):
    """(TPF, DIM) forecast of the frame just past the clip, conditioned on a
    constant null action + constant plausible pose."""
    import torch
    import torch.nn.functional as F
    enc, pred, device = _load()

    frames = _frames_at_fps(path_str)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1) * 255.0
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1) * 255.0
    x = torch.from_numpy(np.ascontiguousarray(frames)).float().permute(3, 0, 1, 2)
    x = ((x - mean) / std).to(device)
    T = x.shape[1]
    # Per-frame encoding: duplicate each frame to fill the 2-frame tubelet.
    clip = x.permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 2, 1, 1)

    with torch.inference_mode():
        h = enc(clip)                                   # [T, TPF, DIM]
        z = h.reshape(1, T * h.shape[1], h.shape[-1])
        z = F.layer_norm(z, (z.shape[-1],))             # normalize_reps=True
        a = torch.tensor(NULL_ACTION, device=device).view(1, 1, 7).repeat(1, T, 1)
        s = torch.tensor(NULL_STATE, device=device).view(1, 1, 7).repeat(1, T, 1)
        out = pred(z, a, s)
        out = F.layer_norm(out, (out.shape[-1],))
        # Block-causal: the last frame's slot holds the forecast of frame T+1.
        nxt = out[0, -h.shape[1]:].float().cpu()
    return nxt.numpy()


class VJEPA2ACEncoder(Encoder):
    """next_raw / next_mean = the AC predictor's forecast of the frame just
    past the clip, under a constant null action and a constant plausible
    end-effector pose."""

    def __init__(self, readout="next_raw"):
        assert readout in ("next_raw", "next_mean")
        self.readout = readout
        self.name = f"vjepa2ac_{readout}"

    def encode(self, video_path):
        grid = _next_frame_grid(str(video_path))
        if self.readout.endswith("mean"):
            return grid.mean(axis=0).ravel()
        return grid.ravel()
