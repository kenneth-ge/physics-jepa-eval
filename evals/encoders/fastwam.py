"""FastWAM (ActionDiT) behind the black-box Encoder interface.

FastWAM is a robot world-action model, not a video encoder: it conditions on
the CURRENT observation (VAE-encoded first frame) + text context + proprio and
denoises an action. We use it as a feature extractor by capturing the hidden
state of the middle+1 ActionDiT block during one forward pass.

Key facts (from reading github.com/yuantianyuan01/FastWAM):
- Target stack: `model.action_expert.blocks`, 30 DiTBlocks, hidden dim 1024.
  middle+1 index = 30//2 + 1 = 16.
- A plain forward hook on blocks[16] never fires (MoT decomposes blocks
  manually); we hook each block's `.gate` and keep its LAST call (= block
  output). Three taps per forward: action-expert mid+1 [1, 32, 1024], and
  video-expert mid+1 / final [1, 98, 3072] (the z the action head reads).
- Conditioning: the FIXED cached prompt context (FASTWAM_CTX, built by
  scripts/make_fastwam_context.py) — training always used a real T5 prompt
  embedding; zeros-with-mask-on decodes as all-padding (OOD). Proprio is
  omitted uniformly (the checkpoint is proprio-conditioned; a uniform shift).
- Image input is TWO camera views concatenated horizontally to 224x448, RGB,
  normalized to [-1, 1]. Our stereo rig provides these (A.mp4 + A_right.mp4).
- Runs in its OWN environment (torch 2.7.1 / transformers 4.49); use
  `evals.precompute` there to write vectors, then measure via `saved:fastwam_*`.

Env vars: FASTWAM_CONFIG (path to configs/model/fastwam.yaml),
FASTWAM_CKPT (the 12GB libero_uncond_2cam224.pt), DIFFSYNTH_MODEL_BASE_PATH.
This module imports the FastWAM repo lazily, only when instantiated.
"""

import functools
import os
import pathlib

import numpy as np

from ..common.video import load_video
from .base import Encoder

_model = None


def _last_frame_224(path):
    from PIL import Image
    frames, _ = load_video(path)
    img = Image.fromarray(frames[-1]).resize((224, 224), Image.BILINEAR)
    return np.asarray(img)  # (224, 224, 3) uint8


def _load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    cfg = OmegaConf.load(os.environ["FASTWAM_CONFIG"])
    # The released model config interpolates into the (absent) data + model
    # roots when loaded standalone; fill every dangling ref with its concrete
    # value from configs/data/libero_2cam.yaml (action 7, proprio 8) and the
    # literal mot_checkpoint_mixed_attn=true (checkpointing is a no-op here).
    cfg.proprio_dim = 8
    cfg.video_dit_config.action_dim = 7
    cfg.action_dit_config.action_dim = 7
    cfg.video_dit_config.use_gradient_checkpointing = False
    cfg.action_dit_config.use_gradient_checkpointing = False
    model = instantiate(cfg, model_dtype=torch.bfloat16,
                        device="cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.environ.get("FASTWAM_CKPT")
    if ckpt:
        model.load_checkpoint(ckpt)
    model.eval()
    _model = model
    return model


_ctx_cache = None


def _fixed_context(model):
    """The cached T5 embedding of the fixed prompt (built once by
    scripts/make_fastwam_context.py). Training conditions every sample on a
    real prompt embedding whose zero positions MEAN padding — an all-zeros
    context with an all-ones mask reads as a 128-token all-padding prompt
    (structurally OOD). Never pass zeros."""
    global _ctx_cache
    import torch
    if _ctx_cache is None:
        path = os.environ.get("FASTWAM_CTX", "/data/fastwam/fixed_ctx.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"fixed prompt context {path} missing — run "
                "scripts/make_fastwam_context.py once in the fastwam venv "
                "(GPU box) or set FASTWAM_CTX")
        d = torch.load(path, map_location="cpu")
        _ctx_cache = (d["context"], d["context_mask"])
    ctx, mask = _ctx_cache
    return (ctx.to(model.device, model.torch_dtype),
            mask.to(model.device, torch.bool))


def _stereo_image(video_path):
    import torch
    left = _last_frame_224(video_path)
    right_path = video_path.parent / f"{video_path.stem}_right.mp4"
    right = _last_frame_224(right_path) if right_path.exists() else left
    img = np.concatenate([left, right], axis=1)          # (224, 448, 3)
    t = torch.from_numpy(img).permute(2, 0, 1)[None]     # (1, 3, 224, 448)
    return (t.float() * (2.0 / 255.0) - 1.0)


VID_TOKENS = 98   # 224x448 -> VAE 16x + patchify [1,2,2] -> 7x14 tokens


@functools.lru_cache(maxsize=8)
def _capture(path_str):
    """ONE infer_action forward per clip, capturing three taps at once (each
    block's `gate` fires last with the block output):
      action    : action-expert mid+1 block   (32, 1024)
      vid_mid   : video-expert mid+1 block    (98, 3072)
      vid_final : video-expert final block    (98, 3072)
    The video expert runs once (prefill, clean frame-0 tokens at timestep 0 —
    the trained regime); action tokens then read it via shared-attention K/V."""
    import torch
    video_path = pathlib.Path(path_str)
    model = _load_model()
    image = _stereo_image(video_path).to(model.device, model.torch_dtype)
    context, context_mask = _fixed_context(model)

    a_blocks = model.action_expert.blocks
    v_blocks = model.mot.mixtures["video"].blocks
    taps = {"action": a_blocks[len(a_blocks) // 2 + 1],
            "vid_mid": v_blocks[len(v_blocks) // 2 + 1],
            "vid_final": v_blocks[-1]}
    captured, handles = {}, []
    for tap, blk in taps.items():
        def hook(_m, _in, out, tap=tap):
            captured[tap] = out.detach()   # last call == block output
        handles.append(blk.gate.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model.infer_action(prompt=None, input_image=image,
                               action_horizon=32, proprio=None,
                               context=context, context_mask=context_mask,
                               num_inference_steps=1, seed=0)
    finally:
        for h in handles:
            h.remove()

    missing = [t for t in taps if t not in captured]
    if missing:
        raise RuntimeError(
            f"taps {missing} never fired (captured {sorted(captured)}) — "
            "the video-expert prefill path may not route through block.gate")
    grids = {t: captured[t][0].float().cpu().numpy() for t in taps}
    assert grids["vid_mid"].shape[0] == VID_TOKENS, grids["vid_mid"].shape
    return grids


#: readout name -> (capture tap, pooled?)
READOUTS = {
    "raw": ("action", False), "mean": ("action", True),
    "z_raw": ("vid_mid", False), "z_mean": ("vid_mid", True),
    "zf_raw": ("vid_final", False), "zf_mean": ("vid_final", True),
}


class FastWAMEncoder(Encoder):
    """raw/mean = action-expert mid block (what the action head extracted);
    z_* = video-expert mid+1 block (the world representation the action
    expert cross-attends into); zf_* = video-expert final block."""

    def __init__(self, readout="raw"):
        assert readout in READOUTS
        self.readout = readout
        self.name = f"fastwam_{readout}"

    def encode(self, video_path):
        tap, pooled = READOUTS[self.readout]
        grid = _capture(str(video_path))[tap]
        return grid.mean(axis=0).ravel() if pooled else grid.ravel()
