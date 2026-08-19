"""FastWAM (ActionDiT) behind the black-box Encoder interface.

FastWAM is a robot world-action model, not a video encoder: it conditions on
the CURRENT observation (VAE-encoded first frame) + text context + proprio and
denoises an action. We use it as a feature extractor by capturing the hidden
state of the middle+1 ActionDiT block during one forward pass.

Key facts (from reading github.com/yuantianyuan01/FastWAM):
- Target stack: `model.action_expert.blocks`, 30 DiTBlocks, hidden dim 1024.
  middle+1 index = 30//2 + 1 = 16.
- A plain forward hook on blocks[16] never fires (MoT decomposes blocks
  manually); we hook `blocks[16].gate` and keep its LAST call (= block output,
  shape [1, 32, 1024]).
- Image input is TWO camera views concatenated horizontally to 224x448, RGB,
  normalized to [-1, 1]. Our stereo rig provides these (A.mp4 + A_right.mp4).
- Runs in its OWN environment (torch 2.7.1 / transformers 4.49); use
  `evals.precompute` there to write vectors, then measure via `saved:fastwam_*`.

Env vars: FASTWAM_CONFIG (path to configs/model/fastwam.yaml),
FASTWAM_CKPT (the 12GB libero_uncond_2cam224.pt), DIFFSYNTH_MODEL_BASE_PATH.
This module imports the FastWAM repo lazily, only when instantiated.
"""

import os

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


class FastWAMEncoder(Encoder):
    def __init__(self, readout="raw"):
        assert readout in ("raw", "mean")
        self.readout = readout
        self.name = f"fastwam_{readout}"

    def _input_image(self, video_path):
        import torch
        left = _last_frame_224(video_path)
        right_path = video_path.parent / f"{video_path.stem}_right.mp4"
        right = _last_frame_224(right_path) if right_path.exists() else left
        img = np.concatenate([left, right], axis=1)          # (224, 448, 3)
        t = torch.from_numpy(img).permute(2, 0, 1)[None]     # (1, 3, 224, 448)
        return (t.float() * (2.0 / 255.0) - 1.0)

    def encode(self, video_path):
        import torch
        model = _load_model()
        dev = model.device
        dt = model.torch_dtype
        image = self._input_image(video_path).to(dev, dt)
        context = torch.zeros(1, 128, 4096, device=dev, dtype=dt)
        context_mask = torch.ones(1, 128, device=dev, dtype=torch.bool)

        blocks = model.action_expert.blocks
        i = len(blocks) // 2 + 1
        captured = {}

        def hook(_m, _in, out):
            captured["h"] = out.detach()   # last call == block output [1,32,1024]

        handle = blocks[i].gate.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model.infer_action(prompt=None, input_image=image,
                                   action_horizon=32, proprio=None,
                                   context=context, context_mask=context_mask,
                                   num_inference_steps=1, seed=0)
        finally:
            handle.remove()

        grid = captured["h"][0].float().cpu().numpy()        # (32, 1024)
        return grid.mean(axis=0).ravel() if self.readout == "mean" else grid.ravel()
