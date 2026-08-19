"""Cosmos 3 (Cosmos3-Super) behind the black-box Encoder interface.

Cosmos 3 is a 64B multimodal Mixture-of-Transformers (a single
Cosmos3OmniTransformer: a causal text "understanding" stream + a bidirectional
"generation" stream that carries video/sound/action, sharing one stack of
MoE decoder layers). We use it as a feature extractor: run ONE transformer
forward on the clip's last few frames and capture the middle+1 decoder layer's
generation-stream (= video) hidden state.

Facts (diffusers main + nvidia/Cosmos3-Super config):
- Blocks: `pipe.transformer.layers`, 64 layers, hidden 5120; middle+1 = 33.
  Each layer returns (und_seq, gen_seq); the VIDEO stream is gen_seq = out[1].
- No output_hidden_states -> forward hook. No encode-only helper -> drive the
  pipeline once (num_inference_steps=1, guidance_scale=1.0 so a single forward,
  output_type="latent", safety checker off) and short-circuit via a sentinel.
- Not gated (OpenMDW1.1). Super bf16 ~120GB fits ONE H200 for a short, low-res
  encode-only forward. No official FP8 exists; set COSMOS_MODEL_ID=nvidia/
  Cosmos3-Nano (16B) if VRAM is tight (verify its layer count).
Runs in its own env (diffusers from git main); precompute -> saved:cosmos_*.
"""

import os

import numpy as np

from ..common.video import load_video
from .base import Encoder

DEFAULT_MODEL_ID = "nvidia/Cosmos3-Super"
N_FRAMES = 5          # Cosmos3 conditions on a few frames; last-second end state
SIZE = 256
FPS = 24.0

_state = {}


class _Stop(Exception):
    pass


def _load():
    if _state:
        return _state
    import torch
    from diffusers import Cosmos3OmniPipeline
    model_id = os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL_ID)
    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        enable_safety_checker=False)
    pipe.transformer.gradient_checkpointing = False
    try:
        pipe.transformer.set_attention_backend("native")   # GQA needs SDPA/native
    except Exception:
        pass
    n = pipe.transformer.config.num_hidden_layers
    _state.update(pipe=pipe, mid=n // 2 + 1)
    return _state


def _last_frames(video_path):
    from PIL import Image
    frames, _ = load_video(video_path)
    last = frames[-N_FRAMES:]
    return [Image.fromarray(f).resize((SIZE, SIZE), Image.BILINEAR) for f in last]


class CosmosEncoder(Encoder):
    def __init__(self, readout="raw"):
        assert readout in ("raw", "mean")
        self.readout = readout
        self.name = f"cosmos_{readout}"

    def encode(self, video_path):
        import torch
        s = _load()
        pipe = s["pipe"]
        video = _last_frames(video_path)

        captured = {}

        def hook(_m, _in, out):
            captured["h"] = out[1].detach().float()   # gen_seq = video stream (N,5120)
            raise _Stop()                              # skip VAE decode / sampling

        handle = pipe.transformer.layers[s["mid"]].register_forward_hook(hook)
        try:
            with torch.no_grad():
                pipe(video=video, prompt="", num_frames=len(video),
                     num_inference_steps=1, guidance_scale=1.0,
                     height=SIZE, width=SIZE, fps=FPS,
                     enable_safety_check=False, output_type="latent")
        except _Stop:
            pass
        finally:
            handle.remove()

        grid = captured["h"].cpu().numpy()            # (num_video_tokens, 5120)
        return grid.mean(axis=0).ravel() if self.readout == "mean" else grid.ravel()
