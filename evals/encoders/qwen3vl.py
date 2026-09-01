"""Qwen3.6 (natively-multimodal MoE) behind the black-box Encoder interface.

A vision-language model with native video. We take an intermediate decoder
layer's hidden state (middle+1), restrict to the video tokens, and pool the
last-second temporal span.

Default Qwen/Qwen3.6-35B-A3B (MoE ~35B/3B active, 40 layers, hidden 2048 →
mid+1 = layer 21; class Qwen3_5MoeForConditionalGeneration via
AutoModelForImageTextToText, not gated, ~70GB bf16 — fits one H200). Set
QWEN_MODEL_ID for the FP8 variant / dense sibling. Needs its own env
(transformers >= 4.57.1); run via `evals.precompute`, measure via
`saved:qwen3vl_*`.
"""

import os

import numpy as np

from .base import Encoder

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
VIDEO_TOKEN_ID = 248057   # Qwen3.6; overridden by config.video_token_id if present
FPS = 2.0


def _num_layers(cfg):
    tc = getattr(cfg, "text_config", None)
    return getattr(tc, "num_hidden_layers", None) or cfg.num_hidden_layers


def _temporal_patch(cfg):
    vc = getattr(cfg, "vision_config", None)
    return getattr(vc, "temporal_patch_size", 2) if vc else 2


_state = {}


def _load():
    if _state:
        return _state
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_id = os.environ.get("QWEN_MODEL_ID", DEFAULT_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype="auto", device_map="auto",
        attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(model_id)
    _state.update(model=model, processor=processor,
                  mid=_num_layers(model.config) // 2 + 1,      # index into hidden_states
                  vid_tok=getattr(model.config, "video_token_id", VIDEO_TOKEN_ID),
                  tps=_temporal_patch(model.config))
    return _state


class Qwen3VLEncoder(Encoder):
    """raw/mean pool the observed last second. next_raw/next_mean take ONLY
    the FINAL temporal step's video tokens: the model is causal, so the hidden
    state at the last position is the state from which the next token is
    predicted — the closest available approximation to the eval's
    W(o_1:t) = z_{t+1} (a predicted next latent) for a VLM with no explicit
    latent-forecast head."""

    def __init__(self, readout="raw"):
        assert readout in ("raw", "mean", "next_raw", "next_mean")
        self.readout = readout
        self.name = f"qwen3vl_{readout}"

    def encode(self, video_path):
        import torch
        s = _load()
        model, processor = s["model"], s["processor"]
        messages = [{"role": "user", "content": [
            {"type": "video", "video": str(video_path), "fps": FPS},
            {"type": "text", "text": "Describe the final state."}]}]
        # Qwen3.6 processor samples frames + builds video tokens natively.
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            video_load_backend="decord", do_sample_frames=True, fps=FPS,
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states[s["mid"]][0]                  # (seq, hidden)
        ids = inputs["input_ids"][0]
        vid = hs[ids == s["vid_tok"]]                        # (t*h*w, hidden)
        t, h, w = inputs["video_grid_thw"][0].tolist()
        vid = vid.view(t, h * w, -1)                          # (t, tokens, hidden)
        if self.readout.startswith("next"):
            steps = 1                                        # final position only
        else:
            steps = max(1, round(FPS / s["tps"]))            # ~last second
        last = vid[-steps:].reshape(-1, vid.shape[-1]).float().cpu().numpy()
        return (last.mean(axis=0).ravel() if self.readout.endswith("mean")
                else last.ravel())
