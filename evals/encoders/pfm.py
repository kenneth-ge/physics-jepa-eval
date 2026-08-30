"""PFM (Pantheon physics foundation model) checkpoint ladder behind the
black-box Encoder interface.

The ladder checkpoints (kenny-dev /data/pfm/pfm-latent-eval-ladder-checkpoints,
manifest.json maps role -> file) are ~199M-param Llama-style trunks (48 layers,
dim 384, SwiGLU 3072) that consume PRE-ENCODED tokens, not pixels: per frame,
256 visual tokens of dim 384 plus slot/modality/view/source embeddings. Heads:
visual_head (masked-encoding prediction), beast_head (action tokens),
latent_head 384->32 — the readout we use, per its name and the zip's
("pfm-latent-eval-ladder").

Input pipeline (user-decided 2026-08-20): DINOv3 ViT-S/16 patch tokens as the
visual encodings, visual-only sequence (action/proprio tokens omitted). The
ungated timm re-host `timm/vit_small_patch16_dinov3.lvd1689m` serves the same
lvd1689m weights as the gated facebook repo; at 256px input it yields exactly
16x16=256 patch tokens of dim 384 — matching slot_emb(256) and in_proj(384).

Assumptions where the training repo is unavailable (identical across all six
checkpoints, so ladder-internal comparisons stay valid even if one deviates
from training): token = in_proj(dino) + slot_emb[patch] + modality_emb[0]
+ view_emb[0] + source_emb[0]; frame-major flatten; bidirectional attention;
Meta-Llama RoPE (6 heads of 64, theta 1e4) over the flattened sequence;
RMSNorm eps 1e-5. F32/F16/F4 roles read 32/16/4 uniformly sampled frames
(PFM_FRAMES overrides).

Readouts over latent_head outputs of the clip's last second (frame tokens
resampled to LAST_TOKENS frames, vjepa2-style):
  raw  -> (LAST_TOKENS, 256, 32) flattened, position-aligned
  mean -> 32-d mean over frames and slots, position-blind
"""

import functools
import json
import os
import pathlib

import numpy as np

from ..common.video import load_video
from .base import Encoder

CKPT_DIR = os.environ.get("PFM_CKPT_DIR", "/data/pfm/pfm-latent-eval-ladder-checkpoints")
DINO_ID = "hf-hub:timm/vit_small_patch16_dinov3.lvd1689m"
DINO_SIZE = 256          # 16x16 patches -> 256 tokens, matches slot_emb
N_LAYERS, DIM, N_HEADS, FFN, N_SLOTS = 48, 384, 6, 3072, 256
MOD_VISUAL = 0           # modality_emb row for visual tokens (assumed)
LAST_TOKENS = 8          # fixed temporal length of the last-second readout

#: role -> (manifest role string, context frames)
ROLES = {
    "worst": ("F32 worst", 32),
    "low": ("F32 low", 32),
    "middle": ("F32 middle", 32),
    "best": ("F32 best", 32),
    "f16": ("F16 control", 16),
    "f4": ("F4 control", 4),
}

_state = {}   # shared DINOv3 + per-role trunks, loaded once per process


def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    torch = _torch()
    return torch.bfloat16 if _device() == "cuda" else torch.float32


def _build_trunk():
    """Llama-reference-style module whose state dict matches the checkpoint
    exactly (strict load); unused heads/embeddings included for strictness."""
    torch = _torch()
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            x32 = x.float()
            x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
            return (x32 * self.weight.float()).to(x.dtype)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(DIM, DIM, bias=False)
            self.wk = nn.Linear(DIM, DIM, bias=False)
            self.wv = nn.Linear(DIM, DIM, bias=False)
            self.wo = nn.Linear(DIM, DIM, bias=False)

        def forward(self, x, freqs_cis):
            torch = _torch()
            B, T, _ = x.shape
            hd = DIM // N_HEADS
            q = self.wq(x).view(B, T, N_HEADS, hd)
            k = self.wk(x).view(B, T, N_HEADS, hd)
            v = self.wv(x).view(B, T, N_HEADS, hd)
            # Meta-Llama RoPE: interleaved pairs as complex numbers.
            qc = torch.view_as_complex(q.float().reshape(B, T, N_HEADS, hd // 2, 2))
            kc = torch.view_as_complex(k.float().reshape(B, T, N_HEADS, hd // 2, 2))
            fc = freqs_cis[:T].view(1, T, 1, hd // 2)
            q = torch.view_as_real(qc * fc).flatten(3).to(x.dtype)
            k = torch.view_as_real(kc * fc).flatten(3).to(x.dtype)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return self.wo(out.transpose(1, 2).reshape(B, T, DIM))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(DIM, FFN, bias=False)
            self.w2 = nn.Linear(FFN, DIM, bias=False)
            self.w3 = nn.Linear(DIM, FFN, bias=False)

        def forward(self, x):
            torch = _torch()
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.feed_forward = FeedForward()
            self.attention_norm = RMSNorm(DIM)
            self.ffn_norm = RMSNorm(DIM)

        def forward(self, x, freqs_cis):
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            return x + self.feed_forward(self.ffn_norm(x))

    class ContEnc(nn.Module):     # unused (visual-only), present for strict load
        def __init__(self):
            super().__init__()
            self.value_projection = nn.Linear(50, DIM)
            self.validity_projection = nn.Linear(50, DIM, bias=False)

    class PFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask_emb = nn.Parameter(torch.zeros(DIM))
            self.in_proj = nn.Linear(DIM, DIM)
            self.act_emb = nn.Embedding(290, DIM)
            self.modality_emb = nn.Embedding(3, DIM)
            self.view_emb = nn.Embedding(4, DIM)
            self.source_emb = nn.Embedding(4, DIM)
            self.slot_emb = nn.Embedding(N_SLOTS, DIM)
            self.layers = nn.ModuleList(Block() for _ in range(N_LAYERS))
            self.norm = RMSNorm(DIM)
            self.visual_head = nn.Linear(DIM, DIM)
            self.beast_head = nn.Linear(DIM, 256)
            self.latent_head = nn.Linear(DIM, 32)
            self.canonical_cont_enc = ContEnc()

        def latents(self, vis_tokens):
            """vis_tokens (T, S, 384) DINOv3 features -> (T, S, 32) latents."""
            torch = _torch()
            T, S, _ = vis_tokens.shape
            x = self.in_proj(vis_tokens)
            x = x + self.slot_emb.weight[None, :S]
            x = x + self.modality_emb.weight[MOD_VISUAL]
            x = x + self.view_emb.weight[0] + self.source_emb.weight[0]
            x = x.reshape(1, T * S, DIM)
            theta = 10000.0 ** (-torch.arange(0, DIM // N_HEADS, 2,
                                              device=x.device).float() / (DIM // N_HEADS))
            pos = torch.arange(T * S, device=x.device).float()
            freqs_cis = torch.polar(torch.ones_like(pos[:, None] * theta[None]),
                                    pos[:, None] * theta[None])
            for layer in self.layers:
                x = layer(x, freqs_cis)
            return self.latent_head(self.norm(x)).reshape(T, S, -1)

    return PFM()


def _manifest_file(role):
    man = json.load(open(pathlib.Path(CKPT_DIR) / "manifest.json"))
    want = ROLES[role][0]
    for m in man:
        if m["role"] == want:
            return pathlib.Path(CKPT_DIR) / m["filename"]
    raise KeyError(f"role {want!r} not in {CKPT_DIR}/manifest.json")


def _load_trunk(role):
    key = f"trunk:{role}"
    if key not in _state:
        torch = _torch()
        ckpt = torch.load(_manifest_file(role), map_location="cpu", weights_only=True)
        model = _build_trunk()
        model.load_state_dict(ckpt["model"], strict=True)
        _state[key] = model.to(_device(), _dtype()).eval()
    return _state[key]


def _load_dino():
    if "dino" not in _state:
        torch = _torch()
        import timm
        model = timm.create_model(DINO_ID, pretrained=True, num_classes=0,
                                  img_size=DINO_SIZE).to(_device(), _dtype()).eval()
        cfg = timm.data.resolve_model_data_config(model)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        _state["dino"] = (model, mean, std)
    return _state["dino"]


@functools.lru_cache(maxsize=64)
def _dino_tokens(path_str, n_frames):
    """(T, 256, 384) float32 DINOv3 patch tokens + per-frame times, duration.
    Checkpoint-independent, so all six trunks share one cache entry."""
    torch = _torch()
    model, mean, std = _load_dino()
    frames, fps = load_video(pathlib.Path(path_str))
    duration = len(frames) / fps
    idx = np.round(np.linspace(0, len(frames) - 1, n_frames)).astype(int)
    x = torch.from_numpy(frames[idx]).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(DINO_SIZE, DINO_SIZE),
                                        mode="bicubic", align_corners=False)
    x = ((x - mean) / std).to(_device(), _dtype())
    with torch.inference_mode():
        toks = model.forward_features(x)
    toks = toks[:, model.num_prefix_tokens:, :]     # drop cls + register tokens
    assert toks.shape[1] == N_SLOTS, f"expected {N_SLOTS} patch tokens, got {toks.shape}"
    return toks.float().cpu(), idx / fps, duration


@functools.lru_cache(maxsize=64)
def _latent_grid(role, path_str, seconds):
    """(LAST_TOKENS, 256, 32) numpy: last-second latent_head outputs."""
    torch = _torch()
    n_frames = int(os.environ.get("PFM_FRAMES", 0)) or ROLES[role][1]
    toks, times, duration = _dino_tokens(path_str, n_frames)
    trunk = _load_trunk(role)
    with torch.inference_mode():
        lat = trunk.latents(toks.to(_device(), _dtype())).float().cpu()
    ls = lat[times >= duration - seconds]
    if len(ls) == 0:
        ls = lat[-1:]
    sel = np.round(np.linspace(0, len(ls) - 1, LAST_TOKENS)).astype(int)
    return ls[sel].numpy()


class PFMEncoder(Encoder):
    def __init__(self, role, readout="raw", seconds=1.0):
        assert role in ROLES and readout in ("raw", "mean")
        self.role = role
        self.readout = readout
        self.seconds = seconds
        self.name = f"pfm_{role}_{readout}"

    def encode(self, video_path):
        grid = _latent_grid(self.role, str(video_path), self.seconds)
        if self.readout == "mean":
            return grid.mean(axis=(0, 1)).ravel()   # 32-d, position-blind
        return grid.ravel()                          # position-aligned
