"""Encode the fixed FastWAM prompt once and cache (context, context_mask).

FastWAM training conditions every sample on a T5-encoded task prompt
(fastwam robot_video_dataset.py DEFAULT_PROMPT); encode_prompt zeroes the
padding positions and returns an all-ones mask, so zeros-with-mask-on MEANS
padding — the all-zeros context our encoder used to pass was structurally
OOD. This script loads the model WITH its text encoder (config ships
load_text_encoder=false), encodes one neutral prompt, and caches the tensors;
every clip is then conditioned on the identical context (constant across all
A/B/C comparisons).

Run once in the fastwam venv on a GPU box (FASTWAM_CONFIG set):
  python scripts/make_fastwam_context.py --out /data/fastwam/fixed_ctx.pt
"""

import argparse
import os

# DEFAULT_PROMPT template from fastwam/datasets/lerobot/robot_video_dataset.py
# (hardcoded rather than imported: the dataset module drags in lerobot deps).
PROMPT = ("A video recorded from a robot's point of view executing the "
          "following instruction: observe the scene.")


def main():
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/data/fastwam/fixed_ctx.pt")
    args = parser.parse_args()

    cfg = OmegaConf.load(os.environ["FASTWAM_CONFIG"])
    cfg.proprio_dim = 8
    cfg.video_dit_config.action_dim = 7
    cfg.action_dit_config.action_dim = 7
    cfg.video_dit_config.use_gradient_checkpointing = False
    cfg.action_dit_config.use_gradient_checkpointing = False
    cfg.load_text_encoder = True   # the one place we need the T5
    model = instantiate(cfg, model_dtype=torch.bfloat16,
                        device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        ctx, mask = model.encode_prompt(PROMPT)
    torch.save({"context": ctx.detach().cpu(),
                "context_mask": mask.detach().cpu(),
                "prompt": PROMPT}, args.out)
    print(f"wrote {args.out}: prompt={PROMPT!r} "
          f"context {tuple(ctx.shape)} mask {tuple(mask.shape)}")


if __name__ == "__main__":
    main()
