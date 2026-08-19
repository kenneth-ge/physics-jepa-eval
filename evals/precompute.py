"""Run an in-environment encoder over eval videos and save the vectors.

For every *.mp4 under the eval subdirs, encode with the chosen encoder and
write `<stem>__<tag>.npy` beside the video. `measure --encoders saved:<tag>`
then reads these — so a model with incompatible deps runs here in its own
environment and still lands in the same comparison tables.

Paradigm-agnostic: contrastive cases hold A/B/C.mp4, continuous cases hold a
manifest-described ladder of clips; both are just "encode every clip in the
dir". The saved tag is the encoder name, so `measure --encoders saved:<name>`
picks it up. Multiple encoders that share a backbone (e.g. qwen3vl_raw/_mean)
load the model once when passed together.

Usage (in the model's own venv):
  python -m evals.precompute --root /data/videos/cube --encoders qwen3vl_raw qwen3vl_mean
"""

import argparse
import pathlib

import numpy as np

from .encoders import get_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True,
                        help="dir of eval subdirs each holding clip mp4s")
    parser.add_argument("--encoders", nargs="+", required=True,
                        help="registered encoder name(s) to run in THIS env")
    parser.add_argument("--recursive", action="store_true",
                        help="find every dir with an mp4 under --root (e.g. sweep seeds)")
    args = parser.parse_args()

    encoders = [(name, get_encoder(name)) for name in args.encoders]
    if args.recursive:
        dirs = sorted({p.parent for p in args.root.rglob("*.mp4")})
    else:
        dirs = sorted(d for d in args.root.iterdir()
                      if d.is_dir() and any(d.glob("*.mp4")))
    if not dirs:
        raise SystemExit(f"no eval subdirs with mp4s under {args.root}")
    for d in dirs:
        clips = sorted(d.glob("*.mp4"))
        for tag, enc in encoders:
            for clip in clips:
                vec = np.asarray(enc.encode(clip))
                np.save(d / f"{clip.stem}__{tag}.npy", vec)
        print(f"{d.name}: wrote {len(clips)} vectors for "
              f"{[t for t, _ in encoders]} (dim={vec.shape})", flush=True)


if __name__ == "__main__":
    main()
