"""Run an in-environment encoder over eval videos and save the vectors.

For each A/B/C.mp4 under the eval subdirs, encode with the chosen encoder and
write `<name>__<tag>.npy` beside the video. `measure --encoders saved:<tag>`
then reads these — so a model with incompatible deps runs here in its own
environment and still lands in the same comparison tables.

The saved tag is the encoder name, so `measure --encoders saved:<name>` picks
it up. Multiple encoders that share a backbone (e.g. qwen3vl_raw/_mean) load
the model once when passed together.

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
                        help="dir of eval subdirs each with A/B/C.mp4")
    parser.add_argument("--encoders", nargs="+", required=True,
                        help="registered encoder name(s) to run in THIS env")
    parser.add_argument("--recursive", action="store_true",
                        help="find every dir with A.mp4 under --root (e.g. sweep seeds)")
    args = parser.parse_args()

    encoders = [(name, get_encoder(name)) for name in args.encoders]
    if args.recursive:
        dirs = sorted({p.parent for p in args.root.rglob("A.mp4")})
    else:
        dirs = sorted(d for d in args.root.iterdir() if (d / "A.mp4").exists())
    if not dirs:
        raise SystemExit(f"no eval subdirs with A.mp4 under {args.root}")
    for d in dirs:
        for tag, enc in encoders:
            for name in ("A", "B", "C"):
                vec = np.asarray(enc.encode(d / f"{name}.mp4"))
                np.save(d / f"{name}__{tag}.npy", vec)
        print(f"{d.name}: wrote vectors for {[t for t, _ in encoders]} "
              f"(dim={vec.shape})", flush=True)


if __name__ == "__main__":
    main()
