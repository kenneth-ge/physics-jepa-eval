"""Render the bounce restitution sweep into <root>/seed{S}/factor_{F}/.

Renders A/B/C for every (seed, factor) with the chosen rig, so each model can
then be run via `evals.precompute` and combined by `evals.aggregate_bounce`.
The factor is encoded in the directory name so the aggregator can recover it.

Usage:
  python -m evals.render_bounce --out-root /data/videos/bounce_sweep \
      --seeds 6 --rig stereo
"""

import argparse
import pathlib

import numpy as np

from .bounce import FACTORS, R1_BASE, DURATION, FPS, build_case, render_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--factors", type=float, nargs="+", default=FACTORS)
    parser.add_argument("--r1", type=float, default=R1_BASE)
    parser.add_argument("--rig", default="mono")
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--size", type=int, default=256)
    p = parser.parse_args()

    for seed in range(p.seeds):
        for factor in p.factors:
            meta = build_case(seed, factor, r1=p.r1)
            out = p.out_root / f"seed{seed}" / f"factor_{factor:.2f}"
            out.mkdir(parents=True, exist_ok=True)
            render_case(out, meta, rig=p.rig, fps=p.fps, size=p.size,
                        duration=p.duration)
            print(f"seed{seed} factor_{factor:.2f}: r1={meta['r1']:.2f} "
                  f"r2={meta['r2']:.2f}", flush=True)
        print(f"seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
