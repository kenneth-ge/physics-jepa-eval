"""Render the pendulum mass sweep into <root>/seed{S}/factor_{F}/.

Layout matches the bounce sweep so `evals.contrastive.aggregate_bounce` reads it
directly. factor = m_C / m (factor 1.00 = control, C == A).

  python -m evals.contrastive.render_pendulum --out-root /data/videos/pendulum \
      --seeds 8 --rig stereo
"""

import argparse
import pathlib

from .pendulum import FACTORS, FPS, build_case, render_case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--factors", type=float, nargs="+", default=FACTORS)
    ap.add_argument("--rig", default="mono")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--size", type=int, default=256)
    p = ap.parse_args()

    for seed in range(p.seeds):
        for factor in p.factors:
            meta = build_case(seed, factor)
            out = p.out_root / f"seed{seed}" / f"factor_{factor:.2f}"
            out.mkdir(parents=True, exist_ok=True)
            render_case(out, meta, rig=p.rig, fps=p.fps, size=p.size)
            print(f"seed{seed} factor_{factor:.2f}: m={meta['m']:.3f} "
                  f"m2={meta['m2']:.3f} L={meta['L']:.2f}", flush=True)
        print(f"seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
