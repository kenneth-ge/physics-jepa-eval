"""Render the deform stiffness sweep into <root>/seed{S}/factor_{F}/.

Layout matches the bounce sweep so `evals.contrastive.aggregate_bounce` reads it
directly. --variant {history,adjusted} selects the timing (see deform.py).

  python -m evals.contrastive.render_deform --out-root /data/videos/deform_history \
      --variant history --seeds 8 --rig stereo
"""

import argparse
import pathlib

from .deform import FACTORS, K1_BASE, build_case, render_case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    ap.add_argument("--variant", choices=("history", "adjusted"), default="history")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--factors", type=float, nargs="+", default=FACTORS)
    ap.add_argument("--rig", default="mono")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--size", type=int, default=256)
    p = ap.parse_args()

    for seed in range(p.seeds):
        for factor in p.factors:
            meta = build_case(seed, factor)
            out = p.out_root / f"seed{seed}" / f"factor_{factor:.2f}"
            out.mkdir(parents=True, exist_ok=True)
            render_case(out, meta, variant=p.variant, rig=p.rig, fps=p.fps,
                        size=p.size)
            print(f"seed{seed} factor_{factor:.2f}: k1={meta['k1']:.2f} "
                  f"k2={meta['k2']:.2f}", flush=True)
        print(f"seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
