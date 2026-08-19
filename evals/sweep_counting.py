"""Run basic_counting across multiple seeds and aggregate pass rates.

For each seed and each count X, render A/B/C (C = X+1 fresh prisms by
default), encode with V-JEPA 2 (loaded ONCE), and tally how often the
invariant holds per readout/metric. Also reports the mean normalized margin
  m = (min(d_AC, d_BC) - d_AB) / d_AB
so a value near 0 means "essentially a three-way tie" and >0 means the
invariant genuinely separates {A,B} from C.

Usage: python -m evals.sweep_counting --seeds 8 --out-root /data/videos/counting_sweep
"""

import argparse
import itertools
import pathlib

import numpy as np

from .basic_counting import sample_poses, render_scene
from .measure import invariant
from .encoders import DEFAULT_ENCODERS, get_encoder


def build_scenes(x, seed, c_mode):
    rng = np.random.default_rng([seed, x])
    pool = sample_poses(2 * x, rng)
    C = pool if c_mode == "union" else sample_poses(x + 1, rng)
    return {"A": pool[:x], "B": pool[x:], "C": C}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--counts", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--c-mode", choices=("fresh", "union"), default="fresh")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    p = parser.parse_args()

    encoders = [(name, get_encoder(name)) for name in p.encoders]
    enc_names = [n for n, _ in encoders]
    # accumulators[x][key] = list of per-seed values (bool passes / margins)
    keys = [f"{e}_{m}" for e in enc_names for m in ("cos", "l1")]
    passes = {x: {k: [] for k in keys} for x in p.counts}
    margins = {x: {e: [] for e in enc_names} for x in p.counts}

    for seed in range(p.seeds):
        for x in p.counts:
            scenes = build_scenes(x, seed, p.c_mode)
            out = p.out_root / f"seed{seed}" / f"eval_{x:02d}"
            paths = {}
            for name, poses in scenes.items():
                render_scene(poses, out / f"{name}.mp4", p.duration, p.fps, p.size)
                paths[name] = out / f"{name}.mp4"
            for ename, enc in encoders:
                vecs = {n: enc.encode(paths[n]) for n in ("A", "B", "C")}
                res = invariant(vecs)
                passes[x][f"{ename}_cos"].append(res["cos_pass"])
                passes[x][f"{ename}_l1"].append(res["l1_pass"])
                l1 = res["l1"]
                margins[x][ename].append((min(l1["AC"], l1["BC"]) - l1["AB"]) / l1["AB"])
        print(f"seed {seed} done", flush=True)

    print(f"\n=== basic_counting sweep: {p.seeds} seeds, c-mode={p.c_mode}, "
          f"encoders={enc_names} ===")
    print("pass rate = fraction of seeds where the invariant holds "
          "(chance under noise ~= 1/3); marg = mean normalized L1 margin "
          "(min(d_AC,d_BC)-d_AB)/d_AB per encoder")
    rate_cols = "".join(f"{k:>13}" for k in keys)
    marg_cols = "".join(f"{e+'_marg':>13}" for e in enc_names)
    print(f"{'X':>3} {'A/C':>7} |{rate_cols} |{marg_cols}")
    for x in p.counts:
        cnt = f"{x}/{x+1}" if p.c_mode == "fresh" else f"{x}/{2*x}"
        rates = "".join(f"{np.mean(passes[x][k]):>13.2f}" for k in keys)
        margs = "".join(f"{np.mean(margins[x][e]):>+13.3f}" for e in enc_names)
        print(f"{x:>3} {cnt:>7} |{rates} |{margs}")
    agg = "".join(f"{np.mean([passes[x][k] for x in p.counts]):>13.2f}" for k in keys)
    print(f"{'agg':>3} {'':>7} |{agg}")


if __name__ == "__main__":
    main()
