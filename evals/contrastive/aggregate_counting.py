"""Aggregate a rendered counting sweep across seeds, for arbitrary encoders.

Reads <root>/seed*/eval_XX/{A,B,C}.mp4 (+ any precomputed `__<tag>.npy`
vectors) and reports, per count X and per encoder, the fraction of seeds
where the invariant holds (cosine and L1) plus the mean normalized L1 margin.
Runs in the default env: vjepa2 loads its model; `saved:<tag>` encoders read
vectors precomputed in another env, so every model lands in one table without
re-rendering.

Usage:
  python -m evals.aggregate_counting --root /data/videos/counting_sweep \
      --encoders vjepa2_raw vjepa2_mean saved:qwen3vl_raw saved:qwen3vl_mean
"""

import argparse
import collections
import pathlib

import numpy as np

from ..encoders import DEFAULT_ENCODERS, get_encoder
from .measure import measure_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    args = parser.parse_args()

    encoders = [(name, get_encoder(name)) for name in args.encoders]
    seeds = sorted(args.root.glob("seed*"))
    if not seeds:
        raise SystemExit(f"no seed* dirs under {args.root}")

    # acc[X][enc] = {"cos": [...], "l1": [...], "marg": [...]}
    acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"cos": [], "l1": [], "marg": []}))
    for seed in seeds:
        for evd in sorted(seed.glob("eval_*")):
            x = int(evd.name.split("_")[1])
            res = measure_dir(evd, encoders)
            for name, _ in encoders:
                r = res[name]
                acc[x][name]["cos"].append(r["cos_pass"])
                acc[x][name]["l1"].append(r["l1_pass"])
                l1 = r["l1"]
                acc[x][name]["marg"].append(
                    (min(l1["AC"], l1["BC"]) - l1["AB"]) / l1["AB"])
        print(f"{seed.name} done", flush=True)

    names = [n for n, _ in encoders]
    print(f"\n===== counting sweep ({len(seeds)} seeds) — pass rate per count "
          f"(chance ~0.33) =====")
    cols = [(n, m) for n in names for m in ("cos", "l1")]
    header = "".join(f"{n.split(':')[-1][:9]+'_'+m:>14}" for n, m in cols)
    print(f"{'X':>3}{header}")
    for x in sorted(acc):
        row = "".join(f"{np.mean(acc[x][n][m]):>14.2f}" for n, m in cols)
        print(f"{x:>3}{row}")
    print("\nmean normalized L1 margin per count (>0 = invariant separates):")
    mh = "".join(f"{n.split(':')[-1][:11]:>13}" for n in names)
    print(f"{'X':>3}{mh}")
    for x in sorted(acc):
        row = "".join(f"{np.mean(acc[x][n]['marg']):>+13.3f}" for n in names)
        print(f"{x:>3}{row}")


if __name__ == "__main__":
    main()
