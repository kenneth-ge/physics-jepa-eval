"""Aggregate the bounce restitution sweep: pass rate per multiplicative factor.

Reads <root>/seed*/factor_*/{A,B,C}.mp4 (+ precomputed __<tag>.npy), and for
each factor and encoder reports the fraction of seeds where the invariant holds
(cosine and L1) plus the mean normalized L1 margin. factor=1.00 is the control
(C is identical to A) and should sit at/below chance; the factor at which the
pass rate climbs is where the model starts resolving the restitution change.

Usage:
  python -m evals.aggregate_bounce --root /data/videos/bounce_sweep \
      --encoders saved:vjepa2_raw saved:vjepa2_mean saved:cosmos_raw ...
"""

import argparse
import collections
import pathlib

import numpy as np

from .encoders import DEFAULT_ENCODERS, get_encoder
from .measure import measure_dir


def _factor_of(dirname):
    return float(dirname.split("_", 1)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    args = parser.parse_args()

    encoders = [(name, get_encoder(name)) for name in args.encoders]
    seeds = sorted(args.root.glob("seed*"))
    if not seeds:
        raise SystemExit(f"no seed* dirs under {args.root}")

    # acc[factor][enc] = {"cos": [...], "l1": [...], "marg": [...]}
    acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"cos": [], "l1": [], "marg": []}))
    for seed in seeds:
        for fd in sorted(seed.glob("factor_*")):
            f = _factor_of(fd.name)
            res = measure_dir(fd, encoders)
            for name, _ in encoders:
                r = res[name]
                acc[f][name]["cos"].append(r["cos_pass"])
                acc[f][name]["l1"].append(r["l1_pass"])
                l1 = r["l1"]
                acc[f][name]["marg"].append(
                    (min(l1["AC"], l1["BC"]) - l1["AB"]) / l1["AB"])
        print(f"{seed.name} done", flush=True)

    names = [n for n, _ in encoders]
    print(f"\n===== bounce sweep ({len(seeds)} seeds) — pass rate per r2/r1 "
          f"factor (factor 1.00 = control, C==A) =====")
    cols = [(n, m) for n in names for m in ("cos", "l1")]
    header = "".join(f"{n.split(':')[-1][:9]+'_'+m:>14}" for n, m in cols)
    print(f"{'fac':>5}{header}")
    for f in sorted(acc):
        row = "".join(f"{np.mean(acc[f][n][m]):>14.2f}" for n, m in cols)
        print(f"{f:>5.2f}{row}")
    print("\nmean normalized L1 margin per factor (>0 = invariant separates):")
    mh = "".join(f"{n.split(':')[-1][:11]:>13}" for n in names)
    print(f"{'fac':>5}{mh}")
    for f in sorted(acc):
        row = "".join(f"{np.mean(acc[f][n]['marg']):>+13.3f}" for n in names)
        print(f"{f:>5.2f}{row}")


if __name__ == "__main__":
    main()
