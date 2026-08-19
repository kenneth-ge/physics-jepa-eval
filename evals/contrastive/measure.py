"""Measure an A/B/C eval, model-agnostically.

Each encoder is a black box: video file -> one 1-D vector (evals/encoders/).
This module only ever sees vectors, so it works with any registered model.
For every encoder we test the invariant in cosine similarity and L1 distance:

  A,B (same end state, different history) closer to each other than to C.

Both are reported per encoder. The default encoder set is vjepa2_raw
(position-aligned, the unreduced comparison) and vjepa2_mean (position-blind
contrast probe). Exit code is always 0 unless there is an error: the goal is
mapping where the invariant holds and where it fails, not gating on it.

Usage:
  python -m evals.measure --videos <dir with A/B/C.mp4>
  python -m evals.measure --root  <dir of eval subdirs>
  python -m evals.measure --root  <dir> --encoders vjepa2_raw other_model
"""

import argparse
import itertools
import pathlib

import numpy as np

from ..encoders import DEFAULT_ENCODERS, get_encoder

PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _l1(a, b):
    return float(np.abs(a - b).sum())


def invariant(vecs):
    """vecs: {'A','B','C' -> 1-D vector}. Returns cos/l1 tables + pass flags."""
    cos = {x + y: _cos(vecs[x], vecs[y]) for x, y in PAIRS}
    l1 = {x + y: _l1(vecs[x], vecs[y]) for x, y in PAIRS}
    return {
        "cos": cos, "l1": l1,
        "cos_pass": cos["AB"] > cos["AC"] and cos["AB"] > cos["BC"],
        "l1_pass": l1["AB"] < l1["AC"] and l1["AB"] < l1["BC"],
    }


def measure_dir(videos, encoders, out=None):
    """Returns {encoder_name: invariant-result}. `encoders` is a list of
    (name, Encoder). Saves per-encoder A/B/C vectors to `out` if given."""
    saved = {}
    results = {}
    for name, enc in encoders:
        vecs = {n: enc.encode(videos / f"{n}.mp4") for n in ("A", "B", "C")}
        saved[name] = vecs
        results[name] = invariant(vecs)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **{f"{e}_{n}": v for e, d in saved.items()
                         for n, v in d.items()})
    return results


def print_detail(name, dirname, res):
    print(f"\n[{dirname} :: {name}]")
    print("pair  cosine   L1")
    for pair in ("AB", "AC", "BC"):
        print(f"{pair}    {res['cos'][pair]:.4f}   {res['l1'][pair]:.2f}")
    print(f"cosine invariant -> {'PASS' if res['cos_pass'] else 'FAIL'}   "
          f"L1 invariant -> {'PASS' if res['l1_pass'] else 'FAIL'}")


def print_summary(root_name, per_dir, enc_names):
    cols = [(e, m) for e in enc_names for m in ("cos", "l1")]
    header = "".join(f"{e.split('_')[-1]+'_'+m:>13}" for e, m in cols)
    print(f"\n===== summary: {root_name} (PASS = invariant holds) =====")
    print("  ".join(enc_names))
    print(f"{'eval':<16}{header}")
    tally = {c: 0 for c in cols}
    for dirname, res in per_dir.items():
        cells = []
        for e, m in cols:
            ok = res[e][f"{m}_pass"]
            tally[(e, m)] += bool(ok)
            cells.append(f"{'PASS' if ok else 'fail':>13}")
        print(f"{dirname:<16}" + "".join(cells))
    n = len(per_dir)
    print(f"{'TOTAL '+str(n):<16}" + "".join(f"{tally[c]:>8}/{n:<4}" for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--videos", type=pathlib.Path,
                       help="one dir containing A/B/C.mp4")
    group.add_argument("--root", type=pathlib.Path,
                       help="measure every immediate subdir that has A/B/C.mp4")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    encoders = [(name, get_encoder(name)) for name in args.encoders]

    if args.videos is not None:
        res = measure_dir(args.videos, encoders, args.out)
        for name, _ in encoders:
            print_detail(name, args.videos.name, res[name])
        return

    dirs = sorted(d for d in args.root.iterdir() if (d / "A.mp4").exists())
    per_dir = {}
    for d in dirs:
        res = measure_dir(d, encoders)
        for name, _ in encoders:
            print_detail(name, d.name, res[name])
        per_dir[d.name] = res
    print_summary(args.root.name, per_dir, [n for n, _ in encoders])


if __name__ == "__main__":
    main()
