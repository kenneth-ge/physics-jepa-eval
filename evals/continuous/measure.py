"""Measure a pairwise-continuous eval, model-agnostically.

Each case dir holds N clips at parameter values theta (declared in
manifest.json, see evals/common/manifest.py) and each encoder is a black box:
video file -> one 1-D vector. The metric is NEIGHBOR ADJACENCY: sort the clips
by theta; for every interior clip, its two nearest neighbors in embedding
space should be exactly the clips right before and right after it on the
ladder. The score (nn_cos / nn_l1, one per distance) is the fraction of
interior clips where that holds, reported for both cosine distance and L1.

Chance level is 1/C(N-1, 2) — 0.010 for a 16-clip ladder — since a clip's two
nearest neighbors could be any unordered pair drawn from the other N-1 clips.

Adjacency is a local-topology test: it asks whether the encoder resolves one
step along the parameter, not merely whether far-apart clips look far apart.
As with the contrastive suite, exit code is 0 unless there is an error: the
goal is mapping where local structure survives, not gating on it.

Usage:
  python -m evals.continuous.measure --root <dir of case subdirs> \
      --encoders saved:vjepa2_raw saved:vjepa2_mean ...
"""

import argparse
import pathlib

import numpy as np

from ..common.manifest import read_manifest
from ..encoders import DEFAULT_ENCODERS, get_encoder


def _cos_dist(a, b):
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _l1(a, b):
    return float(np.abs(a - b).sum())


def neighbor_acc(vecs, thetas, dist_fn):
    """Fraction of interior clips (theta-sorted) whose two nearest embedding
    neighbors are exactly the ladder-adjacent clips."""
    order = list(np.argsort(thetas))
    if len(order) < 3:
        return float("nan")
    hits = 0
    for k in range(1, len(order) - 1):
        i = order[k]
        dists = sorted((dist_fn(vecs[i], vecs[j]), j)
                       for j in range(len(vecs)) if j != i)
        hits += {j for _, j in dists[:2]} == {order[k - 1], order[k + 1]}
    return hits / (len(order) - 2)


def measure_case(case_dir, encoders):
    """Returns {encoder_name: {"nn_cos", "nn_l1", "n_clips"}} for one case dir
    with a manifest."""
    clips = read_manifest(case_dir)["clips"]
    results = {}
    for name, enc in encoders:
        vecs = [np.asarray(enc.encode(case_dir / c["file"])) for c in clips]
        thetas = [float(c["theta"]) for c in clips]
        results[name] = {
            "nn_cos": neighbor_acc(vecs, thetas, _cos_dist),
            "nn_l1": neighbor_acc(vecs, thetas, _l1),
            "n_clips": len(clips),
        }
    return results


def print_summary(root_name, per_case, enc_names):
    cols = [(e, f"nn_{m}") for e in enc_names for m in ("cos", "l1")]
    header = "".join(f"{e.split(':')[-1][:9]+'_'+m.split('_')[-1]:>14}"
                     for e, m in cols)
    print(f"\n===== continuous nn: {root_name} "
          f"(frac. interior clips whose 2 embedding-NN = ladder neighbors) =====")
    print(f"{'case':<20}{header}")
    for case, res in per_case.items():
        row = "".join(f"{res[e][m]:>14.3f}" for e, m in cols)
        print(f"{case:<20}{row}")
    n = len(per_case)
    means = "".join(
        f"{np.mean([per_case[c][e][m] for c in per_case]):>14.3f}"
        for e, m in cols)
    print(f"{'MEAN '+str(n):<20}{means}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", type=pathlib.Path,
                       help="one case dir containing manifest.json + clips")
    group.add_argument("--root", type=pathlib.Path,
                       help="measure every subdir (recursively) with a manifest.json")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    args = parser.parse_args()

    encoders = [(name, get_encoder(name)) for name in args.encoders]
    cases = ([args.case] if args.case is not None else
             sorted(p.parent for p in args.root.rglob("manifest.json")))
    if not cases:
        raise SystemExit(f"no case dirs with manifest.json under {args.root}")

    per_case = {}
    for d in cases:
        res = measure_case(d, encoders)
        key = str(d.relative_to(args.root)) if args.root else d.name
        per_case[key] = res
        for name, _ in encoders:
            r = res[name]
            print(f"[{key} :: {name}] n={r['n_clips']} "
                  f"nn cos {r['nn_cos']:.2f} l1 {r['nn_l1']:.2f}")
    print_summary((args.root or args.case).name, per_case,
                  [n for n, _ in encoders])


if __name__ == "__main__":
    main()
