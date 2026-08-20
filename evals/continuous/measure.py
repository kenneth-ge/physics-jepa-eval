"""Measure a pairwise-continuous eval, model-agnostically.

Each case dir holds N clips at parameter values theta (declared in
manifest.json, see evals/common/manifest.py) and each encoder is a black box:
video file -> one 1-D vector. For every encoder and every unordered clip pair
(i, j) we compute the embedding distance (cosine distance and L1) and score

  rho = Spearman( |theta_i - theta_j| ,  d(emb_i, emb_j) )

over all pairs. rho ~ +1 means embedding distance grows monotonically with the
physical parameter gap (the encoder resolves the parameter continuously);
rho ~ 0 means the parameter is not linearly readable from distances; the
contrastive suite's pass/fail is the special case of comparing just three
pairs. If the manifest names a "ref" clip we also report rho_ref over the
distances from ref only (N-1 pairs), which isolates one-vs-all monotonicity.

We also report NEIGHBOR ADJACENCY (nn_cos / nn_l1): sort the clips by theta;
for every interior clip, its two nearest neighbors in embedding space should
be exactly the clips right before and right after it on the ladder. The score
is the fraction of interior clips where that holds — a local-topology check
that is stricter than rho about ordering but indifferent to global scale.

As with the contrastive suite, exit code is 0 unless there is an error: the
goal is mapping where distance tracks the parameter, not gating on it.

Usage:
  python -m evals.continuous.measure --root <dir of case subdirs> \
      --encoders saved:vjepa2_raw saved:vjepa2_mean ...
"""

import argparse
import itertools
import pathlib

import numpy as np

from ..common.manifest import read_manifest
from ..encoders import DEFAULT_ENCODERS, get_encoder


def _cos_dist(a, b):
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _l1(a, b):
    return float(np.abs(a - b).sum())


def _rank(x):
    """Average ranks (ties averaged), no scipy dependency."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else 0.0


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
    """Returns {encoder_name: {"rho_cos", "rho_l1", "nn_cos", "nn_l1",
    "rho_ref_cos", "rho_ref_l1", "n_clips"}} for one case dir with a manifest."""
    man = read_manifest(case_dir)
    clips = man["clips"]
    ref = man.get("ref")
    results = {}
    for name, enc in encoders:
        vecs = [np.asarray(enc.encode(case_dir / c["file"])) for c in clips]
        thetas = [float(c["theta"]) for c in clips]
        gaps, dcos, dl1 = [], [], []
        for i, j in itertools.combinations(range(len(clips)), 2):
            gaps.append(abs(thetas[i] - thetas[j]))
            dcos.append(_cos_dist(vecs[i], vecs[j]))
            dl1.append(_l1(vecs[i], vecs[j]))
        res = {
            "rho_cos": spearman(gaps, dcos),
            "rho_l1": spearman(gaps, dl1),
            "nn_cos": neighbor_acc(vecs, thetas, _cos_dist),
            "nn_l1": neighbor_acc(vecs, thetas, _l1),
            "n_clips": len(clips),
        }
        if ref is not None:
            r = next(k for k, c in enumerate(clips) if c["file"] == ref)
            others = [k for k in range(len(clips)) if k != r]
            rg = [abs(thetas[k] - thetas[r]) for k in others]
            res["rho_ref_cos"] = spearman(
                rg, [_cos_dist(vecs[k], vecs[r]) for k in others])
            res["rho_ref_l1"] = spearman(
                rg, [_l1(vecs[k], vecs[r]) for k in others])
        results[name] = res
    return results


def print_summary(root_name, per_case, enc_names, metric="rho", title=""):
    cols = [(e, f"{metric}_{m}") for e in enc_names for m in ("cos", "l1")]
    header = "".join(f"{e.split(':')[-1][:9]+'_'+m.split('_')[-1]:>14}"
                     for e, m in cols)
    print(f"\n===== continuous {metric}: {root_name} {title} =====")
    print(f"{'case':<20}{header}")
    for case, res in per_case.items():
        row = "".join(f"{res[e][m]:>+14.3f}" for e, m in cols)
        print(f"{case:<20}{row}")
    n = len(per_case)
    means = "".join(
        f"{np.mean([per_case[c][e][m] for c in per_case]):>+14.3f}"
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
            ref = (f"   ref cos {r['rho_ref_cos']:+.3f} l1 {r['rho_ref_l1']:+.3f}"
                   if "rho_ref_cos" in r else "")
            print(f"[{key} :: {name}] n={r['n_clips']} "
                  f"rho cos {r['rho_cos']:+.3f} l1 {r['rho_l1']:+.3f} "
                  f"nn cos {r['nn_cos']:.2f} l1 {r['nn_l1']:.2f}{ref}")
    enc_names = [n for n, _ in encoders]
    root_name = (args.root or args.case).name
    print_summary(root_name, per_case, enc_names, metric="rho",
                  title="(Spearman |dtheta| vs embedding distance, +1 = tracks)")
    print_summary(root_name, per_case, enc_names, metric="nn",
                  title="(frac. interior clips whose 2 embedding-NN = ladder neighbors)")


if __name__ == "__main__":
    main()
