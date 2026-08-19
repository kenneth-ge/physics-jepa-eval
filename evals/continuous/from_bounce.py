"""Adapt an existing contrastive bounce sweep into continuous cases, in place.

The bounce sweep tree is <root>/seed*/factor_F/{A,B,C}.mp4 where only C's
restitution varies with F (r2 = F * r1) and A is the r1 reference. That is
already a restitution ladder per seed: C across factors plus one A as ref.
This writes a manifest.json into each seed dir pointing at the existing clips
(theta = restitution), so `evals.continuous.measure --root <root>` re-scores
the sweep's precomputed vectors with zero re-rendering or re-encoding.

The degenerate control factor (1.0, C == A) is skipped by default: a duplicate
clip contributes theta-gap ties against the ref pair, not signal.

Usage:
  python -m evals.continuous.from_bounce --root /data/videos/bounce_sweep
  python -m evals.continuous.measure --root /data/videos/bounce_sweep \
      --encoders saved:vjepa2_raw saved:vjepa2_mean ...
"""

import argparse
import pathlib

from ..common.manifest import write_manifest

R1_DEFAULT = 0.45


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--r1", type=float, default=R1_DEFAULT,
                        help="base restitution (theta of the A reference)")
    parser.add_argument("--ref-factor", type=float, default=None,
                        help="factor dir whose A.mp4 is the ref (default: lowest)")
    parser.add_argument("--keep-control", action="store_true",
                        help="keep the factor==1.0 control clip (C identical to A)")
    args = parser.parse_args()

    seeds = sorted(args.root.glob("seed*"))
    if not seeds:
        raise SystemExit(f"no seed* dirs under {args.root}")
    for seed in seeds:
        factors = sorted((float(fd.name.split("_", 1)[1]), fd)
                         for fd in seed.glob("factor_*"))
        clips = [{"file": f"{fd.name}/C.mp4", "theta": f * args.r1}
                 for f, fd in factors
                 if args.keep_control or abs(f - 1.0) > 1e-9]
        ref_fd = (min(factors)[1] if args.ref_factor is None else
                  next(fd for f, fd in factors
                       if abs(f - args.ref_factor) < 1e-9))
        clips.append({"file": f"{ref_fd.name}/A.mp4", "theta": args.r1})
        path = write_manifest(seed, "restitution", clips,
                              ref=f"{ref_fd.name}/A.mp4")
        print(f"{path}: {len(clips)} clips, "
              f"theta {clips[0]['theta']:.3f}..{clips[-2]['theta']:.3f} + ref")


if __name__ == "__main__":
    main()
