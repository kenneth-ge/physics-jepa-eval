"""Render the count3 nested-counting sweep into <root>/seed{S}/eval_{X:02d}/.

Layout matches basic_counting so `evals.contrastive.aggregate_counting` reads
it directly.

Usage:
  python -m evals.contrastive.render_count3 --out-root /data/videos/count3_sweep \
      --seeds 8 --rig stereo
"""

import argparse
import pathlib

from .count2 import render_scene
from .count3 import build_scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--counts", type=int, nargs="+", default=list(range(2, 11)))
    parser.add_argument("--rig", default="mono")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    p = parser.parse_args()

    for seed in range(p.seeds):
        for x in p.counts:
            scenes = build_scenes(x, seed)
            out = p.out_root / f"seed{seed}" / f"eval_{x:02d}"
            out.mkdir(parents=True, exist_ok=True)
            for name, poses in scenes.items():
                render_scene(out, name, poses, p.rig, p.fps, p.size, p.duration)
            print(f"seed{seed} eval_{x:02d}: A={x} B={x+1} C={x+3} cubes", flush=True)
        print(f"seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
