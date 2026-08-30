"""count3 eval family: nested counting with small constant gaps.

Like count2 (cubes are only ever ADDED, never moved) but with counts
X, X+1, X+3 instead of X, X+1, 2X:

  A: X cubes;
  B: X + 1 cubes (A's cubes plus one more, same positions);
  C: X + 3 cubes (B's cubes plus two more, same positions).

So A ⊂ B ⊂ C with pairwise count gaps AB=1, BC=2, AC=3 — strictly monotone
ground truth, so BOTH legs of the invariant (AB<AC, AB<BC) are
discriminative and a perfect counter passes outright. (X+2 would tie AB and
BC at one cube each — BC becomes a positional coin flip.) Unlike count2,
the gaps don't grow with X, so the sweep measures how 1-vs-2-cube
resolution degrades as the set gets larger (Weber-fraction style).

Renders <root>/seed{S}/eval_{X:02d}/{A,B,C}.mp4 so
`evals.contrastive.aggregate_counting` reads it directly. Scenes are static;
`--rig stereo` also writes the right cam.
"""

import numpy as np

from .basic_counting import sample_poses


def build_scenes(x, seed):
    """Nested cube layouts for count X: A=pool[:X], B=pool[:X+1], C=pool[:X+3]."""
    rng = np.random.default_rng([seed, x, 3])
    pool = sample_poses(x + 3, rng)
    return {"A": pool[:x], "B": pool[:x + 1], "C": pool[:x + 3]}
