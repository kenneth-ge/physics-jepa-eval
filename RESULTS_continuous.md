# Continuous paradigm results

Metrics (`evals.continuous.measure`, scored on precomputed `saved:` vectors):

- **ρ** — Spearman correlation between the parameter gap |Δθ| and the embedding
  distance, over all clip pairs in a case (cos = cosine distance, l1 = L1).
  +1 = distance grows monotonically with the parameter; 0 = the parameter is
  not readable from distances. No p-value is reported: the pairs are not
  independent (each clip appears in N−1 of them), so a textbook significance
  test would be badly anticonservative. Judge stability from the spread across
  cases instead.
- **nn** — ladder adjacency: fraction of interior clips (θ-sorted) whose two
  nearest embedding neighbors are exactly the clips right before and right
  after them. A local-topology check: stricter than ρ about ordering, and
  indifferent to global scale. Chance level is 1/C(N−1, 2) — **0.010** for a
  16-clip ladder.

## translation — horizontal camera sweep (10 cases, θ = camera x, 16 clips)

`evals.continuous.translation` (job 9055). One static field of 6–10
multicolored rotated cubes per case; the camera trucks x = −1.2…+1.2 in 16
even steps of 0.16 m (orientation, height and distance fixed; stereo rig moves
in lockstep). 120 pairs, 14 interior points per case. Mean over 10 cases:

| model × readout | nn cos | nn l1 | ρ cos | ρ l1 |
|---|---|---|---|---|
| Cosmos raw | **1.00** | **1.00** | +0.96 | +0.98 |
| Qwen3-VL raw | 0.98 | 0.99 | +0.95 | +0.93 |
| V-JEPA2 mean | 0.92 | 0.97 | +0.96 | +0.94 |
| V-JEPA2 raw | 0.71 | 0.75 | +0.96 | +0.96 |
| FastWAM raw | 0.64 | 0.68 | +0.96 | +0.96 |
| FastWAM mean | 0.54 | 0.59 | +0.94 | +0.95 |
| Qwen3-VL mean | 0.37 | 0.41 | +0.70 | +0.70 |
| Cosmos mean | 0.11 | 0.12 | +0.94 | +0.95 |

Per-case ρ and nn are printed by `evals.continuous.measure`; the spread across
the 10 cases is tight (e.g. Cosmos-raw nn = 1.00 in every case, V-JEPA2-raw
nn l1 ranges 0.50–0.86).

**Reading:** ρ is high for nearly everyone, so it barely discriminates here —
a large camera gap means a large pixel gap, making global monotonicity cheap.
Adjacency is the informative metric. Cosmos-raw is perfect (140/140 interior
points, both distances) and Qwen-raw nearly so: their position-aligned raw
readouts embed the sweep as a clean 1-D curve in which a single 0.16 m step is
resolvable. Cosmos-mean is the sharp dissociation — ρ +0.95 but nn 0.12
(against 0.010 chance, so above chance but far from ordered): pooling
preserves the coarse global ordering while destroying the local viewpoint
resolution, leaving each clip's nearest neighbors scattered among nearby
rungs. V-JEPA is the one model whose mean beats its raw (0.97 vs 0.75); its
raw grid tokens shift with the viewpoint, making adjacent-step distances
jumpier than its pooled readout.

**Framing note:** cubes enter and leave the frame toward the sweep extremes
(visible cube area peaks mid-sweep and falls ~1.6× at either end). This is
intended — content leaving view is part of a real camera translation. It gives
no monotone shortcut (correlation of cube area with x is −0.07) and if
anything makes adjacency harder, since frames equidistant from the centre have
similar cube area and thus compete to be each other's nearest neighbor.
A contact sheet of case_00's 16 frames is at
`preview/translation_case_00_grid.png`.
