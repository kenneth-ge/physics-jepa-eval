# Continuous paradigm results

Metric (`evals.continuous.measure`, scored on precomputed `saved:` vectors):

- **nn** — ladder adjacency: sort a case's clips by the swept parameter θ; for
  every interior clip, its two nearest neighbors in embedding space should be
  exactly the clips right before and right after it. The score is the fraction
  of interior clips where that holds, reported for cosine distance (`nn cos`)
  and L1 (`nn l1`).
- **Chance level** is 1/C(N−1, 2) — **0.010** for a 16-clip ladder — since a
  clip's two nearest neighbors could be any unordered pair of the other N−1.

Adjacency is a local test: it asks whether the encoder resolves a single step
along the parameter, not merely whether far-apart clips look far apart.

## translation — horizontal camera sweep (10 cases, θ = camera x, 16 clips)

`evals.continuous.translation` (job 9055). One static field of 6–10
multicolored rotated cubes per case; the camera trucks x = −1.2…+1.2 in 16
even steps of 0.16 m (orientation, height and distance fixed; stereo rig moves
in lockstep). 14 interior points per case, 140 across the family.

| model × readout | nn cos | nn l1 |
|---|---|---|
| Cosmos raw | **1.00** | **1.00** |
| Qwen3-VL raw | 0.98 | 0.99 |
| V-JEPA2 mean | 0.92 | 0.97 |
| V-JEPA2 raw | 0.71 | 0.75 |
| FastWAM raw | 0.64 | 0.68 |
| FastWAM mean | 0.54 | 0.59 |
| Qwen3-VL mean | 0.37 | 0.41 |
| Cosmos mean | 0.11 | 0.12 |

Cosine and L1 agree closely throughout, so the ranking is not an artifact of
the distance choice. Spread across the 10 cases is tight (Cosmos-raw is 1.00
in every case; V-JEPA2-raw nn l1 ranges 0.50–0.86).

**Reading:** the raw, position-aligned readouts win this axis. Cosmos-raw is
perfect — 140/140 interior points under both distances — and Qwen-raw nearly
so: they embed the sweep as a clean 1-D curve in which a single 0.16 m camera
step is resolvable. Cosmos-mean collapses to 0.11 (barely above the 0.010
chance level): mean-pooling discards the spatial detail that localizes
viewpoint, leaving each clip's nearest neighbors scattered among nearby rungs.
Qwen-mean shows the same pooling penalty less severely (0.41 vs 0.98 raw).
V-JEPA2 is the one model whose mean beats its raw (0.97 vs 0.75); its raw grid
tokens shift with the viewpoint, making adjacent-step distances jumpier than
its pooled readout. FastWAM sits mid-table on both readouts — expected, since
it sees only the final frame and so has the least to work with.

**Framing note:** cubes enter and leave the frame toward the sweep extremes
(visible cube area peaks mid-sweep and falls ~1.6× at either end). This is
intended — content leaving view is part of a real camera translation. It gives
no monotone shortcut (correlation of cube area with x is −0.07) and if
anything makes adjacency harder, since frames equidistant from the centre have
similar cube area and thus compete to be each other's nearest neighbor.
A contact sheet of case_00's 16 frames is at
`preview/translation_case_00_grid.png`.
