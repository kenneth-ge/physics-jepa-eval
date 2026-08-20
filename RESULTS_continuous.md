# Continuous paradigm results

Metrics (`evals.continuous.measure`, scored on precomputed `saved:` vectors):

- **ρ** — Spearman correlation between the parameter gap |Δθ| and the embedding
  distance, over all clip pairs in a case (cos = cosine distance, l1 = L1).
  +1 = distance grows monotonically with the parameter; 0 = the parameter is
  not readable from distances. No p-value is reported: the pairs are not
  independent (each clip appears in N−1 of them), so a textbook significance
  test would be badly anticonservative. Judge stability from the spread across
  cases/seeds instead.
- **nn** — ladder adjacency: fraction of interior clips (θ-sorted) whose two
  nearest embedding neighbors are exactly the clips right before and right
  after them. A local-topology check: stricter than ρ about ordering, and
  indifferent to global scale. Chance level is 1/C(N−1, 2) — **0.010** for a
  16-clip ladder, **0.048** for an 8-clip one.

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

## bounce — restitution ladder (8 seeds, θ = restitution 0.225–0.90, 8 clips)

Adapted from the bounce-adjusted sweep (job 8913, still-prefix design) by
`evals.continuous.from_bounce`: per seed the 7 non-control C clips
(r2 = factor·r1) plus one A (r1 = 0.45 reference) form the ladder; the
factor-1.00 control (C ≡ A) is excluded. Unevenly spaced, 28 pairs, only 6
interior points per seed. Mean over 8 seeds:

| model × readout | nn cos | nn l1 | ρ cos | ρ l1 |
|---|---|---|---|---|
| Qwen3-VL mean | **0.42** | **0.42** | +0.83 | +0.83 |
| Cosmos mean | 0.33 | 0.40 | +0.81 | +0.86 |
| Qwen3-VL raw | 0.08 | 0.13 | +0.64 | +0.67 |
| Cosmos raw | 0.04 | 0.10 | +0.37 | +0.51 |
| V-JEPA2 raw | 0.06 | 0.06 | +0.34 | +0.33 |
| V-JEPA2 mean | 0.04 | 0.04 | +0.22 | +0.20 |
| FastWAM raw | 0.02 | 0.04 | +0.43 | +0.44 |
| FastWAM mean | 0.02 | 0.02 | +0.40 | +0.41 |

**Reading:** the readout preference inverts relative to translation — here the
**mean** readouts carry the signal (Qwen-mean and Cosmos-mean at ρ ≈ +0.85,
nn ≈ 0.4 against 0.048 chance, i.e. ~8× chance), while every raw readout sits
at or near chance on adjacency. Restitution is a magnitude cue (rebound height
in the last-second window) that survives pooling, whereas camera x is a
spatial cue that does not. Absolute nn is much lower than on translation for
structural reasons as well as model ones: this ladder has 8 rungs rather than
16, they are unevenly spaced (θ steps range 0.07–0.11), and neighbors are
therefore closer together relative to the noise, so exact 2-NN adjacency is a
harsh test. ρ remains the fairer summary for this family; nn is best read as a
ranking of the models rather than an absolute score.

## Cross-family finding

The two ladders probe orthogonal axes and cleanly invert:

| axis | family | winner |
|---|---|---|
| position / viewpoint | translation | **raw** (Cosmos-raw 1.00 nn) |
| magnitude | bounce restitution | **mean** (Qwen/Cosmos-mean ρ ≈ +0.85) |

No single readout serves both. Mean-pooling discards spatial location but
retains magnitude; the position-aligned raw readouts do the reverse. The
Cosmos pair makes this vivid: the same model is the best on translation
(nn 1.00 raw) and among the best on bounce (mean), through *different*
readouts, and each readout fails on the other family's axis.
