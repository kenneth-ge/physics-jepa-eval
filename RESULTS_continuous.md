# Continuous paradigm results

Metrics (`evals.continuous.measure`, on precomputed `saved:` vectors):
- **ρ** — Spearman between parameter gap |Δθ| and embedding distance over all
  clip pairs (cos = cosine distance, l1 = L1). +1 = distance tracks the
  parameter monotonically; 0 = parameter not readable from distances.
- **nn** — ladder adjacency: fraction of interior clips (θ-sorted) whose two
  nearest embedding neighbors are exactly the clips right before and right
  after them on the ladder. Local-topology check; stricter than ρ about
  ordering, indifferent to global scale.

## translation — horizontal camera sweep (10 cases, θ = camera x, 16 points)

Purpose-built family (`evals.continuous.translation`, job 9055 / kenny-translation):
one static field of 6–10 multicolored rotated cubes per case, camera trucked
x = −1.2…+1.2 (step 0.16 m), stereo rig in lockstep. Mean over 10 cases:

| model × readout | nn cos | nn l1 | ρ cos | ρ l1 |
|---|---|---|---|---|
| Cosmos raw | **1.00** | **1.00** | +0.96 | +0.98 |
| Qwen3-VL raw | **0.98** | **0.99** | +0.95 | +0.92 |
| V-JEPA2 mean | 0.92 | 0.97 | +0.96 | +0.94 |
| V-JEPA2 raw | 0.71 | 0.75 | +0.96 | +0.95 |
| FastWAM raw | 0.64 | 0.68 | +0.95 | +0.95 |
| FastWAM mean | 0.54 | 0.59 | +0.94 | +0.94 |
| Qwen3-VL mean | 0.37 | 0.41 | +0.70 | +0.70 |
| Cosmos mean | **0.11** | **0.12** | +0.94 | +0.95 |

**Reading:** ρ is high for nearly everyone — big viewpoint gaps produce big
pixel/embedding gaps, so global monotonicity is cheap here. The adjacency
metric is the discriminative one. Cosmos-raw is perfect (140/140 interior
points, both distances) and Qwen-raw nearly so: their position-aligned raw
readouts embed the sweep as a clean 1-D curve where a 0.16 m camera step is
resolvable. Cosmos-mean is the headline dissociation: ρ +0.95 but nn ≈ 0.11 —
mean-pooling keeps the coarse ordering yet destroys local viewpoint
resolution, so a clip's nearest neighbors are effectively shuffled among
nearby rungs. This is the continuous-paradigm signature of the contrastive
`basic_position` finding (raw passes, mean fails: Qwen-mean 21/30,
Cosmos-mean 22/23): translation is a position axis, and it inverts the bounce
ladder (a magnitude axis) where mean ≫ raw. V-JEPA is the exception whose
mean beats its raw (0.97 vs 0.75 l1) — its raw grid tokens shift with the
viewpoint, making adjacent-step distances jumpier than its pooled readout.

## bounce — restitution ladder (8 seeds, θ = restitution 0.225–0.90 + ref 0.45)

Adapted from the contrastive bounce-adjusted sweep (job 8913, still-prefix
design) via `evals.continuous.from_bounce`: per seed, the 7 non-control C
clips (r2 = factor·r1) + one A (r1 ref) form an 8-clip ladder; factor 1.00
control excluded. 28 pairs per case. Mean ρ across seeds:

| model × readout | ρ cos | ρ l1 |
|---|---|---|
| V-JEPA2 raw | +0.34 | +0.33 |
| V-JEPA2 mean | +0.22 | +0.20 |
| Qwen3-VL raw | +0.63 | +0.66 |
| Qwen3-VL mean | **+0.84** | **+0.84** |
| Cosmos raw | +0.36 | +0.50 |
| Cosmos mean | **+0.82** | **+0.86** |
| FastWAM raw | +0.42 | +0.42 |
| FastWAM mean | +0.39 | +0.39 |

Per-seed ρ (l1):

| seed | vjepa2_raw | vjepa2_mean | qwen_raw | qwen_mean | cosmos_raw | cosmos_mean | fastwam_raw | fastwam_mean |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.28 | +0.12 | +0.64 | +0.85 | +0.54 | +0.82 | +0.45 | +0.39 |
| 1 | +0.41 | +0.30 | +0.67 | +0.75 | +0.50 | +0.83 | +0.43 | +0.35 |
| 2 | +0.41 | +0.22 | +0.71 | +0.87 | +0.52 | +0.88 | +0.50 | +0.46 |
| 3 | +0.29 | +0.33 | +0.66 | +0.83 | +0.52 | +0.85 | +0.36 | +0.30 |
| 4 | +0.36 | +0.08 | +0.65 | +0.90 | +0.42 | +0.90 | +0.40 | +0.41 |
| 5 | +0.29 | +0.12 | +0.62 | +0.82 | +0.46 | +0.89 | +0.45 | +0.44 |
| 6 | +0.24 | +0.12 | +0.74 | +0.79 | +0.53 | +0.88 | +0.43 | +0.47 |
| 7 | +0.38 | +0.27 | +0.59 | +0.89 | +0.48 | +0.84 | +0.34 | +0.34 |

**Reading:** Qwen-mean and Cosmos-mean carry a strong continuous restitution
signal (ρ ≈ +0.84); their raw readouts are markedly weaker — restitution here
is a magnitude cue (rebound height in the last-second window), which
mean-pooling keeps and position-aligned raw dilutes. FastWAM sits at ρ ≈ +0.4
both readouts; V-JEPA2 is weakest (mean ρ ≈ +0.2), consistent with its late
contrastive breakpoint on the bounce sweep. The continuous eval grades what
the contrastive one thresholds: instead of "at which factor does the triplet
flip", ρ measures how faithfully the whole restitution axis is embedded.
