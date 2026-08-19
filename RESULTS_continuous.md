# Continuous paradigm results

Metric: Spearman ρ between parameter gap |Δθ| and embedding distance over all
clip pairs in a case (`evals.continuous.measure`; cos = cosine distance,
l1 = L1). +1 = embedding distance tracks the physical parameter monotonically;
0 = parameter not readable from distances. Scored on precomputed `saved:`
vectors — same clips and embeddings as the contrastive tables.

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
