# Continuous paradigm results

Metric: **ladder adjacency (nn)** — sort a case's clips by the swept parameter θ; for every interior clip, its two nearest neighbours in embedding space should be exactly the clips right before and right after it. Cells are the fraction of interior clips where that holds, in cosine distance and L1. Scored by `evals.continuous.measure` on precomputed `saved:` vectors. `raw` = unreduced vector (the contrastive doc calls this `flat`); `mean` = mean-pooled, carried for V-JEPA only, since it is the one model whose pooled readout beats its raw one.

**Chance level is 0.010** (1/C(N−1,2) for a 16-clip ladder), so shading ramps over the full 0–1 range — **red = at or near chance**, pale-yellow = 0.50, **green = near-perfect local ordering** — rather than diverging around chance as the contrastive tables do. The value is printed in every cell.

Adjacency is a local test: it asks whether the encoder resolves a single step along the parameter, not merely whether far-apart clips look far apart.

> ⚠️ **Stale — numbers below are from the previous scene geometry** (sweep ±1.2 m, 0.16 m steps, cubes allowed to leave frame). The family was changed so every cube stays framed at every sweep point, which required shrinking the sweep to ±0.55 m (0.073 m steps); job `kenny-translation-v2` is re-rendering and re-scoring. This table will be replaced when it lands.

## translation — horizontal camera sweep

`evals.continuous.translation` (job 9055). 10 cases; each is one static field of 6–10 multicoloured rotated cubes, with the camera trucked x = −1.2…+1.2 in 16 even steps of 0.16 m (orientation, height and distance fixed; stereo rig moves in lockstep). 14 interior points per case, 140 across the family.

### Mean over 10 cases

<table>
<thead><tr><th align="center">distance</th><th align="center">VJEPA raw</th><th align="center">VJEPA mean</th><th align="center">Qwen raw</th><th align="center">Cosmos raw</th><th align="center">FastWAM raw</th></tr></thead>
<tbody>
<tr><td><b>nn cos</b></td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#3ea862;color:#ffffff">0.92</td><td align="center" style="background:#249c55;color:#ffffff">0.98</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#bee29f;color:#0b0b0b">0.64</td></tr>
<tr><td><b>nn l1</b></td><td align="center" style="background:#8ccc88;color:#0b0b0b">0.75</td><td align="center" style="background:#279e56;color:#ffffff">0.97</td><td align="center" style="background:#209b53;color:#ffffff">0.99</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#adda97;color:#0b0b0b">0.68</td></tr>
</tbody>
</table>

Cosine and L1 agree closely throughout, so the ranking is not an artifact of the distance choice.

### Per case (L1)

<table>
<thead><tr><th align="center">case</th><th align="center">VJEPA raw</th><th align="center">VJEPA mean</th><th align="center">Qwen raw</th><th align="center">Cosmos raw</th><th align="center">FastWAM raw</th></tr></thead>
<tbody>
<tr><td><b>case_00</b></td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td></tr>
<tr><td><b>case_01</b></td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#ffffbf;color:#0b0b0b">0.50</td></tr>
<tr><td><b>case_02</b></td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td></tr>
<tr><td><b>case_03</b></td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#def0af;color:#0b0b0b">0.57</td></tr>
<tr><td><b>case_04</b></td><td align="center" style="background:#bee29f;color:#0b0b0b">0.64</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td></tr>
<tr><td><b>case_05</b></td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#f9e2a9;color:#0b0b0b">0.43</td></tr>
<tr><td><b>case_06</b></td><td align="center" style="background:#ffffbf;color:#0b0b0b">0.50</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td></tr>
<tr><td><b>case_07</b></td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#ffffbf;color:#0b0b0b">0.50</td></tr>
<tr><td><b>case_08</b></td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td></tr>
<tr><td><b>case_09</b></td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#bee29f;color:#0b0b0b">0.64</td></tr>
</tbody>
</table>

**Reading:** Cosmos-raw is perfect — 140/140 interior points under both distances — and Qwen-raw nearly so (0.99): they embed the sweep as a clean 1-D curve in which a single 0.16 m camera step is resolvable. FastWAM-raw sits mid-table at 0.68, expected since it sees only the final frame and so has the least to work with. V-JEPA2 is the model that justifies keeping a pooled readout: its mean (0.97) clearly beats its raw (0.75), because its raw grid tokens shift with the viewpoint and make adjacent-step distances jumpier than the pooled vector does.

The other models' mean readouts are not run for continuous evals. They were measured once on this family and pooling cost them heavily on this axis (Cosmos 1.00 raw → 0.12 mean, Qwen 0.99 → 0.41), which is the expected behaviour for a spatial parameter: mean-pooling discards the detail that localises viewpoint. Those numbers are in git history.

**Framing note:** cubes enter and leave the frame toward the sweep extremes (visible cube area peaks mid-sweep and falls ~1.6× at either end). This is intended — content leaving view is part of a real camera translation. It gives no monotone shortcut (correlation of cube area with x is −0.07) and if anything makes adjacency harder, since frames equidistant from the centre have similar cube area and thus compete to be each other's nearest neighbour. Every case dir carries a `grid.png` contact sheet (2×8, one still per sweep point in ladder order, labelled with its camera x), written by the render step; case_00's copy is also at `preview/translation_case_00_grid.png`.
