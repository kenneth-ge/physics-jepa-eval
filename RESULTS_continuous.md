# Continuous paradigm results

Metric: **ladder adjacency (nn)** — sort a case's clips by the swept parameter θ; for every interior clip, its two nearest neighbours in embedding space should be exactly the clips right before and right after it. Cells are the fraction of interior clips where that holds, in cosine distance and L1. Scored by `evals.continuous.measure` on precomputed `saved:` vectors. `raw` = unreduced vector (the contrastive doc calls this `flat`); `mean` = mean-pooled, carried for V-JEPA only, since it is the one model whose pooled readout beats its raw one.

**Chance level is 0.010** (1/C(N−1,2) for a 16-clip ladder), so shading ramps over the full 0–1 range — **red = at or near chance**, pale-yellow = 0.50, **green = near-perfect local ordering** — rather than diverging around chance as the contrastive tables do. The value is printed in every cell.

Adjacency is a local test: it asks whether the encoder resolves a single step along the parameter, not merely whether far-apart clips look far apart.

## translation — horizontal camera sweep

`evals.continuous.translation` (job 9471). 10 cases; each is one static field of 6–8 multicoloured rotated cubes, with the camera trucked x = −0.55…+0.55 in 16 even steps of 0.073 m (orientation, height and distance fixed; stereo rig moves in lockstep). Every cube stays fully framed at every sweep point. 14 interior points per case, 140 across the family.

### Mean over 10 cases

<table>
<thead><tr><th align="center">distance</th><th align="center">VJEPA raw</th><th align="center">VJEPA mean</th><th align="center">Qwen raw</th><th align="center">Cosmos raw</th><th align="center">FastWAM raw</th></tr></thead>
<tbody>
<tr><td><b>nn cos</b></td><td align="center" style="background:#2ea15a;color:#ffffff">0.96</td><td align="center" style="background:#d8edac;color:#0b0b0b">0.59</td><td align="center" style="background:#a0d491;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>nn l1</b></td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#c4e4a2;color:#0b0b0b">0.63</td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
</tbody>
</table>

Cosine and L1 agree closely throughout, so the ranking is not an artifact of the distance choice.

### Per case (L1)

<table>
<thead><tr><th align="center">case</th><th align="center">VJEPA raw</th><th align="center">VJEPA mean</th><th align="center">Qwen raw</th><th align="center">Cosmos raw</th><th align="center">FastWAM raw</th></tr></thead>
<tbody>
<tr><td><b>case_00</b></td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#def0af;color:#0b0b0b">0.57</td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_01</b></td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_02</b></td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#dd4d3d;color:#ffffff">0.07</td><td align="center" style="background:#bee29f;color:#0b0b0b">0.64</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_03</b></td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#f9e2a9;color:#0b0b0b">0.43</td><td align="center" style="background:#def0af;color:#0b0b0b">0.57</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_04</b></td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#def0af;color:#0b0b0b">0.57</td><td align="center" style="background:#bee29f;color:#0b0b0b">0.64</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_05</b></td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_06</b></td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#ffffbf;color:#0b0b0b">0.50</td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_07</b></td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#bee29f;color:#0b0b0b">0.64</td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_08</b></td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#5bb570;color:#0b0b0b">0.86</td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
<tr><td><b>case_09</b></td><td align="center" style="background:#3ba760;color:#ffffff">0.93</td><td align="center" style="background:#7cc480;color:#0b0b0b">0.79</td><td align="center" style="background:#9dd38f;color:#0b0b0b">0.71</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td></tr>
</tbody>
</table>

**Reading:** Cosmos-raw and FastWAM-raw are both perfect (140/140 interior points, both distances) and V-JEPA2-raw is close behind at 0.93. Qwen-raw resolves only about 5 steps in 7 (0.71), and V-JEPA2's pooled readout is the weakest at 0.63 — with one case (case_02) collapsing to 0.07, i.e. essentially no local ordering at all.

### Effect of the re-framing

This family originally swept ±1.2 m in 0.16 m steps and let cubes leave the frame near the extremes; it now sweeps ±0.55 m in 0.073 m steps with every cube framed throughout. Same 10 seeds, same metric — mean nn l1 before:

<table>
<thead><tr><th align="center">run</th><th align="center">VJEPA raw</th><th align="center">VJEPA mean</th><th align="center">Qwen raw</th><th align="center">Cosmos raw</th><th align="center">FastWAM raw</th></tr></thead>
<tbody>
<tr><td><b>nn l1 (old)</b></td><td align="center" style="background:#8ccc88;color:#0b0b0b">0.75</td><td align="center" style="background:#279e56;color:#ffffff">0.97</td><td align="center" style="background:#209b53;color:#ffffff">0.99</td><td align="center" style="background:#1a9850;color:#ffffff">1.00</td><td align="center" style="background:#adda97;color:#0b0b0b">0.68</td></tr>
</tbody>
</table>

The step is less than half the size, so the naive expectation was that every score would fall. Three rose instead: FastWAM-raw 0.68 → 1.00, V-JEPA2-raw 0.75 → 0.93, Cosmos-raw held at 1.00. Two fell: Qwen-raw 0.99 → 0.71 and V-JEPA2-mean 0.97 → 0.63.

The likely reason the easier-looking version scored worse for some models is that cubes leaving frame was itself the confound: visible cube area then peaked mid-sweep and fell at both ends, so frames equidistant from the centre resembled each other and competed to be one another's nearest neighbour. Removing that leaves a clean 1-D viewpoint manifold, which helps any model with fine spatial resolution — and FastWAM, which sees only the final frame, benefits most. What it costs is coarse-cue headroom: Qwen-raw and the pooled V-JEPA readout could track 0.16 m shifts but cannot reliably resolve 0.073 m, so the re-framed family measures spatial precision where the old one partly measured how much of the scene was on screen.

Two models now saturate at 1.00, so this axis no longer separates the top of the field; distinguishing Cosmos-raw from FastWAM-raw would need a finer step or a harder scene.

Mean readouts other than V-JEPA's are not run for continuous evals. Measured once on the original geometry, pooling cost them heavily on this axis (Cosmos 1.00 raw → 0.12 mean, Qwen 0.99 → 0.41), as expected for a spatial parameter. Those numbers are in git history.

**Framing:** the contract now proves numerically that every cube's bounding corner clears the frame edge at both sweep extremes, and the render was checked pixelwise — 0 of 160 frames have a cube touching an edge, and visible cube area varies only 1.08× across a sweep (it was 1.62× before). Every case dir carries a `grid.png` contact sheet (2×8, one still per sweep point in ladder order, labelled with its camera x), written by the render step; case_00's copy is also at `preview/translation_case_00_grid.png`.
