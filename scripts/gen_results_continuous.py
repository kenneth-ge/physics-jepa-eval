"""Generate RESULTS_continuous.md with colour-shaded HTML tables.

Same visual language as scripts/gen_results.py (which owns the contrastive
doc): red = bad, pale-yellow = middle, green = good, with the value printed in
every cell so colour is never the sole signal. HTML tables render in the VS
Code markdown preview.

One deliberate difference: the contrastive scale diverges around its chance
level (~0.33). Adjacency's chance level is 1/C(N-1,2) = 0.010 at N=16, so
centring on chance would paint almost everything green. These tables ramp
linearly over the full 0..1 range instead, with the midpoint at 0.50 — a red
cell is therefore at or near chance, and only a near-perfect ladder is green.
"""

import pathlib

# Readouts: raw for every model, plus V-JEPA's mean — the one model whose
# mean-pooled readout beats its raw one, so it is worth carrying. The other
# models' mean readouts are not run for continuous evals (they remain
# registered encoders and are still used by the contrastive suite).
MODELS = ["VJEPA raw", "VJEPA mean", "Qwen raw", "Cosmos raw", "FastWAM raw"]

# --- translation: mean over 10 cases, one row per distance (job 9471) --------
TRANSLATION_MEAN = [
    ("nn cos", [0.957, 0.586, 0.707, 1.000, 1.000]),
    ("nn l1",  [0.929, 0.629, 0.714, 1.000, 1.000]),
]

# --- translation per case (L1), 14 interior points each ----------------------
TRANSLATION_CASES = [
    ("case_00", [0.929, 0.571, 0.714, 1.000, 1.000]),
    ("case_01", [1.000, 1.000, 0.714, 1.000, 1.000]),
    ("case_02", [1.000, 0.071, 0.643, 1.000, 1.000]),
    ("case_03", [0.714, 0.429, 0.571, 1.000, 1.000]),
    ("case_04", [0.929, 0.571, 0.643, 1.000, 1.000]),
    ("case_05", [0.929, 0.857, 0.929, 1.000, 1.000]),
    ("case_06", [1.000, 0.500, 0.714, 1.000, 1.000]),
    ("case_07", [0.929, 0.643, 0.714, 1.000, 1.000]),
    ("case_08", [0.929, 0.857, 0.786, 1.000, 1.000]),
    ("case_09", [0.929, 0.786, 0.714, 1.000, 1.000]),
]

# --- rotation: mean over 10 cases, one row per distance (job 9472) -----------
ROTATION_MEAN = [
    ("nn cos", [0.614, 0.121, 0.757, 1.000, 0.900]),
    ("nn l1",  [0.629, 0.171, 0.914, 1.000, 0.907]),
]

# --- rotation per case (L1) --------------------------------------------------
ROTATION_CASES = [
    ("case_00", [0.929, 0.143, 1.000, 1.000, 1.000]),
    ("case_01", [0.857, 0.500, 1.000, 1.000, 0.929]),
    ("case_02", [0.429, 0.143, 0.857, 1.000, 0.857]),
    ("case_03", [0.643, 0.071, 0.643, 1.000, 0.714]),
    ("case_04", [0.429, 0.143, 0.857, 1.000, 0.929]),
    ("case_05", [0.429, 0.000, 0.929, 1.000, 1.000]),
    ("case_06", [0.500, 0.214, 0.929, 1.000, 0.929]),
    ("case_07", [0.643, 0.143, 1.000, 1.000, 1.000]),
    ("case_08", [0.571, 0.071, 0.929, 1.000, 0.714]),
    ("case_09", [0.857, 0.286, 1.000, 1.000, 1.000]),
]

# --- the same family before re-framing, for comparison (job 9055) ------------
# sweep +/-1.2m (0.16m steps) with cubes allowed to leave frame at the extremes
TRANSLATION_V1_MEAN = [
    ("nn l1 (old)", [0.750, 0.971, 0.986, 1.000, 0.679]),
]

CHANCE = 1.0 / 105.0        # 1/C(15,2) for a 16-clip ladder
MID_AT = 0.50               # colour midpoint (see module docstring)
BAD, MID, GOOD = (215, 48, 39), (255, 255, 191), (26, 152, 80)


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def cell(v):
    r, g, b = (_lerp(BAD, MID, v / MID_AT) if v <= MID_AT
               else _lerp(MID, GOOD, (v - MID_AT) / (1.0 - MID_AT)))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "#0b0b0b" if lum > 140 else "#ffffff"
    return (f'<td align="center" style="background:#{r:02x}{g:02x}{b:02x};'
            f'color:{fg}">{v:.2f}</td>')


def table(headers, rows):
    out = ['<table>', '<thead><tr>'
           + ''.join(f'<th align="center">{h}</th>' for h in headers)
           + '</tr></thead>', '<tbody>']
    for label, vals in rows:
        out.append('<tr><td><b>' + str(label) + '</b></td>'
                   + ''.join(cell(v) for v in vals) + '</tr>')
    out += ['</tbody>', '</table>']
    return "\n".join(out)


doc = [
    "# Continuous paradigm results\n",
    "Metric: **ladder adjacency (nn)** — sort a case's clips by the swept "
    "parameter θ; for every interior clip, its two nearest neighbours in "
    "embedding space should be exactly the clips right before and right after "
    "it. Cells are the fraction of interior clips where that holds, in cosine "
    "distance and L1. Scored by `evals.continuous.measure` on precomputed "
    "`saved:` vectors. `raw` = unreduced vector (the contrastive doc calls "
    "this `flat`); `mean` = mean-pooled, carried for V-JEPA only, since it is "
    "the one model whose pooled readout beats its raw one.\n",
    f"**Chance level is {CHANCE:.3f}** (1/C(N−1,2) for a 16-clip ladder), so "
    "shading ramps over the full 0–1 range — **red = at or near chance**, "
    "pale-yellow = 0.50, **green = near-perfect local ordering** — rather than "
    "diverging around chance as the contrastive tables do. The value is "
    "printed in every cell.\n",
    "Adjacency is a local test: it asks whether the encoder resolves a single "
    "step along the parameter, not merely whether far-apart clips look far "
    "apart.\n",

    "## translation — horizontal camera sweep\n",
    "`evals.continuous.translation` (job 9471). 10 cases; each is one static "
    "field of 6–8 multicoloured rotated cubes, with the camera trucked "
    "x = −0.55…+0.55 in 16 even steps of 0.073 m (orientation, height and "
    "distance fixed; stereo rig moves in lockstep). Every cube stays fully "
    "framed at every sweep point. 14 interior points per case, 140 across the "
    "family.\n",

    "### Mean over 10 cases\n",
    table(["distance"] + MODELS, TRANSLATION_MEAN) + "\n",
    "Cosine and L1 agree closely throughout, so the ranking is not an "
    "artifact of the distance choice.\n",

    "### Per case (L1)\n",
    table(["case"] + MODELS, TRANSLATION_CASES) + "\n",

    "**Reading:** Cosmos-raw and FastWAM-raw are both perfect (140/140 "
    "interior points, both distances) and V-JEPA2-raw is close behind at "
    "0.93. Qwen-raw resolves only about 5 steps in 7 (0.71), and V-JEPA2's "
    "pooled readout is the weakest at 0.63 — with one case (case_02) "
    "collapsing to 0.07, i.e. essentially no local ordering at all.\n",

    "### Effect of the re-framing\n",
    "This family originally swept ±1.2 m in 0.16 m steps and let cubes leave "
    "the frame near the extremes; it now sweeps ±0.55 m in 0.073 m steps with "
    "every cube framed throughout. Same 10 seeds, same metric — mean nn l1 "
    "before:\n",
    table(["run"] + MODELS, TRANSLATION_V1_MEAN) + "\n",
    "The step is less than half the size, so the naive expectation was that "
    "every score would fall. Three rose instead: FastWAM-raw 0.68 → 1.00, "
    "V-JEPA2-raw 0.75 → 0.93, Cosmos-raw held at 1.00. Two fell: Qwen-raw "
    "0.99 → 0.71 and V-JEPA2-mean 0.97 → 0.63.\n",
    "The likely reason the easier-looking version scored worse for some "
    "models is that cubes leaving frame was itself the confound: visible cube "
    "area then peaked mid-sweep and fell at both ends, so frames equidistant "
    "from the centre resembled each other and competed to be one another's "
    "nearest neighbour. Removing that leaves a clean 1-D viewpoint manifold, "
    "which helps any model with fine spatial resolution — and FastWAM, which "
    "sees only the final frame, benefits most. What it costs is coarse-cue "
    "headroom: Qwen-raw and the pooled V-JEPA readout could track 0.16 m "
    "shifts but cannot reliably resolve 0.073 m, so the re-framed family "
    "measures spatial precision where the old one partly measured how much "
    "of the scene was on screen.\n",
    "Two models now saturate at 1.00, so this axis no longer separates the "
    "top of the field; distinguishing Cosmos-raw from FastWAM-raw would need "
    "a finer step or a harder scene.\n",
    "Mean readouts other than V-JEPA's are not run for continuous evals. "
    "Measured once on the original geometry, pooling cost them heavily on "
    "this axis (Cosmos 1.00 raw → 0.12 mean, Qwen 0.99 → 0.41), as expected "
    "for a spatial parameter. Those numbers are in git history.\n",

    "## rotation — 180° camera orbit\n",
    "`evals.continuous.rotation` (job 9472). Same kind of static cube field "
    "(6–10 cubes in a disc of radius 0.70), but the camera orbits a "
    "semicircle at fixed radius 2.8, height 1.1 and look-at target: azimuth "
    "−90…+90° in 16 steps of 12°, with azimuth 0 reproducing the shared front "
    "camera. Only the viewing angle changes, so the field stays centred while "
    "cubes swap depth order, occlude one another differently and turn "
    "different faces to the camera. An infinite floor plane is used so the "
    "finite floor's edge does not sweep the frame as an azimuth cue.\n",

    "### Mean over 10 cases\n",
    table(["distance"] + MODELS, ROTATION_MEAN) + "\n",
    "### Per case (L1)\n",
    table(["case"] + MODELS, ROTATION_CASES) + "\n",

    "**Reading:** Cosmos-raw is again perfect, and the ordering broadly "
    "matches translation — but the two families separate the middle of the "
    "field differently. Qwen-raw *improves* on rotation (0.91 vs 0.71 on "
    "translation) while V-JEPA2-raw *degrades* (0.63 vs 0.93), so the two "
    "swap places: whatever Qwen encodes tracks angular viewpoint better than "
    "fine lateral position, and V-JEPA's raw grid the reverse. FastWAM-raw "
    "drops slightly off its translation ceiling (0.91 vs 1.00), which is "
    "consistent with a single-frame spatial readout meeting a transformation "
    "that changes occlusion and visible faces rather than merely shifting "
    "content.\n",
    "V-JEPA2-mean is the clear failure: 0.17, barely above the 0.010 chance "
    "level, with three cases at or near zero. Pooling costs more here than on "
    "translation (0.63), which fits — a rotation changes which cubes are "
    "visible and how they overlap, and a mean-pooled vector keeps almost none "
    "of the spatial arrangement needed to order those views.\n",
    "Cosine and L1 diverge more on this family than on translation, most "
    "visibly for Qwen-raw (0.76 cos vs 0.91 l1). Where the two disagree, the "
    "ranking should be treated as softer than the single numbers suggest.\n",

    "## Framing and contact sheets\n",
    "Both families keep every cube fully in frame at every point on the "
    "ladder. Each family's `--check` contract proves it numerically (the "
    "cube's bounding corner must clear the frame edge by ≥0.04 world units at "
    "every camera position), and both renders were verified pixelwise: 0 of "
    "160 frames per family have a cube touching an edge. On translation, "
    "visible cube area now varies only 1.08× across a sweep, down from 1.62× "
    "when cubes were allowed to leave view.\n",
    "Every case dir carries a `grid.png` contact sheet — 2×8, one still per "
    "ladder point in order, labelled with its parameter value — written by "
    "the render step. Copies of two are at "
    "`preview/translation_case_00_grid.png` and "
    "`preview/rotation_case_02_grid.png`.\n",
]

pathlib.Path("RESULTS_continuous.md").write_text("\n".join(doc))
print("wrote RESULTS_continuous.md")
