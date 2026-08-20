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

# --- translation: mean over 10 cases, one row per distance -------------------
TRANSLATION_MEAN = [
    ("nn cos", [0.714, 0.921, 0.979, 1.000, 0.643]),
    ("nn l1",  [0.750, 0.971, 0.986, 1.000, 0.679]),
]

# --- translation per case (L1), 14 interior points each ----------------------
TRANSLATION_CASES = [
    ("case_00", [0.786, 1.000, 0.929, 1.000, 0.786]),
    ("case_01", [0.714, 1.000, 1.000, 1.000, 0.500]),
    ("case_02", [0.786, 1.000, 1.000, 1.000, 0.857]),
    ("case_03", [0.857, 0.929, 1.000, 1.000, 0.571]),
    ("case_04", [0.643, 1.000, 1.000, 1.000, 0.857]),
    ("case_05", [0.786, 0.929, 1.000, 1.000, 0.429]),
    ("case_06", [0.500, 1.000, 1.000, 1.000, 0.857]),
    ("case_07", [0.857, 1.000, 1.000, 1.000, 0.500]),
    ("case_08", [0.786, 1.000, 0.929, 1.000, 0.786]),
    ("case_09", [0.786, 0.857, 1.000, 1.000, 0.643]),
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
    "`evals.continuous.translation` (job 9055). 10 cases; each is one static "
    "field of 6–10 multicoloured rotated cubes, with the camera trucked "
    "x = −1.2…+1.2 in 16 even steps of 0.16 m (orientation, height and "
    "distance fixed; stereo rig moves in lockstep). 14 interior points per "
    "case, 140 across the family.\n",

    "### Mean over 10 cases\n",
    table(["distance"] + MODELS, TRANSLATION_MEAN) + "\n",
    "Cosine and L1 agree closely throughout, so the ranking is not an "
    "artifact of the distance choice.\n",

    "### Per case (L1)\n",
    table(["case"] + MODELS, TRANSLATION_CASES) + "\n",

    "**Reading:** Cosmos-raw is perfect — 140/140 interior points under both "
    "distances — and Qwen-raw nearly so (0.99): they embed the sweep as a "
    "clean 1-D curve in which a single 0.16 m camera step is resolvable. "
    "FastWAM-raw sits mid-table at 0.68, expected since it sees only the "
    "final frame and so has the least to work with. V-JEPA2 is the model that "
    "justifies keeping a pooled readout: its mean (0.97) clearly beats its "
    "raw (0.75), because its raw grid tokens shift with the viewpoint and "
    "make adjacent-step distances jumpier than the pooled vector does.\n",
    "The other models' mean readouts are not run for continuous evals. They "
    "were measured once on this family and pooling cost them heavily on this "
    "axis (Cosmos 1.00 raw → 0.12 mean, Qwen 0.99 → 0.41), which is the "
    "expected behaviour for a spatial parameter: mean-pooling discards the "
    "detail that localises viewpoint. Those numbers are in git history.\n",

    "**Framing note:** cubes enter and leave the frame toward the sweep "
    "extremes (visible cube area peaks mid-sweep and falls ~1.6× at either "
    "end). This is intended — content leaving view is part of a real camera "
    "translation. It gives no monotone shortcut (correlation of cube area "
    "with x is −0.07) and if anything makes adjacency harder, since frames "
    "equidistant from the centre have similar cube area and thus compete to "
    "be each other's nearest neighbour. A contact sheet of case_00's 16 "
    "frames is at `preview/translation_case_00_grid.png`.\n",
]

pathlib.Path("RESULTS_continuous.md").write_text("\n".join(doc))
print("wrote RESULTS_continuous.md")
