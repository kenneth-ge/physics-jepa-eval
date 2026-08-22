"""Generate RESULTS_pfm_ladder.md with colour-shaded HTML tables.

Same visual language as scripts/gen_results.py / gen_results_continuous.py:
red = bad, pale-yellow = middle, green = good, value printed in every cell.
Rows mix metrics (pass counts of 30, pass rates, nn adjacency), so each row
declares its own 0..1 normalisation scale; colour ramps linearly 0..1 with the
midpoint at 0.50, like the continuous doc. The row's best value is bold
(ties jointly).
"""

import pathlib

RUNGS = ["worst", "low", "middle", "best", "F16", "F4"]

# (label, scale, [worst, low, middle, best, f16, f4], fmt, verdict)
#   scale: divide values by this for colour;  fmt: "n" ints, "r" 2-dp rates
ROWS = [
    ("translation (nn adj, raw)", 1.0,
     [0.100, 0.093, 0.614, 0.393, 0.686, 0.071], "r",
     "separates strongly; middle>best"),
    ("rotation (nn adj, raw)", 1.0,
     [0.079, 0.057, 0.300, 0.071, 0.214, 0.021], "r",
     "separates; middle≫best"),
    ("velocity (nn adj, raw)", 1.0,
     [0.064, 0.050, 0.279, 0.179, 0.307, 0.071], "r",
     "separates; middle>best, F16 top — first pure MOTION axis to rank the "
     "ladder (job 9552)"),
    ("count2 (pass X=4–10, raw)", 1.0,
     [0.66, 0.68, 1.00, 0.66, 0.98, 0.61], "r",
     "separates; middle>best"),
    ("cube (of 30, raw)", 30.0,
     [23, 21, 30, 27, 30, 27], "n",
     "separates; middle>best"),
    ("basic_color (of 30, raw)", 30.0,
     [15, 19, 26, 24, 29, 22], "n",
     "separates; middle>best"),
    ("occlusion (of 30, mean)", 30.0,
     [16, 27, 28, 29, 16, 13], "n",
     "<b>monotone worst&lt;low&lt;middle&lt;best</b>"),
    ("basic_position (of 30, raw)", 30.0,
     [21, 23, 26, 23, 25, 20], "n",
     "weak"),
    ("roll (of 30, raw)", 30.0,
     [18, 20, 20, 22, 22, 19], "n",
     "flat — not discriminative"),
    ("bowl (of 4, raw)", 4.0,
     [4, 2, 4, 3, 3, 3], "n",
     "n=4; worst degenerate (all cos=1.0000)"),
    ("bounce (pass @factor 2.0, raw)", 1.0,
     [0.75, 0.25, 0.88, 0.88, 0.75, 0.62], "r",
     "weak/noisy"),
    ("counting_sweep (pass X=3–10, raw; chance 0.33)", 1.0,
     [0.33, 0.25, 0.14, 0.19, 0.24, 0.31], "r",
     "all ≤chance; TRAINED rungs BELOW untrained (anti-signal — training "
     "strengthens the position cue counting punishes)"),
    ("bounce_void (pass @2.0, raw)", 1.0,
     [0.25, 0.62, 0.12, 0.12, 0.25, 0.00], "r",
     "noise, no ladder order (last-second readout can't recover a "
     "history-only signal)"),
    ("collision (pass @2.0, raw)", 1.0,
     [0.12, 0.38, 0.00, 0.38, 0.12, 0.38], "r",
     "noise at every rung, margins all negative (no mass encoding)"),
    ("deform_adjusted (pass @2.0, raw)", 1.0,
     [0.25, 0.38, 0.00, 0.50, 0.00, 0.00], "r",
     "noise"),
    ("deform_history (pass @2.0, raw)", 1.0,
     [0.25, 0.12, 0.00, 0.62, 0.00, 0.38], "r",
     "mostly noise; best is the only rung with positive margins "
     "(+0.18 raw/+0.48 mean @2.0) — weak hint"),
]

MID_AT = 0.50
BAD, MID, GOOD = (215, 48, 39), (255, 255, 191), (26, 152, 80)


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def cell(v, scale, fmt, bold):
    t = max(0.0, min(1.0, v / scale))
    r, g, b = (_lerp(BAD, MID, t / MID_AT) if t <= MID_AT
               else _lerp(MID, GOOD, (t - MID_AT) / (1.0 - MID_AT)))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "#0b0b0b" if lum > 140 else "#ffffff"
    text = f"{v:.0f}" if fmt == "n" else f"{v:.2f}"
    if bold:
        text = f"<b>{text}</b>"
    return (f'<td align="center" style="background:#{r:02x}{g:02x}{b:02x};'
            f'color:{fg}">{text}</td>')


def table(rows):
    out = ['<table>', '<thead><tr><th>family (metric)</th>'
           + ''.join(f'<th align="center">{h}</th>' for h in RUNGS)
           + '<th>ladder verdict</th></tr></thead>', '<tbody>']
    for label, scale, vals, fmt, verdict in rows:
        best = max(vals)
        out.append('<tr><td><b>' + label + '</b></td>'
                   + ''.join(cell(v, scale, fmt, v == best) for v in vals)
                   + f'<td>{verdict}</td></tr>')
    out += ['</tbody>', '</table>']
    return "\n".join(out)


doc = [
    "# PFM latent-eval ladder — do our evals rank checkpoint quality?\n",
    "Job 9490 `kenny-pfm-ladder` (all families) + job 9552 `kenny-velocity` "
    "(velocity row), 2026-08-20, both SUCCEEDED on one H200 each. "
    "Six PFM checkpoints (`/data/pfm/pfm-latent-eval-ladder-checkpoints`, "
    "manifest roles F32 worst/low/middle/best + F16/F4 controls) run over "
    "every eval family via `evals/encoders/pfm.py`: DINOv3 ViT-S/16 tokens "
    "(ungated timm re-host) → reimplemented 48L×384 Llama-style trunk "
    "(strict state-dict load) → **latent_head 32-d readout**, visual-only "
    "sequence. F32 roles read 32 frames, F16→16, F4→4. raw = last-second "
    "8×256×32 grid, mean = 32-d pool.\n",
    "Question inverted from our usual tables: checkpoint order is known by "
    "construction (worst=step20 < low=step1000 < middle=step10000 < "
    "best=step10000, middle/best differing only in visual-pathway weights); "
    "we score whether each EVAL recovers it.\n",
    "## Summary (strongest readout per family; cos column)\n",
    "Cell colour is the value normalised to the row's own scale (of-30 rows "
    "/30, of-4 /4, rates as-is), ramping red (0) → pale-yellow (0.5) → green "
    "(1). **Bold = row best** (ties jointly). Colour compares rungs within a "
    "row and rough levels across rows; the verdict column is the read.\n",
    table(ROWS) + "\n",
    "## Findings\n",
    "1. **The suite does discriminate the ladder.** worst/low sit at the "
    "floor and step-10000 checkpoints clearly above it wherever there is "
    "signal (translation 0.10→0.61, rotation, velocity 0.06→0.28, count2 "
    "0.66→1.00, cube 23→30, basic_color 15→26). The most discriminative "
    "axes: viewpoint-continuity (translation/rotation nn adjacency), motion "
    "(velocity) and nested counting (count2). velocity matters because it is "
    "the first PURE-motion axis to rank the ladder: all 16 rungs share the "
    "final position, so the step-10000 gains there are speed encoding, not "
    "statics.\n",
    "2. **Consistent anomaly: `middle` outranks `best`** on every "
    "discriminative family except occlusion-mean. middle and best are BOTH "
    "step 10000 and differ only in visual-pathway tensors — so per our "
    "latent probe, middle's visual pathway is the stronger representation. "
    "Worth asking Mo/Sambhav what \"best\" was selected on (plausibly "
    "overall train loss incl. action/proprio streams, which our visual-only "
    "probe doesn't see).\n",
    "3. **F16 control ≈ top of the table** (translation 0.69, velocity 0.31, "
    "count2 0.98, basic_color 29/30, cube 30/30) — evidence the model's "
    "native context is "
    "nearer 16 frames, or that the F-axis isn't degrading what our evals "
    "measure. F4 lands near the floor on continuous families (0.07/0.02). "
    "Untangle by re-running f16 at PFM_FRAMES=32 (cheap, vectors only).\n",
    "4. **occlusion splits by readout**: mean is near-perfect and the ONLY "
    "perfectly monotone family (16<27<28<29); raw *collapses* for the "
    "trained checkpoints (middle 7, best 9, f16 1 of 30) — "
    "variable-duration B=2× interacts with position-aligned latents. "
    "Raw-vs-mean dissociation again carries information about what the "
    "trunk encodes.\n",
    "5. **PFM shares the universal failures** of the four public models: "
    "cardinality (counting_sweep X≥3), history-only signals (bounce_void), "
    "mass (collision), stiffness (deform) — at every rung, so these "
    "families can't rank these checkpoints (they may still rank stronger "
    "ladders).\n",
    "## Caveats\n",
    "- Input contract is guessed (DINOv3-S/16 @256, token = in_proj(dino)+"
    "slot+modality0+view0+source0, frame-major RoPE, bidirectional). "
    "Identical assumptions across all six checkpoints → within-ladder "
    "ordering is fair, but absolute levels may understate the model.\n",
    "- bowl's worst \"4/4\" is degenerate: step-20 latents give cos≈1.0000 "
    "on all pairs, passes are coin-flips on ~0 margins. Treat tiny-n bowl "
    "as noise.\n",
    "- F16/F4 read 16/4 frames by assumption (F = context frames). If F "
    "means something else, their columns shift meaning (F32 columns "
    "unaffected).\n",
    "Raw tables: `sky jobs logs 9490`; vectors: "
    "`<clip>__pfm_<role>_{raw,mean}.npy` beside every eval clip under "
    "/data/videos/.\n",
]

pathlib.Path("RESULTS_pfm_ladder.md").write_text("\n".join(doc))
print("wrote RESULTS_pfm_ladder.md")
