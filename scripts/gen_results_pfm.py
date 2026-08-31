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
    ("count3 (pass X=4–10, raw; job 15324)", 1.0,
     [0.52, 0.60, 0.88, 0.54, 0.86, 0.50], "r",
     "separates; middle≫best — best COLLAPSES at X≥8 (0.25) despite huge "
     "low-X margins"),
    ("cube v1 (of 30, raw; job 9490)", 30.0,
     [23, 21, 30, 27, 30, 27], "n",
     "separates; middle>best (v1 design: A/B different starts, C an "
     "independent path — videos since replaced by v2 on the volume)"),
    ("cube v2 (of 30, raw; job 15324)", 30.0,
     [29, 27, 30, 29, 30, 30], "n",
     "SATURATED — v2 (shared endpoints, C = A's path translated) is too easy "
     "for the ladder; weak rungs failed v1 on the path/start variation, not "
     "on position"),
    ("basic_color v2 (of 30, raw; job 15324)", 30.0,
     [22, 23, 27, 30, 30, 29], "n",
     "<b>monotone worst&lt;low&lt;middle&lt;best</b> — at the fixed 0.04 hue "
     "gap, best finally outranks middle (v1: 15→26, middle>best)"),
    ("occlusion v1 (of 30, mean; leaky design)", 30.0,
     [16, 27, 28, 29, 16, 13], "n",
     "was the monotone showcase — but v1 had a stereo rim leak + codec "
     "residue (see RESULTS_contrastive.md), so this ordering is tainted"),
    ("occlusion v2 (of 30, mean cos; job 15735)", 30.0,
     [16, 30, 19, 23, 4, 24], "n",
     "v1 monotonicity DEAD under the fixed design: low at ceiling (30/30, "
     "raw too: 29-30), middle/best drop to 19/23, F16 0-4 — the only "
     "encoder to retain the early-in-window reveal (red visible until ~T/4 "
     "before the end), and training past step 1000 LOSES it"),
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

# Superseded rows are kept in the table for history but excluded from means.
_SUPERSEDED = ("cube v1", "occlusion v1")
# The user-designated canonical set (2026-08-31, see RESULTS.md), matched by
# row-label prefix; latest version of each family.
_CANONICAL = ("translation", "rotation", "velocity", "count3", "cube v2",
              "basic_color v2", "occlusion v2", "basic_position", "roll",
              "bowl", "bounce_void", "collision", "deform_history")


def _avg_row(label, keep, verdict):
    rows = [r for r in ROWS if keep(r[0])]
    vals = [round(sum(r[2][i] / r[1] for r in rows) / len(rows), 3)
            for i in range(6)]
    return (f"{label} [{len(rows)} rows]", 1.0, vals, "r", verdict)


ROWS.append(_avg_row(
    "AVERAGE — all current tasks",
    lambda l: not l.startswith(_SUPERSEDED),
    "mean of each row normalised to its own scale; superseded v1 rows "
    "excluded"))
ROWS.append(_avg_row(
    "AVERAGE — canonical set",
    lambda l: l.startswith(_CANONICAL),
    "same, over the canonical families only (RESULTS.md 2026-08-31)"))

RUNGS = ("worst", "low", "middle", "best")


def _rank_stats(rows):
    """Mean rank per F32 rung (1 = best, fractional ties) + pairwise wins."""
    norm = [[r[2][i] / r[1] for i in range(4)] for r in rows]
    def ranks(v):
        return [1 + sum(o > x for o in v) + (sum(o == x for o in v) - 1) / 2
                for x in v]
    mean_rank = [sum(ranks(v)[i] for v in norm) / len(norm) for i in range(4)]
    wins = [[(sum(v[i] > v[j] for v in norm), sum(v[i] == v[j] for v in norm))
             for j in range(4)] for i in range(4)]
    return mean_rank, wins


def _rank_section():
    canon = [r for r in ROWS
             if not r[0].startswith(_SUPERSEDED + ("AVERAGE",))
             and r[0].startswith(_CANONICAL)]
    mr, wins = _rank_stats(canon)
    lines = [
        "## Rank-based ladder summary (canonical set)\n",
        "Scale-free companion to the AVERAGE rows (a plain score average "
        "lets a few big-margin families dominate and lets noise families "
        "pad every rung): within each canonical family (RESULTS.md "
        "2026-08-31), rank the four F32 training rungs by row-normalised "
        "score (rank 1 = best; ties share the mean rank), then aggregate. "
        "F16/F4 are frame-count controls, not training rungs, so they sit "
        "out; non-canonical rows are excluded throughout.\n",
        "**Mean rank (1 = best):**\n",
        "| task set | " + " | ".join(RUNGS) + " |",
        "|---|---|---|---|---|",
        f"| canonical set ({len(canon)}) | "
        + " | ".join(f"{v:.2f}" for v in mr) + " |",
        "",
        f"**Pairwise dominance over the {len(canon)} canonical families** "
        "— cell = families where ROW strictly beats COLUMN (ties in "
        "parens):\n",
        "| beats → | " + " | ".join(RUNGS) + " |",
        "|---|---|---|---|---|",
    ]
    for i, name in enumerate(RUNGS):
        cells = ["—" if i == j else f"{w} ({t})"
                 for j, (w, t) in enumerate(wins[i])]
        lines.append(f"| **{name}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Read: training clearly helps early (middle beats worst "
        f"{wins[2][0][0]}–{wins[0][2][0]}), and on the canonical set "
        "middle leads the top of the ladder — mean rank "
        f"{mr[2]:.2f} vs {mr[3]:.2f}, middle over best "
        f"{wins[2][3][0]}–{wins[3][2][0]} with {wins[2][3][1]} tie(s) — "
        "consistent with finding 2 (middle's visual pathway is the "
        "stronger representation).\n")
    return "\n".join(lines)

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
    "**2026-08-29 update (job 15324):** cube and basic_color re-run on their "
    "v2 designs, count3 added (nested X, X+1, X+3 — constant 1-vs-2-cube "
    "gaps). Three reads: (a) cube v2 saturates every rung — controlling path "
    "shape out of A-vs-C made the family too easy to rank checkpoints, so "
    "prefer v1-era rows or harder position axes for ladder work; (b) "
    "basic_color v2 becomes the SECOND perfectly monotone family (after "
    "occlusion-mean), and the only one where best beats middle — the fine "
    "hue axis may probe whatever \"best\" was selected on; (c) count3 "
    "reproduces middle≫best, and best's pass rate collapses at X≥8 while "
    "its margins stay huge at low X.\n",
    "**2026-08-31 update (job 15735): occlusion re-run on v2** (plate rim "
    "inset — the v1 static leak — plus full-turn endpoint and the new "
    "suite-wide all-intra `-g 1` encoding that removes shared-history codec "
    "residue). The v1 monotone occlusion-mean row does NOT survive: under "
    "the clean design `low` (step 1000) is at CEILING (raw 29-30/30, mean "
    "30/30) while middle/best (step 10000) fall to 19/23 mean-cos and 2-5/30 "
    "raw, F16 to 0-4. Since every public model (V-JEPA, Qwen, Cosmos, "
    "FastWAM, incl. the JEPA predictor readout) is at ~0/30 on v2, pfm-low "
    "is the only encoder whose last-second readout retains the IN-WINDOW "
    "red reveal (the plate hides at 270°, i.e. ~T/4 = 0.4-0.7s before the "
    "end, so red is on camera for the first 0.3-0.6s of the measured "
    "window; every public model washes it out) — and continued training "
    "destroys it. Note this is early-in-window retention, NOT recall of "
    "content outside the window (bounce_void/deform_history test that, and "
    "everything including pfm is at chance there). Read with care: it "
    "inverts the ladder, like counting_sweep.\n",
    _rank_section(),
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
    "discriminative family except basic_color v2 (occlusion v1's exception "
    "was retracted with the v2 re-run). middle and best are BOTH "
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
    "4. **occlusion (v2) is the suite's strangest row**: the v1 monotone "
    "showcase was leak/residue-tainted; under the clean v2 design pfm-low "
    "alone sits at ceiling (29-30/30 both readouts) on a family every "
    "public model scores ~0/30 on — early-in-window content (the red "
    "reveal, visible until ~T/4 before the end) retained in a last-second "
    "readout — and the step-10000 checkpoints largely lose it (raw 2-5/30). "
    "Whatever step 1000 keeps about recent visual history, further training "
    "trades away.\n",
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
