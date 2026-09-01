"""Generate RESULTS_contrastive.md with colour-shaded HTML tables (pass-rate heatmap).

Cells carry the invariant pass rate on the FLAT (raw) latent — one cosine and
one L1 column per model. Background is a diverging scale centred on chance
(~0.33): red = at/below chance (bad), pale-yellow ~chance, green = high (good).
Colour is never the sole signal (the number is in every cell). HTML tables
render in the VS Code markdown preview.

All values recomputed 2026-08-30 from the saved `__<tag>.npy` vectors under
/data/videos on kenny-dev (raw readout only). Mean-pooled columns were dropped
from the doc at the same time; the flat/mean L1 tables live in git history.
Legacy exceptions (V-JEPA ran live before vectors were saved): bowl row uses
the recorded 4/4-all-readouts result; counting_sweep V-JEPA/Qwen stay a text
note (only X=1 passes).
"""

import pathlib

# 4 models x (cos, L1), flat/raw readout only; VJEPA also gets its NEXT-STEP
# readout (vjepa2_next_raw, job 16521): the whole clip is context and the
# predictor forecasts the next temporal token (~0.1s) past its end. This
# replaces the older vjepa2_pred_raw (which masked the observed last second,
# hiding the action in exactly the families designed to put it there).
COLS8 = ["VJEPA cos", "VJEPA L1", "VJEPA next cos", "VJEPA next L1",
         "Qwen cos", "Qwen L1",
         "Cosmos cos", "Cosmos L1", "FastWAM cos", "FastWAM L1"]

# --- fixed-scene families (pass rate; per model: cos, l1) --------------------
FIXED = [
    ("bowl (4)",                [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.75, 0.75, 1.00, 1.00]),
    ("cube v2 (30)",            [1.00, 1.00, 0.97, 0.97, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    ("basic_color v2 (30)",     [0.93, 0.93, 1.00, 0.97, 0.97, 1.00, 1.00, 1.00, 1.00, 1.00]),
    ("basic_position (30)",     [0.97, 0.97, 1.00, 1.00, 0.97, 0.97, 1.00, 0.97, 1.00, 1.00]),
    ("roll (30)",               [0.00, 0.00, 1.00, 1.00, 0.77, 0.83, 0.93, 0.97, 0.40, 0.43]),
    ("occlusion v2 (30)",       [0.00, 0.00, 0.00, 0.00, 0.07, 0.03, 0.00, 0.00, 0.00, 0.00]),
]

# --- basic_counting: disjoint layouts, X+1 vs X (Cosmos/FastWAM measured) ----
COUNTING = [
    (1, [1.00, 0.88, 0.88, 0.88]), (2, [0.50, 0.50, 0.50, 0.50]),
    (3, [0.12, 0.12, 0.25, 0.12]), (4, [0.25, 0.12, 0.25, 0.25]),
    (5, [0.00, 0.00, 0.25, 0.25]), (6, [0.00, 0.00, 0.00, 0.00]),
    (7, [0.00, 0.00, 0.25, 0.25]), (8, [0.00, 0.00, 0.12, 0.12]),
    (9, [0.00, 0.00, 0.00, 0.00]), (10, [0.00, 0.00, 0.12, 0.12]),
]

# --- count2: NESTED counting (A subset B subset C; X, X+1, 2X) ---------------
COUNT2 = [
    (2,  [0.25, 0.25, 0.00, 0.00, 0.12, 0.12, 0.12, 0.12, 0.38, 0.38]),
    (3,  [0.75, 0.75, 0.50, 0.62, 0.50, 0.75, 0.88, 1.00, 1.00, 0.88]),
    (4,  [0.88, 0.88, 0.88, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (5,  [0.88, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (6,  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (7,  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (8,  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (9,  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (10, [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
]

# --- count3: NESTED counting, constant gaps (A⊂B⊂C; X, X+1, X+3; job 14914) --
COUNT3 = [
    (2,  [0.75, 0.75, 1.00, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (3,  [0.75, 0.75, 0.62, 0.50, 0.75, 0.88, 1.00, 1.00, 1.00, 1.00]),
    (4,  [0.88, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (5,  [0.88, 0.88, 0.75, 0.62, 0.75, 0.88, 0.88, 1.00, 0.50, 0.75]),
    (6,  [1.00, 1.00, 0.75, 0.88, 0.88, 1.00, 0.88, 0.88, 1.00, 1.00]),
    (7,  [0.75, 0.75, 0.62, 0.62, 0.75, 0.88, 1.00, 0.88, 0.88, 1.00]),
    (8,  [1.00, 1.00, 0.75, 0.75, 0.75, 0.75, 1.00, 1.00, 0.88, 0.75]),
    (9,  [0.75, 0.75, 0.88, 0.88, 0.75, 0.88, 1.00, 1.00, 1.00, 1.00]),
    (10, [0.75, 0.62, 0.38, 0.38, 0.75, 0.88, 0.88, 0.88, 1.00, 1.00]),
]

# --- bounce ADJUSTED: rebound in last second; A,B share drop point, B = A + a
# 0.5s still prefix (no position confound); all 4 models (job 8913) -----------
BOUNCE_ADJ = [
    (0.50, [0.00, 0.00, 0.25, 0.12, 0.25, 0.50, 0.00, 0.00]),
    (0.70, [0.00, 0.00, 0.12, 0.00, 0.25, 0.38, 0.00, 0.00]),
    (0.85, [0.00, 0.00, 0.25, 0.12, 0.25, 0.25, 0.00, 0.00]),
    (1.00, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.20, [0.00, 0.00, 0.88, 0.88, 0.75, 0.50, 1.00, 1.00]),
    (1.50, [0.00, 0.00, 0.88, 1.00, 0.62, 0.62, 1.00, 1.00]),
    (1.75, [0.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (2.00, [0.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
]

# --- deform ADJUSTED: cube still squashed in the last second; sweep k2/k1 -----
# (deform HISTORY — same end image — is uniformly at/below chance, like void.)
DEFORM_ADJ = [
    (0.50, [0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00]),
    (0.70, [0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00]),
    (0.85, [0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 0.88, 0.88]),
    (1.00, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.20, [0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00]),
    (1.50, [0.00, 0.00, 0.00, 0.00, 0.88, 0.88, 0.88, 0.88]),
    (1.75, [0.00, 0.00, 0.00, 0.00, 0.75, 0.88, 0.88, 0.75]),
    (2.00, [0.00, 0.00, 0.00, 0.00, 0.50, 0.50, 0.75, 0.75]),
]

# --- collision: MASS via momentum transfer; matched impact momentum; sweep m2/m
COLLISION = [
    (0.50, [0.00, 0.00, 0.38, 0.25, 1.00, 1.00, 1.00, 1.00]),
    (0.70, [0.00, 0.00, 0.12, 0.00, 0.88, 0.25, 1.00, 1.00]),
    (0.85, [0.00, 0.00, 0.00, 0.00, 0.12, 0.00, 1.00, 1.00]),
    (1.00, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.20, [0.00, 0.00, 0.00, 0.00, 0.12, 0.00, 1.00, 1.00]),
    (1.50, [0.00, 0.00, 0.00, 0.00, 0.62, 0.12, 1.00, 1.00]),
    (1.75, [0.00, 0.00, 0.25, 0.12, 1.00, 0.38, 1.00, 1.00]),
    (2.00, [0.00, 0.00, 0.75, 0.62, 1.00, 0.88, 1.00, 1.00]),
]

# --- bounce VOID: signal only in history. V-JEPA/Qwen on the mono root;
# Cosmos/FastWAM added 2026-08-31 as LEAK CONTROLS on bounce_void_stereo
# (same seeds -> same physics; jobs 15733 / 15729, FastWAM under fixed ctx) ---
BOUNCE_VOID = [
    (0.50, [0.00, 0.00, 0.12, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (0.70, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (0.85, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.00, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.20, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.50, [0.00, 0.00, 0.00, 0.00, 0.00, 0.12, 0.12, 0.00]),
    (1.75, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (2.00, [0.00, 0.00, 0.12, 0.12, 0.00, 0.00, 0.12, 0.12]),
]

# --- deform HISTORY: same end image, signal only in the squash history
# (scored 2026-08-31 on kenny-dev from the saved vectors; all-model table) ----
DEFORM_HIST = [
    (0.50, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.38]),
    (0.70, [0.00, 0.00, 0.00, 0.00, 0.12, 0.00, 0.25, 0.25]),
    (0.85, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.12]),
    (1.00, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.20, [0.00, 0.00, 0.00, 0.00, 0.12, 0.00, 0.12, 0.00]),
    (1.50, [0.00, 0.00, 0.00, 0.00, 0.12, 0.00, 0.38, 0.25]),
    (1.75, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.25]),
    (2.00, [0.00, 0.00, 0.00, 0.00, 0.12, 0.12, 0.25, 0.25]),
]

NEXT_BY_FAMILY = {'BOUNCE_ADJ': [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.12, 0.0), (0.12, 0.12), (0.88, 1.0), (1.0, 1.0)], 'DEFORM_ADJ': [(0.12, 0.12), (0.0, 0.0), (0.12, 0.12), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], 'COLLISION': [(0.12, 0.12), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], 'BOUNCE_VOID': [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], 'DEFORM_HIST': [(0.0, 0.0), (0.25, 0.25), (0.12, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]}


def with_next(rows, family):
    """Insert the VJEPA next-step (cos, l1) columns for a sweep family
    (job 16521) after the two observed-raw columns."""
    nx = NEXT_BY_FAMILY[family]
    return [(f, v[:2] + list(nx[i]) + v[2:]) for i, (f, v) in enumerate(rows)]


CHANCE = 1.0 / 3.0
BAD, MID, GOOD = (215, 48, 39), (255, 255, 191), (26, 152, 80)


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def cell(v):
    r, g, b = (_lerp(BAD, MID, v / CHANCE) if v <= CHANCE
               else _lerp(MID, GOOD, (v - CHANCE) / (1.0 - CHANCE)))
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


doc = ["# Results\n",
       "Invariant pass rates (A,B closer than C, on the last-second embedding). "
       "Latent = the model's **flat (raw) vector** only; per model, one "
       "**cosine** and one **L1** column. Cells shaded diverging around chance "
       "(~0.33): **red = at/below chance**, pale-yellow ~chance, **green = "
       "high**; the value is in every cell. (Mean-pooled readouts dropped "
       "2026-08-30; the old flat/mean L1 tables are in git history.)\n"]

doc.append("## Fixed-scene families\n")
doc.append(table(["scenario (N)"] + COLS8, FIXED) + "\n")
doc.append("cube v2 (job 14797): A,B share BOTH endpoints via different paths "
           "(identical first frame + last second); C = A's exact path shape "
           "translated — position alone separates C. Raw is perfect for all "
           "4 models (the retired mean readouts were the only ones that "
           "cracked: VJEPA 27/30, Qwen 29/30). basic_color v2 (same job): "
           "A/B hue separation fixed at 0.04 of the wheel (~14°, the old "
           "minimum); only V-JEPA drops any. bowl V-JEPA vectors are now "
           "saved (job 15396) — 4/4 confirmed. **VJEPA next** = the JEPA "
           "predictor's forecast of the NEXT temporal token (~0.1s) past the "
           "end of the clip, with the WHOLE clip as context (job 16521). It "
           "is the canonical V-JEPA dynamics readout as of 2026-08-31, "
           "replacing the older pred readout, which masked the observed last "
           "second and so hid the action in exactly the families designed to "
           "put it there (roll: pred 13/30 vs next 30/30). Headline result: "
           "**roll goes 0/30 observed → 30/30 next** — A,B are moving and C "
           "is static but all three END at the same point, so their final "
           "frames are near-identical and no static cue can explain the "
           "separation; the dynamics are in the model, just not in its "
           "observed-state representation. Caveat to state plainly: V-JEPA 2 "
           "pretrains with multi-block masks at temporal mask ratio ≈1.0 "
           "(spatial blocks spanning ALL frames), so it was never trained to "
           "extrapolate along time — both next and pred are out-of-"
           "distribution probes. The invariant is relative and the same OOD "
           "condition applies to A, B and C, so it cancels, but these are "
           "not 'the model as trained'.\n")
doc.append("FastWAM re-encoded from scratch (job 15391) under the corrected "
           "conditioning: a fixed cached T5 prompt embedding replaces the "
           "old all-zeros context, which decoded as a 128-token all-padding "
           "prompt (structurally OOD vs training). Pass rates reproduce the "
           "old numbers almost exactly (roll/occlusion shift ≤2 cases) — the "
           "constant OOD context was benign for the relative invariant. New "
           "video-expert z readouts (fastwam_z_*/zf_*, the world "
           "representation the action head cross-attends into) were measured "
           "but NOT promoted to the table: the action-stream tap matches or "
           "beats z on every physics sweep (bounce 1.50: action 1.00 vs z "
           "0.50; deform 2.00: 0.75 vs 0.12-0.25) — the action head "
           "distills, not discards. EXCEPTION and eval-design flag: "
           "occlusion z_raw = 30/30 from a SINGLE frame, so the occlusion "
           "family has a static end-frame cue (action tap 21/30 was already "
           "above chance) — it is not history-only for pixel-faithful "
           "readouts.\n")
doc.append("**2026-08-31 occlusion leak identified + v2 fix (table above is "
           "still v1).** The static cue is the red plate's rim: it protruded "
           "13mm past the back face, so the angled `right` stereo camera saw "
           "a thin red strip on the cube's edge in EVERY frame — including "
           "the last (~40 px even at a perfect 360°; verified by raw "
           "re-render). Only FastWAM consumes the stereo view, which is why "
           "only its readouts went above chance while the `fixed`-camera "
           "final frames are pixel-identical A-vs-C (raw render; mp4 diffs "
           "are pure codec noise — raw-pixel last-frame invariant on A.mp4 "
           "is 11/30 L1, 2/30 cos, i.e. no mono leak, so Cosmos 0.80 came "
           "through the clip, not a last-frame cue). Secondary v1 bug, also "
           "fixed: clips stopped one frame short of the full turn (A/C ended "
           "~356°, A vs B end poses differed ~1°). occlusion v2: plate inset "
           "to 2mm proud (0.3mm z-fights and deletes the reveal in 4/30 "
           "cases; ≥5mm leaks again; 1–3mm clean), T quantised to whole "
           "frames with endpoint included. Verified across all 30 cases × "
           "{fixed, right}: A/B/C final frames pixel-identical (ANY-diff "
           "threshold), mid-spin reveal ≥481 px. Occlusion rows here and in "
           "the PFM ladder need re-render + re-encode under v2 before being "
           "read as history-only.\n")
doc.append("**2026-08-31 occlusion v2 MEASURED (job 15735; v2 geometry + "
           "all-intra `-g 1` encoding, now the suite-wide default in "
           "evals/common/video.py — H.264 inter prediction otherwise leaves "
           "shared-history codec residue that pixel-faithful readouts can "
           "score on).** The table above shows v2. The leak is gone: "
           "fastwam z_raw fell 30/30 → 0/30, and every promoted readout of "
           "every model is now at ~0 (Qwen 2/30 cos is the only nonzero "
           "cell). Notable casualty: Cosmos fell 0.80 → 0.00 — its v1 score "
           "matched the measured v1 codec-residue pixel invariant on the "
           "fixed cam (23/30) almost exactly, so it was reading compression "
           "residue, not the clip. occlusion v2 is therefore a clean "
           "negative for all four models. Scope nuance: the red plate is "
           "still on camera for the first ~0.3-0.6s of the measured window "
           "(it hides at 270°, ~T/4 before the end), so v2 tests whether a "
           "last-second readout retains EARLY-IN-WINDOW content — which "
           "every public model washes out — not out-of-window recall; "
           "bounce (void) and deform (history) remain the strict "
           "history-only families.\n")

doc.append("## basic_counting — disjoint layouts, sweep over count X\n")
doc.append(table(["X", "Cosmos cos", "Cosmos L1", "FastWAM cos", "FastWAM L1"],
                 [(str(x), list(v)) for x, v in COUNTING]) + "\n")
doc.append("VJEPA now measured (job 15396, raw cos/L1): 0.75 at X=1 falling "
           "to ≈0–0.25 by X≥4 — same cardinality blind spot; next no better. "
           "Qwen (recorded live run): only X=1 passes.\n")

doc.append("## count2 — NESTED counting (A⊂B⊂C: X, X+1, 2X), sweep over X\n")
doc.append(table(["X"] + COLS8, COUNT2) + "\n")

doc.append("## count3 — NESTED counting, constant gaps (A⊂B⊂C: X, X+1, X+3), "
           "sweep over X\n")
doc.append(table(["X"] + COLS8, COUNT3) + "\n")
doc.append("Gaps stay 1 vs 2 cubes at every X (Weber-style: relative difference "
           "shrinks as X grows), unlike count2 where B→C adds X−1 cubes. "
           "Cosmos/FastWAM near ceiling throughout; V-JEPA and Qwen hold "
           "~0.75–0.88 with no clean collapse by X=10, though margins shrink "
           "with X for all but FastWAM (job 14914).\n")

doc.append("## bounce (adjusted) — rebound in the last second; A,B share drop "
           "point (B = A + 0.5s still prefix); sweep r2/r1 (r1=0.45; 1.00 = control C≡A)\n")
doc.append(table(["factor"] + COLS8,
                 [(f"{f:.2f}", v) for f, v in with_next(BOUNCE_ADJ, "BOUNCE_ADJ")]) + "\n")

doc.append("## deform (adjusted) — material stiffness; cube still squashed in the "
           "last second; sweep k2/k1 (1.00 = control C≡A)\n")
doc.append(table(["factor"] + COLS8,
                 [(f"{f:.2f}", v) for f, v in with_next(DEFORM_ADJ, "DEFORM_ADJ")]) + "\n")
doc.append("Last-frame models (Cosmos, FastWAM) read the squashed shape directly; "
           "whole-clip models (V-JEPA, Qwen) wash it out. deform (history) — same "
           "end image, signal only in the squash history — has its own table "
           "below.\n")

doc.append("## deform (history) — same end image, signal only in the squash "
           "history; sweep k2/k1 (1.00 = control C≡A)\n")
doc.append(table(["factor"] + COLS8,
                 [(f"{f:.2f}", v) for f, v in with_next(DEFORM_HIST, "DEFORM_HIST")]) + "\n")
doc.append("Uniformly at/below chance for every model (like bounce void). "
           "FastWAM's 0.38–0.50 cells hover around chance but its mean "
           "normalized L1 margin is negative at every factor (−0.02 to "
           "−0.12) — noise, not signal.\n")

doc.append("## collision — MASS via momentum transfer; matched impact momentum "
           "(A,B) vs much heavier target (C); sweep m2/m (1.00 = control C≡A)\n")
doc.append(table(["factor"] + COLS8,
                 [(f"{f:.2f}", v) for f, v in with_next(COLLISION, "COLLISION")]) + "\n")
doc.append("FastWAM reads the collision outcome cleanly (rebound vs target "
           "displacement in the last frame); Cosmos partially — and cos≫L1 for "
           "Cosmos here (1.00 vs 0.38 at 1.75), the doc's biggest metric "
           "split; whole-clip models (V-JEPA, Qwen) at/below chance except "
           "Qwen at the extremes.\n")

doc.append("## bounce (void) — signal only in history, sweep r2/r1\n")
doc.append(table(["factor"] + COLS8,
                 [(f"{f:.2f}", v) for f, v in with_next(BOUNCE_VOID, "BOUNCE_VOID")]) + "\n")
doc.append("Whole-clip attention does **not** recover a history-only signal at "
           "the last-second readout — at/below chance at every factor. The "
           "VJEPA next-step readout is ALSO at zero — even forecasting "
           "past the clip from the whole clip as context does not separate "
           "different histories. Same for deform_history. This is the "
           "belief-state test: a representation that had inferred the latent "
           "restitution/stiffness would carry it forward even when the "
           "immediate future looks alike; none does. Cosmos and "
           "FastWAM added 2026-08-31 as leak controls (jobs 15733/15729, on "
           "the stereo re-render bounce_void_stereo — same seeds, same "
           "physics; FastWAM under the fixed prompt context): both at/below "
           "chance with negative margins at every factor, so unlike occlusion "
           "v1 this family has NO static end-frame cue.\n")

pathlib.Path("RESULTS_contrastive.md").write_text("\n".join(doc))
print("wrote RESULTS_contrastive.md")
