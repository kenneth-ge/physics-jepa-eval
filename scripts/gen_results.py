"""Generate RESULTS_contrastive.md with colour-shaded HTML tables (pass-rate heatmap).

Cells carry the L1 pass rate; background is a diverging scale centred on chance
(~0.33): red = at/below chance (bad), pale-yellow ~chance, green = high (good).
Colour is never the sole signal (the number is in every cell). HTML tables
render in the VS Code markdown preview.
"""

import pathlib

MODELS8 = ["VJEPA flat", "VJEPA mean", "Qwen flat", "Qwen mean",
           "Cosmos flat", "Cosmos mean", "FastWAM flat", "FastWAM mean"]

# --- fixed-scene families (L1 pass rate; 4 models x flat/mean) ---------------
FIXED = [
    ("bowl (4)",            [1.00, 1.00, 1.00, 1.00, 0.75, 1.00, 1.00, 1.00]),
    ("cube (30)",           [1.00, 0.87, 1.00, 0.95, 1.00, 1.00, 1.00, 1.00]),
    ("basic_color (30)",    [0.87, 0.83, 0.80, 0.90, 1.00, 1.00, 1.00, 1.00]),
    ("basic_position (30)", [0.97, 1.00, 0.97, 0.70, 0.97, 0.77, 1.00, 1.00]),
    ("roll (30)",           [0.00, 1.00, 0.83, 0.47, 0.97, 1.00, 0.47, 0.50]),
    ("occlusion (30)",      [0.00, 0.00, 0.13, 0.30, 0.83, 0.77, 0.67, 0.67]),
]

# --- basic_counting: disjoint layouts, X+1 vs X (Cosmos/FastWAM measured) ----
COUNTING = [
    (1, 0.88, 0.88, 0.88, 0.88), (2, 0.50, 0.75, 0.50, 0.62),
    (3, 0.12, 0.62, 0.12, 0.25), (4, 0.12, 0.38, 0.25, 0.25),
    (5, 0.00, 0.25, 0.25, 0.12), (6, 0.00, 0.25, 0.00, 0.12),
    (7, 0.00, 0.12, 0.25, 0.12), (8, 0.00, 0.38, 0.12, 0.12),
    (9, 0.00, 0.12, 0.00, 0.12), (10, 0.00, 0.25, 0.12, 0.12),
]

# --- count2: NESTED counting (A subset B subset C; X, X+1, 2X) ---------------
COUNT2 = [
    (2,  [0.25, 0.00, 0.12, 0.00, 0.12, 0.38, 0.38, 0.38]),
    (3,  [0.75, 0.62, 0.75, 0.75, 1.00, 1.00, 0.88, 0.88]),
    (4,  [0.88, 0.62, 1.00, 0.88, 1.00, 1.00, 1.00, 1.00]),
    (5,  [0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (6,  [1.00, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (7,  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (8,  [1.00, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (9,  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
    (10, [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
]

# --- bounce ADJUSTED: rebound in the last second (all 4 models) --------------
BOUNCE_ADJ = [
    (0.50, [0.00, 0.00, 0.00, 0.38, 0.00, 0.62, 0.00, 0.00]),
    (0.70, [0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00]),
    (0.85, [0.00, 0.00, 0.00, 0.12, 0.00, 0.38, 0.00, 0.00]),
    (1.00, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    (1.20, [0.00, 0.00, 0.00, 0.50, 0.00, 0.12, 0.00, 0.00]),
    (1.50, [0.00, 0.00, 0.00, 0.50, 0.00, 0.38, 0.00, 0.00]),
    (1.75, [0.00, 0.00, 0.00, 1.00, 0.00, 1.00, 0.25, 0.25]),
    (2.00, [0.00, 0.00, 0.88, 1.00, 0.00, 1.00, 0.88, 0.88]),
]

# --- bounce VOID: signal only in history; V-JEPA + Qwen only -----------------
BOUNCE_VOID = [
    (0.50, [0.00, 0.00, 0.00, 0.00]), (0.70, [0.00, 0.00, 0.00, 0.00]),
    (0.85, [0.00, 0.00, 0.00, 0.00]), (1.00, [0.00, 0.00, 0.00, 0.00]),
    (1.20, [0.00, 0.00, 0.00, 0.00]), (1.50, [0.00, 0.00, 0.00, 0.00]),
    (1.75, [0.00, 0.00, 0.00, 0.00]), (2.00, [0.00, 0.00, 0.12, 0.12]),
]

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
       "Cells = **L1 pass rate**, shaded diverging around chance (~0.33): "
       "**red = at/below chance**, pale-yellow ~chance, **green = high**; the "
       "value is in every cell. `flat` = raw whole vector, `mean` = mean-pooled.\n"]

doc.append("## Fixed-scene families\n")
doc.append(table(["scenario (N)"] + MODELS8, FIXED) + "\n")

doc.append("## basic_counting — disjoint layouts, sweep over count X\n")
doc.append(table(["X", "Cosmos flat", "Cosmos mean", "FastWAM flat", "FastWAM mean"],
                 [(str(x), list(v)) for x, *v in COUNTING]) + "\n")
doc.append("VJEPA & Qwen: only X=1 passes; X≥2 at/below chance.\n")

doc.append("## count2 — NESTED counting (A⊂B⊂C: X, X+1, 2X), sweep over X\n")
doc.append(table(["X"] + MODELS8, COUNT2) + "\n")

doc.append("## bounce (adjusted) — rebound in the last second, sweep r2/r1 "
           "(r1=0.45; 1.00 = control C≡A)\n")
doc.append(table(["factor"] + MODELS8,
                 [(f"{f:.2f}", v) for f, v in BOUNCE_ADJ]) + "\n")

doc.append("## bounce (void) — signal only in history (V-JEPA + Qwen), sweep r2/r1\n")
doc.append(table(["factor", "VJEPA flat", "VJEPA mean", "Qwen flat", "Qwen mean"],
                 [(f"{f:.2f}", v) for f, v in BOUNCE_VOID]) + "\n")
doc.append("Whole-clip attention does **not** recover a history-only signal at "
           "the last-second readout — at/below chance at every factor.\n")

pathlib.Path("RESULTS_contrastive.md").write_text("\n".join(doc))
print("wrote RESULTS_contrastive.md")
