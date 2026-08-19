"""Plot the bounce restitution sweep from the aggregate_bounce log.

Parses the pass-rate and margin tables out of the saved job log and draws two
panels vs the multiplicative factor r2/r1: (left) fraction of seeds where the
invariant holds, (right) mean normalized L1 margin (its zero-crossing is the
breakpoint). Four models, mean-pool readout (raw tracks it); each model gets a
distinct colour AND marker so identity never rests on colour alone.
"""

import re
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = pathlib.Path("/tmp/bounce_res.log")

# mean-pool L1 column index (after the factor col) in the pass-rate table, and
# the mean-margin column index in the margin table, per model.
MODELS = [
    ("V-JEPA2",       "#2a78d6", "o", 3, 1),
    ("Qwen3.6",       "#eb6834", "s", 7, 3),
    ("Cosmos3-Super", "#1baf7a", "^", 11, 5),
    ("FastWAM",       "#4a3aa7", "D", 15, 7),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def parse_rows(lines, start_marker):
    """Rows [(factor, [floats...])] following a section marker line."""
    rows, on = [], False
    for ln in lines:
        if start_marker in ln:
            on = True
            continue
        if on:
            m = re.match(r"\s*(\d\.\d\d)\s+(.*)", ln)
            if m and re.search(r"[+-]?\d", m.group(2)):
                vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+", m.group(2))]
                rows.append((float(m.group(1)), vals))
            elif rows and not ln.strip():
                break
    return rows


lines = LOG.read_text().splitlines()
passr = parse_rows(lines, "pass rate per r2/r1")
marg = parse_rows(lines, "mean normalized L1 margin")
xs = [f for f, _ in passr]

fig, (axp, axm) = plt.subplots(1, 2, figsize=(12, 5.2))
for ax in (axp, axm):
    ax.axvline(1.0, color=GRID, lw=1.5, ls="--", zorder=0)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_xlabel("restitution factor  r2 / r1   (1.00 = control, C ≡ A)",
                  color=MUTED, fontsize=10)

for name, color, mk, ip, im in MODELS:
    yp = [row[ip] for _, row in passr]
    axp.plot(xs, yp, color=color, lw=2, marker=mk, ms=8, label=name,
             markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ym = [row[im] for _, row in marg]
    axm.plot(xs, ym, color=color, lw=2, marker=mk, ms=8, label=name,
             markeredgecolor="white", markeredgewidth=0.8, zorder=3)

axp.set_title("Invariant pass rate (mean-pool, L1)", color=INK, fontsize=13,
              fontweight="bold", loc="left")
axp.set_ylabel("fraction of 8 seeds where A,B < C holds", color=MUTED, fontsize=10)
axp.set_ylim(-0.05, 1.08)

axm.axhline(0.0, color=INK, lw=1.2, zorder=1)          # separation threshold
axm.set_title("Separation margin (mean-pool, L1)", color=INK, fontsize=13,
              fontweight="bold", loc="left")
axm.set_ylabel("mean normalized L1 margin   ( >0 ⇒ invariant separates )",
               color=MUTED, fontsize=10)
axm.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left")

fig.suptitle("bounce sweep — sensitivity to restitution (r2 = factor · r1, r1 = 0.7)",
             color=INK, fontsize=14, fontweight="bold", x=0.5, y=0.99)
fig.text(0.5, 0.005, "Invariant holds only for factor > 1 (a bouncier C, still "
         "airborne in the last second); factor ≤ 1 never separates. "
         "FastWAM (last-frame height) is most sensitive; V-JEPA least.",
         ha="center", color=MUTED, fontsize=9)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out = pathlib.Path(__file__).with_name("bounce_sweep.png")
fig.savefig(out, dpi=150, facecolor="white")
print("wrote", out)
