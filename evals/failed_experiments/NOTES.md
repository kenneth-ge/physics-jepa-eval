# Failed experiments

Evals that were built and run but retired from the active suite. Kept for
reference (still runnable) so we don't re-derive the same dead end.

## pendulum (retired 2026-08-20)

Mass-sensitivity probe via damped-swing dynamics. A ball on a rigid rod swings
under gravity with fixed joint damping:
- **A**: mass *m*, released at the left extreme, half-swing, ends on the right.
- **B**: same mass *m* (B = A + one extra half-swing of history), ends on the right.
- **C**: mass *factor·m*, released at the left extreme, half-swing, ends on the right.

Sweep `factor = m_C / m` over 0.5x–2x. The ball's visual size is held constant
(only density changes) and an ideal pendulum's period is mass-independent, so the
only cue is the slightly different damped amplitude a heavier ball retains
(damping ratio ∝ 1/mass).

**Result** (job 9469, 8 seeds × 8 factors, all 4 models × raw/mean): at/below
chance at every factor for every model. The amplitude difference (a few degrees
of swing) is too subtle to survive in any readout window.

**Manual human review concluded it wasn't discriminative enough**, so it was
removed from the active contrastive suite.

Files kept here and still runnable:
`python -m evals.failed_experiments.render_pendulum --out-root <dir> --seeds 8`
(imports resolve unchanged — same depth under `evals/` as `contrastive/`).
