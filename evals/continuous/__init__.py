"""Pairwise-continuous paradigm: does embedding distance track physical distance?

Each case renders a ladder of clips at continuous parameter values theta
(restitution, offset, hue, count, ...) plus a manifest.json mapping clip ->
theta. `measure` scores, per encoder, the Spearman correlation between
|theta_i - theta_j| and d(emb_i, emb_j) over all clip pairs — a metric/ordinal
structure test, versus the contrastive paradigm's single triplet inequality.
"""
