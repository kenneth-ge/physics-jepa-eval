"""Pairwise-continuous paradigm: does embedding distance track physical distance?

Each case renders a ladder of clips at continuous parameter values theta
(camera position, offset, hue, count, ...) plus a manifest.json mapping clip
-> theta. `measure` scores, per encoder, neighbor adjacency: the fraction of
interior clips whose two nearest embedding neighbors are the theta-adjacent
clips — a local-topology test, versus the contrastive paradigm's single
triplet inequality.
"""
