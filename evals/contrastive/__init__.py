"""Contrastive paradigm: A/B/C triplets, ordinal invariant d(A,B) < d(A,C), d(B,C).

Families render A/B/C.mp4 per case; `measure` tests the invariant per encoder
(cosine + L1 on the last-second embedding); `aggregate_*` roll up sweeps.
"""
