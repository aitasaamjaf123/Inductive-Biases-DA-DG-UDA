"""Task 4 — Energy-based unknownness score (Liu et al., 2020)."""
import numpy as np


def score_energy(logits):
    m = logits.max(axis=1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1))
    return -lse