"""Task 4 — Maximum Logit Score unknownness score (Vaze et al., 2022)."""
import numpy as np


def score_mls(logits):
    return -logits.max(axis=1)