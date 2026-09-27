"""
Task 4 — PROSER placeholder-based unknownness score. Combines the 5 dummy
logits with the 10 known logits per PROSER_SCORE_MODE (see configs/config.py
for the two implemented, defensible readings of the spec's "combine the
strongest dummy response with the known-class responses").
"""
import numpy as np
from scores.msp import softmax_np
from configs.config import PROSER_SCORE_MODE


def score_proser(logits_known, dummy_logits, mode=PROSER_SCORE_MODE):
    known_max = logits_known.max(axis=1)
    dummy_max = dummy_logits.max(axis=1)
    if mode == "dummy_minus_known":
        return dummy_max - known_max
    elif mode == "dummy_softmax":
        combined = np.concatenate([logits_known, dummy_logits], axis=1)
        p = softmax_np(combined)
        return p[:, logits_known.shape[1]:].sum(axis=1)
    else:
        raise ValueError(f"unknown PROSER_SCORE_MODE: {mode}")