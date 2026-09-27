"""Task 4 — Maximum Softmax Probability unknownness score (Hendrycks & Gimpel, 2017)."""
import numpy as np


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score_msp(logits):
    p = softmax_np(logits)
    return 1.0 - p.max(axis=1)