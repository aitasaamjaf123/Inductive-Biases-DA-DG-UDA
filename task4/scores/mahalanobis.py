"""
Task 4 — Mahalanobis-distance unknownness score. Class means and one shared
diagonal covariance are estimated from unaugmented CIFAR-10 TRAINING
features only, with eps added to every diagonal entry.
"""
import numpy as np

from configs.config import MAHAL_EPS


def fit_mahalanobis(train_feats, train_labels, num_classes=10, eps=MAHAL_EPS):
    means = np.stack([train_feats[train_labels == c].mean(axis=0) for c in range(num_classes)])
    centered = np.concatenate(
        [train_feats[train_labels == c] - means[c] for c in range(num_classes)], axis=0
    )
    var = centered.var(axis=0) + eps
    return means, var


def score_mahalanobis(feats, means, var):
    diffs = feats[:, None, :] - means[None, :, :]         # (N,C,D)
    d2 = (diffs ** 2 / var[None, None, :]).sum(axis=2)      # (N,C)
    return d2.min(axis=1)