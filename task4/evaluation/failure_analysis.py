"""
Task 4 — locates incorrectly accepted unknown examples under a frozen
threshold and records the unknown class, predicted known class, score, and
threshold, for the required qualitative failure analysis.
"""
import numpy as np
from configs.config import CIFAR10_CLASSES
from scores.mls import score_mls


def failure_analysis(logits_test_known, tau, group_name, group_logits, group_fine_labels,
                      cifar100_classes, n=3, score_fn=score_mls):
    """Finds examples from `group_logits` (an unknown group's known-class
    logits) that are INCORRECTLY ACCEPTED under threshold tau, and reports:
    unknown class name, predicted CIFAR-10 class, score, threshold."""
    u = score_fn(group_logits)
    accepted_idx = np.where(u <= tau)[0]
    rows = []
    for idx in accepted_idx[:max(n, len(accepted_idx))]:
        pred = int(group_logits[idx].argmax())
        rows.append({
            "group": group_name,
            "true_unknown_class": cifar100_classes[int(group_fine_labels[idx])],
            "predicted_known_class": CIFAR10_CLASSES[pred],
            "score": float(u[idx]),
            "threshold": float(tau),
        })
        if len(rows) >= n:
            break
    return rows