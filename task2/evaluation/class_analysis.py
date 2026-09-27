# ══════════════════════════════════════════════════════════════════════════
# task2/evaluation/class_analysis.py
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
from sklearn.metrics import confusion_matrix


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/evaluation/class_analysis.py
# ══════════════════════════════════════════════════════════════════════════

def per_class_accuracy(preds, labels, num_classes):
    accs = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            accs[c] = (preds[mask] == c).mean()
    return accs


def class_shift_report(baseline_preds, baseline_labels, method_preds, method_labels, classes, top_k=3):
    """Per-class accuracy delta (method - source-only) plus the dominant
    confusion for the classes that moved the most in each direction."""
    assert np.array_equal(baseline_labels, method_labels), "Label ordering must match between runs (same target set/order)."
    n_classes = len(classes)
    base_acc = per_class_accuracy(baseline_preds, baseline_labels, n_classes)
    meth_acc = per_class_accuracy(method_preds, method_labels, n_classes)
    delta = meth_acc - base_acc

    order = np.argsort(delta)
    worst = order[:top_k]
    best = order[::-1][:top_k]

    cm = confusion_matrix(method_labels, method_preds, labels=list(range(n_classes)))

    def dominant_confusion(cls_idx):
        row = cm[cls_idx].copy()
        row[cls_idx] = -1  # exclude the diagonal (correct predictions)
        if row.max() <= 0:
            return None
        confused_with = int(row.argmax())
        return classes[confused_with], int(cm[cls_idx, confused_with])

    report = {
        "per_class_delta": {classes[i]: float(delta[i]) for i in range(n_classes)},
        "most_improved": [
            {"class": classes[i], "delta": float(delta[i]), "dominant_confusion": dominant_confusion(i)}
            for i in best
        ],
        "most_degraded": [
            {"class": classes[i], "delta": float(delta[i]), "dominant_confusion": dominant_confusion(i)}
            for i in worst
        ],
    }
    return report

