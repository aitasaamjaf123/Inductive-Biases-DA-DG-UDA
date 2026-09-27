# ══════════════════════════════════════════════════════════════════════════
# task2/evaluation/metrics.py
# ══════════════════════════════════════════════════════════════════════════
import math
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score



# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/evaluation/metrics.py  — EVAL: METRICS
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_classifier(backbone, head, loader, device):
    """Returns (accuracy, macro_f1, all_preds, all_labels). Robust to an
    empty loader (returns NaNs rather than dividing by zero)."""
    backbone.eval()
    head.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        feat = backbone(xb)
        logits = head(feat)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    if len(all_preds) == 0:
        return float("nan"), float("nan"), np.array([]), np.array([])
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1, all_preds, all_labels


def mean_source_val_macro_f1(per_domain_f1: dict) -> float:
    vals = [v for v in per_domain_f1.values() if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")

