# ══════════════════════════════════════════════════════════════════════════
# task3/evaluate_sketch.py — FINAL, target-label-touching evaluation only
# ══════════════════════════════════════════════════════════════════════════
# No Sketch image or label may be used anywhere else in Task 3 (training,
# checkpoint selection, diagnostics, controlled study). This module is the
# single place Sketch labels are used, and only after every Task 3 decision
# has been frozen.
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =====================================================================================
# [task3/evaluate_sketch.py]  — FINAL, target-label-touching evaluation only
# =====================================================================================
@torch.no_grad()
def evaluate_on_sketch(model, sketch_loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in sketch_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

    per_class_acc = {}
    for i, c in enumerate(classes):
        mask = all_labels == i
        per_class_acc[c] = float((all_preds[mask] == i).mean()) if mask.sum() > 0 else None

    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist(),
            "per_class_accuracy": per_class_acc, "preds": all_preds, "labels": all_labels}



