# ══════════════════════════════════════════════════════════════════════════
# task3/evaluation/domain_metrics.py — shared accuracy / macro-F1 helpers
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# =====================================================================================
# [task3/evaluation/domain_metrics.py]  — shared accuracy / macro-F1 helpers
# =====================================================================================
@torch.no_grad()
def evaluate_domain(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        n += y.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / max(n, 1)
    return {"accuracy": acc, "macro_f1": macro_f1, "loss": avg_loss,
            "preds": all_preds, "labels": all_labels}


def evaluate_all_sources(model, val_loaders, device, source_domains):
    per_domain = {}
    for d in source_domains:
        per_domain[d] = evaluate_domain(model, val_loaders[d], device)
    accs = [per_domain[d]["accuracy"] for d in source_domains]
    f1s = [per_domain[d]["macro_f1"] for d in source_domains]
    mean_acc, worst_acc = float(np.mean(accs)), float(np.min(accs))
    mean_f1, worst_f1 = float(np.mean(f1s)), float(np.min(f1s))
    return per_domain, {"mean_accuracy": mean_acc, "worst_accuracy": worst_acc,
                         "mean_macro_f1": mean_f1, "worst_macro_f1": worst_f1}

