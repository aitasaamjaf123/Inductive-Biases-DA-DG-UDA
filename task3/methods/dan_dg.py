# ══════════════════════════════════════════════════════════════════════════
# task3/methods/dan_dg.py — pairwise multi-kernel MMD across the THREE
# observed source domains (Photo/Art Painting/Cartoon). Never touches Sketch.
# ══════════════════════════════════════════════════════════════════════════
import os
import json

import numpy as np
import torch
import torch.nn.functional as F

from task3.shared.pacs_protocol import CONFIG, DEVICE, set_seed
from task3.shared.pacs import infinite_loader
from task3.models.backbone import PACSModel, freeze_bn
from task3.selection.source_validation import EarlyStopper
from task3.evaluation.domain_metrics import evaluate_all_sources


# =====================================================================================
# [task3/methods/dan_dg.py]  — pairwise multi-kernel MMD between source domains
# =====================================================================================
def multi_kernel_rbf(X, Y, mults=(0.5, 1.0, 2.0)):
    """Sum of RBF kernels with bandwidths = mult * median(pairwise sq. dist)
    over the combined (X,Y) batch — recomputed fresh every call (i.e. every
    batch / every domain pair), exactly as the manual specifies."""
    XY = torch.cat([X, Y], dim=0)
    sq_dists = torch.cdist(XY, XY, p=2.0) ** 2
    nonzero = sq_dists[sq_dists > 0]
    if nonzero.numel() == 0:
        median = torch.tensor(1.0, device=X.device)
    else:
        median = nonzero.median()
        if median.item() == 0:
            median = torch.tensor(1.0, device=X.device)

    kernel_sum = torch.zeros_like(sq_dists)
    for m in mults:
        bandwidth = median * m
        kernel_sum = kernel_sum + torch.exp(-sq_dists / (bandwidth + 1e-12))
    return kernel_sum


def mmd2(X, Y, mults=(0.5, 1.0, 2.0)):
    """Squared MMD estimate (biased, V-statistic form) between feature sets X, Y."""
    n, m = X.size(0), Y.size(0)
    K = multi_kernel_rbf(X, Y, mults)
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


def pairwise_source_mmd(features_by_domain, mults=(0.5, 1.0, 2.0)):
    """Average MMD^2 over the 3 unordered pairs of source domains."""
    domains = list(features_by_domain.keys())
    pairs = [(domains[i], domains[j])
             for i in range(len(domains)) for j in range(i + 1, len(domains))]
    total = 0.0
    for d1, d2 in pairs:
        total = total + mmd2(features_by_domain[d1], features_by_domain[d2], mults)
    return total / len(pairs)





# =====================================================================================
# [task3/methods/dan_dg.py] (cont.)  — training loop
# =====================================================================================
def train_dan_dg(cfg, classes, train_loaders, val_loaders, lambda_dg, tag="dan_dg_main"):
    set_seed(cfg["SEED"])
    model = PACSModel(num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    stopper = EarlyStopper(cfg["PATIENCE"])

    iters = {d: infinite_loader(train_loaders[d]) for d in cfg["SOURCE_DOMAINS"]}
    history = {"epoch": [], "cls_loss": [], "mmd_loss": [], "total_loss": [],
               "mean_source_macro_f1": []}

    steps_per_epoch = min(len(train_loaders[d]) for d in cfg["SOURCE_DOMAINS"])

    for epoch in range(cfg["MAX_EPOCHS"]):
        model.train()
        freeze_bn(model)
        running_cls, running_mmd, running_total = 0.0, 0.0, 0.0

        for _ in range(steps_per_epoch):
            xs, ys, feats = {}, {}, {}
            optimizer.zero_grad()

            all_logits, all_labels = [], []
            for d in cfg["SOURCE_DOMAINS"]:
                x, y = next(iters[d])
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, feat = model(x, return_features=True)
                xs[d], ys[d], feats[d] = x, y, feat
                all_logits.append(logits)
                all_labels.append(y)

            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            cls_loss = F.cross_entropy(all_logits, all_labels)

            mmd_loss = pairwise_source_mmd(feats, cfg["MMD_KERNEL_MULTS"])
            total_loss = cls_loss + lambda_dg * mmd_loss

            total_loss.backward()
            optimizer.step()

            running_cls += cls_loss.item()
            running_mmd += mmd_loss.item()
            running_total += total_loss.item()

        per_domain, agg = evaluate_all_sources(model, val_loaders, DEVICE, cfg["SOURCE_DOMAINS"])
        model.train()
        freeze_bn(model)

        history["epoch"].append(epoch)
        history["cls_loss"].append(running_cls / steps_per_epoch)
        history["mmd_loss"].append(running_mmd / steps_per_epoch)
        history["total_loss"].append(running_total / steps_per_epoch)
        history["mean_source_macro_f1"].append(agg["mean_macro_f1"])

        improved, stop = stopper.step(agg["mean_macro_f1"], model)
        print(f"[DAN-DG:{tag}] epoch {epoch} cls={history['cls_loss'][-1]:.4f} "
              f"mmd={history['mmd_loss'][-1]:.4f} mean_val_macroF1={agg['mean_macro_f1']:.4f} "
              f"{'*best*' if improved else ''}")
        if stop:
            print(f"[DAN-DG:{tag}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(stopper.best_state)
    ckpt_path = os.path.join(cfg["OUTPUT_DIR"], "checkpoints", f"{tag}.pth")
    torch.save(model.state_dict(), ckpt_path)
    hist_path = os.path.join(cfg["OUTPUT_DIR"], "results", f"{tag}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"[DAN-DG:{tag}] saved checkpoint -> {ckpt_path}")
    return model, history

