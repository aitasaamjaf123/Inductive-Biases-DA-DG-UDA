# ══════════════════════════════════════════════════════════════════════════
# task3/methods/erm.py — ERM baseline
# ══════════════════════════════════════════════════════════════════════════
# The assignment requires the Task 3 ERM baseline to be the SAME checkpoint
# as Task 2's Source-only model, reused unchanged ("load its saved checkpoint
# rather than retraining it under a different configuration"). This file
# provides both paths that the original notebook implemented:
#
#   get_erm_model(...)         -> loads the Task 2 checkpoint (protocol-correct
#                                  path; requires CONFIG["TASK2_CHECKPOINT_PATH"]
#                                  to point at your Task 2 source_only.pt).
#   train_erm_from_scratch(...) -> trains ERM from scratch on Task 3's own
#                                  pipeline. This is what task3/train.py's
#                                  main() actually calls by default, as a
#                                  fallback for when no Task 2 checkpoint path
#                                  is available in the current environment.
#
# If you have your Task 2 checkpoint on disk, prefer get_erm_model() and set
# CONFIG["TASK2_CHECKPOINT_PATH"]; that is what the assignment specifies.
import os
import json

import numpy as np
import torch
import torch.nn.functional as F

from task3.shared.pacs_protocol import CONFIG, DEVICE, set_seed
from task3.shared.pacs import infinite_loader
from task3.models.backbone import PACSModel, freeze_bn, load_task2_checkpoint
from task3.selection.source_validation import EarlyStopper
from task3.evaluation.domain_metrics import evaluate_all_sources

# =====================================================================================
# [task3/methods/erm.py]  — ERM baseline = reuse Task 2 checkpoint, DO NOT retrain
# =====================================================================================
def get_erm_model(cfg, classes, device):
    model = PACSModel(num_classes=len(classes)).to(device)
    model = load_task2_checkpoint(model, cfg["TASK2_CHECKPOINT_PATH"], device)
    model.train()
    freeze_bn(model)
    return model




def train_erm_from_scratch(cfg, classes, train_loaders, val_loaders, tag="erm_scratch"):
    set_seed(cfg["SEED"])
    model = PACSModel(num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    stopper = EarlyStopper(cfg["PATIENCE"])

    iters = {d: infinite_loader(train_loaders[d]) for d in cfg["SOURCE_DOMAINS"]}
    history = {"epoch": [], "cls_loss": [], "mean_source_macro_f1": []}
    steps_per_epoch = min(len(train_loaders[d]) for d in cfg["SOURCE_DOMAINS"])

    for epoch in range(cfg["MAX_EPOCHS"]):
        model.train()
        freeze_bn(model)
        running_cls = 0.0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            all_logits, all_labels = [], []
            for d in cfg["SOURCE_DOMAINS"]:
                x, y = next(iters[d])
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                all_logits.append(logits)
                all_labels.append(y)

            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            loss = F.cross_entropy(all_logits, all_labels)
            loss.backward()
            optimizer.step()
            running_cls += loss.item()

        per_domain, agg = evaluate_all_sources(model, val_loaders, DEVICE, cfg["SOURCE_DOMAINS"])
        model.train()
        freeze_bn(model)

        history["epoch"].append(epoch)
        history["cls_loss"].append(running_cls / steps_per_epoch)
        history["mean_source_macro_f1"].append(agg["mean_macro_f1"])

        improved, stop = stopper.step(agg["mean_macro_f1"], model)
        print(f"[ERM:{tag}] epoch {epoch} cls={history['cls_loss'][-1]:.4f} "
              f"mean_val_macroF1={agg['mean_macro_f1']:.4f} {'*best*' if improved else ''}")
        if stop:
            print(f"[ERM:{tag}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(stopper.best_state)
    ckpt_path = os.path.join(cfg["OUTPUT_DIR"], "checkpoints", f"{tag}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[ERM:{tag}] saved checkpoint -> {ckpt_path}")
    return model, history
    