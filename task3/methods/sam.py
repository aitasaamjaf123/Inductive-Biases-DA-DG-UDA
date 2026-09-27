# ══════════════════════════════════════════════════════════════════════════
# task3/methods/sam.py — standard, non-adaptive Sharpness-Aware Minimization
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
# [task3/methods/sam.py]  — standard, non-adaptive Sharpness-Aware Minimization
# =====================================================================================
class SAM(torch.optim.Optimizer):
    """Foret et al. (2021) SAM wrapper around a base optimizer (AdamW here)."""

    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# =====================================================================================
# [task3/methods/sam.py] (cont.)  — training loop
# =====================================================================================
def train_sam(cfg, classes, train_loaders, val_loaders, rho, tag="sam_main"):
    set_seed(cfg["SEED"])
    model = PACSModel(num_classes=len(classes)).to(DEVICE)
    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=rho,
                     lr=cfg["LR"], weight_decay=cfg["WD"])
    stopper = EarlyStopper(cfg["PATIENCE"])

    iters = {d: infinite_loader(train_loaders[d]) for d in cfg["SOURCE_DOMAINS"]}
    history = {"epoch": [], "cls_loss": [], "mean_source_macro_f1": []}
    steps_per_epoch = min(len(train_loaders[d]) for d in cfg["SOURCE_DOMAINS"])

    for epoch in range(cfg["MAX_EPOCHS"]):
        model.train()
        freeze_bn(model)
        running_loss = 0.0

        for _ in range(steps_per_epoch):
            xb, yb = [], []
            for d in cfg["SOURCE_DOMAINS"]:
                x, y = next(iters[d])
                xb.append(x.to(DEVICE))
                yb.append(y.to(DEVICE))
            xb = torch.cat(xb, dim=0)
            yb = torch.cat(yb, dim=0)

            # ---- first pass: ascent step ----
            logits = model(xb)
            loss1 = F.cross_entropy(logits, yb)
            loss1.backward()
            optimizer.first_step(zero_grad=True)
            freeze_bn(model)  # re-assert BN eval after any internal state churn

            # ---- second pass: descent step at perturbed point ----
            logits2 = model(xb)
            loss2 = F.cross_entropy(logits2, yb)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            freeze_bn(model)

            running_loss += loss1.item()

        per_domain, agg = evaluate_all_sources(model, val_loaders, DEVICE, cfg["SOURCE_DOMAINS"])
        model.train()
        freeze_bn(model)

        history["epoch"].append(epoch)
        history["cls_loss"].append(running_loss / steps_per_epoch)
        history["mean_source_macro_f1"].append(agg["mean_macro_f1"])

        improved, stop = stopper.step(agg["mean_macro_f1"], model)
        print(f"[SAM:{tag}] epoch {epoch} cls={history['cls_loss'][-1]:.4f} "
              f"mean_val_macroF1={agg['mean_macro_f1']:.4f} {'*best*' if improved else ''}")
        if stop:
            print(f"[SAM:{tag}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(stopper.best_state)
    ckpt_path = os.path.join(cfg["OUTPUT_DIR"], "checkpoints", f"{tag}.pth")
    torch.save(model.state_dict(), ckpt_path)
    hist_path = os.path.join(cfg["OUTPUT_DIR"], "results", f"{tag}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"[SAM:{tag}] saved checkpoint -> {ckpt_path}")
    return model, history

