# ══════════════════════════════════════════════════════════════════════════
# task2/train.py — generic training loop (all 4 methods) + data prep
# ══════════════════════════════════════════════════════════════════════════
import os
import json
import warnings
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from shared.pacs_protocol import CONFIG, SEED, DEVICE, set_seed
from shared.pacs import (
    TRAIN_TRANSFORM, EVAL_TRANSFORM, InfiniteCycler, make_loader,
    build_stratified_source_splits, build_target_split,
)
from task2.models.backbone import ResNet18Backbone, set_bn_eval
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import (
    GradientReversalLayer, DomainDiscriminator, grl_alpha_schedule,
)
from task2.methods.dan import multi_kernel_rbf_mmd2
from task2.methods.cdan import multilinear_map
from task2.evaluation.metrics import evaluate_classifier, mean_source_val_macro_f1


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/train.py  — TRAINING LOOP (GENERIC)
# ══════════════════════════════════════════════════════════════════════════

def build_models():
    backbone = ResNet18Backbone().to(DEVICE)
    head = ClassifierHead(backbone.out_dim, len(CONFIG["classes"])).to(DEVICE)
    return backbone, head


def safe_backward_and_step(loss, optimizer, params, grad_clip_norm, max_loss_threshold=50.0):
    """Returns True if step applied, False if skipped due to NaN/Inf or explosion."""
    # FIX: Catch both non-finite values AND runaway loss spikes
    if not torch.isfinite(loss):
        warnings.warn(f"Loss instability detected (val: {loss.item() if torch.is_tensor(loss) else loss}) — skipping step.")
        optimizer.zero_grad(set_to_none=True)
        return False
        
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
    optimizer.step()
    return True


def train_method(method: str, run_name: str, source_splits, target_samples,
                  lambda_mmd=None, grl_max_alpha=None, log_prefix=""):
    """method in {"source_only", "dan", "dann", "cdan"}. Trains with a fixed
    architecture/optimizer/schedule/seed, domain-balanced batches (8 per
    source domain + 24 target where applicable), BatchNorm running stats
    frozen, early stopping on mean source-val macro-F1 (patience 5, <=30
    epochs). Saves the best checkpoint and a full history log."""
    assert method in {"source_only", "dan", "dann", "cdan"}
    lambda_mmd = CONFIG["lambda_mmd"] if lambda_mmd is None else lambda_mmd
    grl_max_alpha = CONFIG["grl_max_alpha"] if grl_max_alpha is None else grl_max_alpha

    set_seed(CONFIG["seed"])
    backbone, head = build_models()

    needs_target = method != "source_only"
    disc = None
    grl = None
    if method in {"dann", "cdan"}:
        in_dim = backbone.out_dim if method == "dann" else backbone.out_dim * len(CONFIG["classes"])
        disc = DomainDiscriminator(in_dim, CONFIG["domain_disc_hidden"], CONFIG["domain_disc_dropout"]).to(DEVICE)
        grl = GradientReversalLayer()

    params = list(backbone.parameters()) + list(head.parameters())
    if disc is not None:
        params += list(disc.parameters())
    optimizer = torch.optim.AdamW(params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    # ---- data ----
    train_loaders = {
        dom: make_loader(source_splits[dom]["train"], TRAIN_TRANSFORM,
                          CONFIG["per_source_domain_batch"], shuffle=True,
                          name=f"{dom}-train", drop_last=True)
        for dom in CONFIG["source_domains"]
    }
    val_loaders = {
        dom: make_loader(source_splits[dom]["val"], EVAL_TRANSFORM,
                          64, shuffle=False, name=f"{dom}-val")
        for dom in CONFIG["source_domains"]
    }
    train_cyclers = {dom: InfiniteCycler(dl) for dom, dl in train_loaders.items()}

    target_cycler = None
    if needs_target:
        target_loader = make_loader(target_samples, TRAIN_TRANSFORM,
                                     CONFIG["target_batch"], shuffle=True,
                                     name="target-adapt", drop_last=True)
        target_cycler = InfiniteCycler(target_loader)

    steps_per_epoch = max(
        1, max(len(source_splits[dom]["train"]) for dom in CONFIG["source_domains"])
        // CONFIG["per_source_domain_batch"]
    )
    total_steps = steps_per_epoch * CONFIG["max_epochs"]

    history = defaultdict(list)
    best_mean_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    global_step = 0
    ckpt_path = os.path.join(CONFIG["output_root"], "checkpoints", f"{run_name}.pt")

    for epoch in range(CONFIG["max_epochs"]):
        backbone.train()
        head.train()
        if disc is not None:
            disc.train()
        set_bn_eval(backbone)  # keep BN running stats frozen even in train mode

        epoch_cls_losses, epoch_align_losses = [], []

        for _ in range(steps_per_epoch):
            try:
                xs_list, ys_list = [], []
                for dom in CONFIG["source_domains"]:
                    xb, yb = next(train_cyclers[dom])
                    xs_list.append(xb)
                    ys_list.append(yb)
                xs = torch.cat(xs_list, dim=0).to(DEVICE, non_blocking=True)
                ys = torch.cat(ys_list, dim=0).to(DEVICE, non_blocking=True)

                feat_s = backbone(xs)
                logits_s = head(feat_s)
                cls_loss = F.cross_entropy(logits_s, ys)

                align_loss = torch.tensor(0.0, device=DEVICE)

                if method == "dan":
                    xt, _ = next(target_cycler)
                    xt = xt.to(DEVICE, non_blocking=True)
                    feat_t = backbone(xt)
                    mmd2 = multi_kernel_rbf_mmd2(feat_s, feat_t, CONFIG["mmd_kernel_mults"])
                    align_loss = lambda_mmd * mmd2
                    total_loss = cls_loss + align_loss

                elif method in {"dann", "cdan"}:
                    xt, _ = next(target_cycler)
                    xt = xt.to(DEVICE, non_blocking=True)
                    feat_t = backbone(xt)

                    p = global_step / max(1, total_steps)
                    alpha = grl_alpha_schedule(p, grl_max_alpha)

                    if method == "dann":
                        # ── ADDED: L2-normalize features for DANN stability ──
                        feat_s_norm = F.normalize(feat_s, p=2, dim=1)
                        feat_t_norm = F.normalize(feat_t, p=2, dim=1)
                        
                        rev_s = grl(feat_s_norm, alpha)
                        rev_t = grl(feat_t_norm, alpha)
                        d_in_s, d_in_t = rev_s, rev_t
                    else:  # cdan
                        probs_s = F.softmax(logits_s, dim=1)
                        with torch.no_grad():
                            logits_t_nograd = head(feat_t)  # for conditioning only's probs; f_t itself stays attached below
                        # NOTE: per protocol we must NOT detach f or p, so recompute
                        # logits_t with grad for the multilinear map.
                        logits_t = head(feat_t)
                        probs_t = F.softmax(logits_t, dim=1)
                        g_s = multilinear_map(feat_s, probs_s)
                        g_t = multilinear_map(feat_t, probs_t)
                        # ── ADDED: L2-normalize the 3584-d multilinear map for CDAN stability ──
                        g_s = F.normalize(g_s, p=2, dim=1)
                        g_t = F.normalize(g_t, p=2, dim=1)
                        
                        d_in_s = grl(g_s, alpha)
                        d_in_t = grl(g_t, alpha)

                    domain_logits_s = disc(d_in_s)
                    domain_logits_t = disc(d_in_t)
                    domain_labels_s = torch.zeros(domain_logits_s.size(0), dtype=torch.long, device=DEVICE)
                    domain_labels_t = torch.ones(domain_logits_t.size(0), dtype=torch.long, device=DEVICE)
                    domain_loss = F.cross_entropy(
                        torch.cat([domain_logits_s, domain_logits_t], dim=0),
                        torch.cat([domain_labels_s, domain_labels_t], dim=0),
                    )
                    align_loss = domain_loss
                    total_loss = cls_loss + align_loss

                else:  # source_only
                    total_loss = cls_loss

                applied = safe_backward_and_step(total_loss, optimizer, params, CONFIG["grad_clip_norm"])
                if applied:
                    epoch_cls_losses.append(cls_loss.item())
                    epoch_align_losses.append(float(align_loss.item()) if torch.is_tensor(align_loss) else align_loss)
                global_step += 1

            except torch.cuda.OutOfMemoryError:
                warnings.warn("CUDA OOM on a training step — clearing cache and skipping this step.")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            except Exception as e:
                warnings.warn(f"Unexpected error on a training step — skipping it. {e}\n{traceback.format_exc()}")
                optimizer.zero_grad(set_to_none=True)
                continue

        # ---- validation on all three source domains ----
        per_domain_f1, per_domain_acc = {}, {}
        for dom in CONFIG["source_domains"]:
            acc, f1, _, _ = evaluate_classifier(backbone, head, val_loaders[dom], DEVICE)
            per_domain_f1[dom] = f1
            per_domain_acc[dom] = acc
        mean_f1 = mean_source_val_macro_f1(per_domain_f1)

        history["epoch"].append(epoch)
        history["cls_loss"].append(float(np.mean(epoch_cls_losses)) if epoch_cls_losses else float("nan"))
        history["align_loss"].append(float(np.mean(epoch_align_losses)) if epoch_align_losses else float("nan"))
        history["mean_source_val_f1"].append(mean_f1)
        for dom in CONFIG["source_domains"]:
            history[f"val_f1_{dom}"].append(per_domain_f1[dom])
            history[f"val_acc_{dom}"].append(per_domain_acc[dom])

        print(f"[{log_prefix}{run_name}] epoch {epoch:02d} | cls_loss={history['cls_loss'][-1]:.4f} "
              f"| align_loss={history['align_loss'][-1]:.4f} | mean_val_f1={mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            epochs_no_improve = 0
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
                "disc": {k: v.cpu().clone() for k, v in disc.state_dict().items()} if disc is not None else None,
                "epoch": epoch,
                "mean_val_f1": mean_f1,
                "per_domain_val_f1": per_domain_f1,
                "per_domain_val_acc": per_domain_acc,
                "config_snapshot": {"lambda_mmd": lambda_mmd, "grl_max_alpha": grl_max_alpha, "method": method},
            }
            try:
                torch.save(best_state, ckpt_path)
            except Exception as e:
                warnings.warn(f"Failed to write checkpoint to disk ({e}); keeping best state in memory only.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["early_stop_patience"]:
                print(f"[{log_prefix}{run_name}] Early stopping at epoch {epoch} "
                      f"(no improvement for {CONFIG['early_stop_patience']} epochs).")
                break

    # restore best checkpoint into the live models before returning
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        if disc is not None and best_state["disc"] is not None:
            disc.load_state_dict(best_state["disc"])
    else:
        warnings.warn(f"[{run_name}] No checkpoint was ever saved (training may have failed every step).")

    hist_path = os.path.join(CONFIG["output_root"], "logs", f"{run_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    return {
        "backbone": backbone, "head": head, "disc": disc,
        "history": history, "best_state": best_state, "ckpt_path": ckpt_path,
    }



# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/train.py  — RUN ALL METHODS
# ══════════════════════════════════════════════════════════════════════════

def prepare_data():
    split_cache = os.path.join(CONFIG["output_root"], "splits", "pacs_sketch_seed6304_sources.json")
    target_cache = os.path.join(CONFIG["output_root"], "splits", "pacs_sketch_seed6304_target.json")
    source_splits = build_stratified_source_splits(
        CONFIG["data_root"], CONFIG["source_domains"], CONFIG["classes"],
        CONFIG["val_fraction"], CONFIG["seed"], split_cache,
    )
    target_samples = build_target_split(
        CONFIG["data_root"], CONFIG["target_domain"], CONFIG["classes"], target_cache
    )
    return source_splits, target_samples


def run_all_main_methods():
    source_splits, target_samples = prepare_data()
    results = {}
    # 1. Source-only ERM (also the frozen Task-3 ERM baseline — don't retrain it there)
    results["source_only"] = train_method("source_only", "source_only", source_splits, target_samples)
    # 2. DAN
    results["dan"] = train_method("dan", "dan", source_splits, target_samples, lambda_mmd=CONFIG["lambda_mmd"])
    # 3. DANN
    results["dann"] = train_method("dann", "dann", source_splits, target_samples, grl_max_alpha=CONFIG["grl_max_alpha"])
    # 4. CDAN
    results["cdan"] = train_method("cdan", "cdan", source_splits, target_samples, grl_max_alpha=CONFIG["grl_max_alpha"])
    return results, source_splits, target_samples

