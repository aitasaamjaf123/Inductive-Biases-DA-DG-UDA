# ══════════════════════════════════════════════════════════════════════════
# task3/train.py — main orchestration (staged): ERM baseline, DAN-DG, SAM,
# source-side diagnostics, controlled design study. Sketch is loaded ONLY in
# the final stage, via task3/evaluate_sketch.py, after everything else is frozen.
# ══════════════════════════════════════════════════════════════════════════
import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from task3.shared.pacs_protocol import CONFIG, DEVICE, set_seed
from task3.shared.pacs import (
    discover_classes, build_transforms, build_or_load_splits,
    build_source_loaders, build_target_loader,
)
from task3.models.backbone import PACSModel, freeze_bn
from task3.methods.erm import train_erm_from_scratch, get_erm_model
from task3.methods.dan_dg import train_dan_dg
from task3.methods.sam import train_sam
from task3.evaluation.domain_metrics import evaluate_all_sources
from task3.evaluation.source_domain_separability import source_domain_separability
from task3.evaluation.sharpness import build_fixed_sharpness_batch, sharpness_proxy
from task3.evaluate_sketch import evaluate_on_sketch

# =====================================================================================
# [task3/train.py]  — MAIN ORCHESTRATION (staged)
# =====================================================================================
def main():
    print("Device:", DEVICE)
    classes = discover_classes(CONFIG)
    print("Classes (index order matters — must match Task 2):", classes)

    train_tf, eval_tf = build_transforms(CONFIG)
    splits = build_or_load_splits(CONFIG, classes)
    train_loaders, val_loaders = build_source_loaders(CONFIG, splits, train_tf, eval_tf)

    results = {}

    # ---------------------------------------------------------------- ERM baseline
    print("\n=== Training ERM baseline from scratch (Task 2 checkpoint bypassed) ===")
    erm_model, erm_hist = train_erm_from_scratch(CONFIG, classes, train_loaders, val_loaders, tag="erm_scratch")
    
    erm_per_domain, erm_agg = evaluate_all_sources(erm_model, val_loaders, DEVICE, CONFIG["SOURCE_DOMAINS"])
    results["erm"] = {
        "per_domain": {d: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for d, v in erm_per_domain.items()},
        "aggregate": erm_agg
    }
    print("ERM source validation:", results["erm"])

    # ---------------------------------------------------------------- DAN-DG
    dan_dg_model = None
    if CONFIG["RUN_TRAIN_DAN_DG"]:
        print("\n=== Training DAN-DG (lambda_DG = %.2f, main comparison) ===" % CONFIG["LAMBDA_DG_MAIN"])
        dan_dg_model, dan_dg_hist = train_dan_dg(
            CONFIG, classes, train_loaders, val_loaders,
            lambda_dg=CONFIG["LAMBDA_DG_MAIN"], tag="dan_dg_main")
        per_domain, agg = evaluate_all_sources(dan_dg_model, val_loaders, DEVICE,
                                                CONFIG["SOURCE_DOMAINS"])
        results["dan_dg"] = {"per_domain": {d: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                                             for d, v in per_domain.items()},
                              "aggregate": agg}
        plot_training_curves(dan_dg_hist, "DAN-DG (main)",
                              os.path.join(CONFIG["OUTPUT_DIR"], "plots", "dan_dg_main_curves.png"))

    # ---------------------------------------------------------------- SAM
    sam_model = None
    if CONFIG["RUN_TRAIN_SAM"]:
        print("\n=== Training SAM (rho = %.2f, main comparison) ===" % CONFIG["RHO_SAM_MAIN"])
        sam_model, sam_hist = train_sam(
            CONFIG, classes, train_loaders, val_loaders,
            rho=CONFIG["RHO_SAM_MAIN"], tag="sam_main")
        per_domain, agg = evaluate_all_sources(sam_model, val_loaders, DEVICE,
                                                CONFIG["SOURCE_DOMAINS"])
        results["sam"] = {"per_domain": {d: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                                          for d, v in per_domain.items()},
                           "aggregate": agg}
        plot_training_curves(sam_hist, "SAM (main)",
                              os.path.join(CONFIG["OUTPUT_DIR"], "plots", "sam_main_curves.png"))

    # ---------------------------------------------------------------- Source-side diagnostics
    if CONFIG["RUN_SOURCE_DIAGNOSTICS"]:
        print("\n=== Source-domain separability + sharpness proxy ===")
        fixed_x, fixed_y = build_fixed_sharpness_batch(CONFIG, splits, eval_tf, DEVICE)

        for name, model in [("erm", erm_model), ("dan_dg", dan_dg_model), ("sam", sam_model)]:
            if model is None:
                continue
            sep = source_domain_separability(CONFIG, model, val_loaders, DEVICE)
            sharp = sharpness_proxy(model, fixed_x, fixed_y, CONFIG["SHARPNESS_RHO"], DEVICE)
            model.train()
            freeze_bn(model)
            results.setdefault(name, {})["source_domain_separability"] = sep
            results.setdefault(name, {})["sharpness_delta"] = sharp
            print(f"[{name}] separability={sep} sharpness_delta={sharp:.4f}")

    with open(os.path.join(CONFIG["OUTPUT_DIR"], "results", "main_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved main comparison results.")

    # ---------------------------------------------------------------- Controlled study
    controlled_results = {}
    if CONFIG["RUN_CONTROLLED_STUDY"]:
        controlled_results = run_controlled_study(CONFIG, classes, train_loaders,
                                                    val_loaders, splits, eval_tf)
        with open(os.path.join(CONFIG["OUTPUT_DIR"], "results", "controlled_study.json"), "w") as f:
            json.dump(controlled_results, f, indent=2, default=str)

    # ---------------------------------------------------------------- FINAL: Sketch (only now!)
    if CONFIG["RUN_FINAL_SKETCH_EVAL"]:
        print("\n=== FINAL SKETCH EVALUATION (all decisions must already be frozen) ===")
        sketch_loader = build_target_loader(CONFIG, classes, eval_tf)
        final = {}
        for name, model in [("erm", erm_model), ("dan_dg", dan_dg_model), ("sam", sam_model)]:
            if model is None:
                continue
            r = evaluate_on_sketch(model, sketch_loader, DEVICE, classes)
            final[name] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                            "per_class_accuracy": r["per_class_accuracy"],
                            "confusion_matrix": r["confusion_matrix"]}
            print(f"[Sketch:{name}] acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")

        if "erm" in final:
            for name in ["dan_dg", "sam"]:
                if name in final:
                    final[name]["sketch_accuracy_delta_vs_erm"] = (
                        final[name]["accuracy"] - final["erm"]["accuracy"])
                    final[name]["sketch_macro_f1_delta_vs_erm"] = (
                        final[name]["macro_f1"] - final["erm"]["macro_f1"])

        with open(os.path.join(CONFIG["OUTPUT_DIR"], "results", "sketch_final_results.json"), "w") as f:
            json.dump(final, f, indent=2, default=str)
        print("Saved final Sketch evaluation results.")

    print("\nAll done. Outputs in:", CONFIG["OUTPUT_DIR"])




# =====================================================================================
# [task3/train.py] (cont.)  — controlled design study (Step 5)
# BOTH sweeps are implemented; CONFIG["CONTROLLED_STUDY_METHOD"] picks which runs.
# =====================================================================================
def run_controlled_study(cfg, classes, train_loaders, val_loaders, splits, eval_tf):
    method = cfg["CONTROLLED_STUDY_METHOD"]
    assert method in ("dan_dg", "sam"), "CONTROLLED_STUDY_METHOD must be 'dan_dg' or 'sam'"
    print(f"\n=== Controlled design study: {method} sweep ===")

    fixed_x, fixed_y = build_fixed_sharpness_batch(cfg, splits, eval_tf, DEVICE)
    sweep_results = {}

    if method == "dan_dg":
        for lam in cfg["LAMBDA_DG_SWEEP"]:
            tag = f"dan_dg_lambda_{lam}"
            model, hist = train_dan_dg(cfg, classes, train_loaders, val_loaders,
                                        lambda_dg=lam, tag=tag)
            per_domain, agg = evaluate_all_sources(model, val_loaders, DEVICE,
                                                     cfg["SOURCE_DOMAINS"])
            sep = source_domain_separability(cfg, model, val_loaders, DEVICE)
            model.train(); freeze_bn(model)
            entry = {"lambda_dg": lam,
                     "source_aggregate": agg,
                     "source_domain_separability": sep}
            sweep_results[str(lam)] = entry
            print(f"[sweep dan_dg] lambda={lam} mean_source_macroF1={agg['mean_macro_f1']:.4f} "
                  f"separability={sep['source_domain_separability_accuracy']:.4f}")

    else:  # method == "sam"
        for rho in cfg["RHO_SAM_SWEEP"]:
            tag = f"sam_rho_{rho}"
            model, hist = train_sam(cfg, classes, train_loaders, val_loaders,
                                     rho=rho, tag=tag)
            per_domain, agg = evaluate_all_sources(model, val_loaders, DEVICE,
                                                     cfg["SOURCE_DOMAINS"])
            sharp = sharpness_proxy(model, fixed_x, fixed_y, cfg["SHARPNESS_RHO"], DEVICE)
            model.train(); freeze_bn(model)
            entry = {"rho": rho,
                     "source_aggregate": agg,
                     "sharpness_delta": sharp}
            sweep_results[str(rho)] = entry
            print(f"[sweep sam] rho={rho} mean_source_macroF1={agg['mean_macro_f1']:.4f} "
                  f"sharpness_delta={sharp:.4f}")

    # Sketch results for the sweep are ANALYSIS ONLY — computed but must not be used
    # to revise lambda_DG=1 / rho=0.05 as the main-comparison settings (per the manual).
    if cfg["RUN_FINAL_SKETCH_EVAL"]:
        print("[sweep] Also evaluating sweep checkpoints on Sketch (analysis only, "
              "not for re-tuning).")
        sketch_loader = build_target_loader(cfg, classes, eval_tf)
        for key, entry in sweep_results.items():
            tag = (f"dan_dg_lambda_{entry['lambda_dg']}" if method == "dan_dg"
                   else f"sam_rho_{entry['rho']}")
            model = PACSModel(num_classes=len(classes)).to(DEVICE)
            ckpt = torch.load(os.path.join(cfg["OUTPUT_DIR"], "checkpoints", f"{tag}.pth"),
                               map_location=DEVICE)
            model.load_state_dict(ckpt)
            r = evaluate_on_sketch(model, sketch_loader, DEVICE, classes)
            entry["sketch_accuracy"] = r["accuracy"]
            entry["sketch_macro_f1"] = r["macro_f1"]

    return {"method": method, "sweep": sweep_results}


# =====================================================================================
# Plot helper (used by task3/train.py)
# =====================================================================================
def plot_training_curves(history, title, save_path):
    fig, axes = plt.subplots(1, 2 if "mmd_loss" in history else 1, figsize=(10, 4))
    axes = np.atleast_1d(axes)
    axes[0].plot(history["epoch"], history["cls_loss"], label="cls loss")
    if "mmd_loss" in history:
        axes[0].plot(history["epoch"], history["mmd_loss"], label="mmd loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title(f"{title}: training losses")

    ax_f1 = axes[1] if len(axes) > 1 else axes[0].twinx()
    ax_f1.plot(history["epoch"], history["mean_source_macro_f1"], color="green",
               label="mean source val macro-F1")
    ax_f1.set_ylabel("macro-F1")
    if len(axes) > 1:
        ax_f1.set_xlabel("epoch")
        ax_f1.set_title(f"{title}: mean source val macro-F1")
        ax_f1.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print("Saved plot ->", save_path)
