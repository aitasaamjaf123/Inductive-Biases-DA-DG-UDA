# ══════════════════════════════════════════════════════════════════════════
# task2/evaluate_final.py — final comparison table, training curves, controlled study
# ══════════════════════════════════════════════════════════════════════════
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shared.pacs_protocol import CONFIG, SEED, DEVICE
from shared.pacs import EVAL_TRANSFORM, make_loader
from task2.evaluation.metrics import evaluate_classifier
from task2.evaluation.domain_separability import extract_features, domain_separability_score
from task2.evaluation.class_analysis import class_shift_report
from task2.train import train_method  # controlled_design_study() trains fresh sweep runs


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/evaluate_final.py  — FINAL EVALUATION TABLE
# ══════════════════════════════════════════════════════════════════════════

def final_target_and_source_eval(results, source_splits, target_samples):
    """Target labels are used ONLY here, after every checkpoint/setting is
    already frozen (Step 5, "Common Evaluation")."""
    val_loaders = {
        dom: make_loader(source_splits[dom]["val"], EVAL_TRANSFORM, 64, shuffle=False, name=f"{dom}-val")
        for dom in CONFIG["source_domains"]
    }
    target_loader = make_loader(target_samples, EVAL_TRANSFORM, 64, shuffle=False, name="target-eval")

    rows = []
    preds_by_method, labels_by_method = {}, {}
    feats_by_method_target = {}
    feats_by_method_source_val = {}

    for method_name, res in results.items():
        backbone, head = res["backbone"], res["head"]
        row = {"method": method_name}
        per_dom_acc, per_dom_f1 = {}, {}
        for dom in CONFIG["source_domains"]:
            acc, f1, _, _ = evaluate_classifier(backbone, head, val_loaders[dom], DEVICE)
            per_dom_acc[dom] = acc
            per_dom_f1[dom] = f1
            row[f"src_val_acc_{dom}"] = acc
            row[f"src_val_f1_{dom}"] = f1
        row["mean_src_val_acc"] = float(np.mean(list(per_dom_acc.values())))
        row["mean_src_val_f1"] = float(np.mean(list(per_dom_f1.values())))

        t_acc, t_f1, t_preds, t_labels = evaluate_classifier(backbone, head, target_loader, DEVICE)
        row["target_acc"] = t_acc
        row["target_macro_f1"] = t_f1
        preds_by_method[method_name] = t_preds
        labels_by_method[method_name] = t_labels

        # features for domain-separability: pool all three source-val sets
        src_feats_list = []
        for dom in CONFIG["source_domains"]:
            f, _ = extract_features(backbone, val_loaders[dom], DEVICE)
            src_feats_list.append(f)
        feats_by_method_source_val[method_name] = np.concatenate(src_feats_list, axis=0)
        tgt_feats, _ = extract_features(backbone, target_loader, DEVICE)
        feats_by_method_target[method_name] = tgt_feats

        sep_score = domain_separability_score(
            feats_by_method_source_val[method_name], feats_by_method_target[method_name], seed=CONFIG["seed"]
        )
        row["domain_separability"] = sep_score

        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")
    if "source_only" in df.index:
        df["target_acc_change_vs_source_only"] = df["target_acc"] - df.loc["source_only", "target_acc"]

    results_csv = os.path.join(CONFIG["output_root"], "results", "main_comparison_table.csv")
    df.to_csv(results_csv)
    print(df)

    # per-class shift analysis for each adapted method vs source-only
    class_reports = {}
    if "source_only" in preds_by_method:
        for method_name in results:
            if method_name == "source_only":
                continue
            class_reports[method_name] = class_shift_report(
                preds_by_method["source_only"], labels_by_method["source_only"],
                preds_by_method[method_name], labels_by_method[method_name],
                CONFIG["classes"],
            )
        with open(os.path.join(CONFIG["output_root"], "results", "per_class_shift.json"), "w") as f:
            json.dump(class_reports, f, indent=2)

    return df, class_reports


def plot_training_curves(results):
    """Classification + alignment/domain loss curves per method, and
    val-F1 curves, saved to results/."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for method_name, res in results.items():
        h = res["history"]
        axes[0].plot(h["epoch"], h["cls_loss"], label=method_name)
        axes[1].plot(h["epoch"], h["align_loss"], label=method_name)
        axes[2].plot(h["epoch"], h["mean_source_val_f1"], label=method_name)
    axes[0].set_title("Classification loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("Alignment / domain loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].set_title("Mean source-val macro-F1"); axes[2].set_xlabel("epoch"); axes[2].legend()
    plt.tight_layout()
    out_path = os.path.join(CONFIG["output_root"], "results", "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved curves to {out_path}")



# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/evaluate_final.py  — CONTROLLED STUDY (Step 6)
# ══════════════════════════════════════════════════════════════════════════

def controlled_design_study(source_splits, target_samples):
    """Bounded ablation. Pick ONE axis (see CONFIG["controlled_study"]["method"]):
       - "dan":  sweep lambda_mmd over {0.1, 1, 10}
       - "dann": sweep grl_max_alpha over {0.25, 0.5, 1}
    Everything else stays fixed. Prints the stated expectation BEFORE
    training so the interpretation isn't post-hoc, then reports results for
    analysis only (not used to revise the main comparison)."""
    which = CONFIG["controlled_study"]["method"]
    val_loaders = {
        dom: make_loader(source_splits[dom]["val"], EVAL_TRANSFORM, 64, shuffle=False, name=f"{dom}-val")
        for dom in CONFIG["source_domains"]
    }
    target_loader = make_loader(target_samples, EVAL_TRANSFORM, 64, shuffle=False, name="target-eval")

    print("=" * 78)
    if which == "dan":
        print("CONTROLLED STUDY (DAN, sweeping lambda_mmd in {0.1, 1, 10}).")
        print("Stated expectation: as lambda_mmd increases, the MMD penalty should "
              "push source and target features closer together, so we expect domain "
              "separability to fall toward chance (50%). Source classification "
              "performance should be roughly stable at small lambda but may degrade "
              "at lambda=10 if alignment pressure starts to compete with the "
              "classification objective. Target recognition should improve while "
              "alignment is helpful, but could plateau or reverse if over-alignment "
              "collapses class-discriminative structure (negative transfer).")
        grid = [("lambda_mmd", v) for v in CONFIG["controlled_study"]["lambda_mmd_grid"]]
        method = "dan"
    else:
        print("CONTROLLED STUDY (DANN, sweeping grl_max_alpha in {0.25, 0.5, 1}).")
        print("Stated expectation: a larger maximum reversal strength should make the "
              "backbone confuse the domain discriminator more strongly, so domain "
              "separability should fall as grl_max_alpha increases. Source accuracy "
              "should stay roughly flat at low strength but may drop at strength=1 if "
              "the reversed gradient starts to dominate the classification signal. "
              "Target accuracy should improve up to a point, with a possible reversal "
              "at the highest strength if alignment overwhelms class structure.")
        grid = [("grl_max_alpha", v) for v in CONFIG["controlled_study"]["grl_max_alpha_grid"]]
        method = "dann"
    print("=" * 78)

    rows = []
    for param_name, value in grid:
        run_name = f"controlled_{method}_{param_name}_{value}"
        kwargs = {param_name: value}
        res = train_method(method, run_name, source_splits, target_samples, log_prefix="[controlled] ", **kwargs)
        backbone, head = res["backbone"], res["head"]

        per_dom_acc = {}
        for dom in CONFIG["source_domains"]:
            acc, _, _, _ = evaluate_classifier(backbone, head, val_loaders[dom], DEVICE)
            per_dom_acc[dom] = acc
        mean_src_acc = float(np.mean(list(per_dom_acc.values())))

        t_acc, t_f1, _, _ = evaluate_classifier(backbone, head, target_loader, DEVICE)

        src_feats_list = [extract_features(backbone, val_loaders[dom], DEVICE)[0] for dom in CONFIG["source_domains"]]
        src_feats = np.concatenate(src_feats_list, axis=0)
        tgt_feats, _ = extract_features(backbone, target_loader, DEVICE)
        sep = domain_separability_score(src_feats, tgt_feats, seed=CONFIG["seed"])

        rows.append({param_name: value, "mean_source_val_acc": mean_src_acc,
                     "target_acc": t_acc, "target_macro_f1": t_f1, "domain_separability": sep})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(CONFIG["output_root"], "results", f"controlled_study_{method}.csv")
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"Saved controlled-study table to {out_csv}")
    return df

