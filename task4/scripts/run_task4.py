"""
Task 4 — main pipeline. Runs Steps 1-6 end to end:
  Vanilla -> post-hoc scores -> GCSC -> PROSER -> (optional RPL) ->
  common evaluation tables, required figure, and failure analysis.

Run from the task4/ directory:
    python scripts/run_task4.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd

from configs.config import (CKPT_DIR, RESULTS_DIR, NUM_EPOCHS_MAIN, NUM_EPOCHS_PROSER,
                             LR_MAIN, LR_PROSER, NUM_DUMMY, RUN_RPL, PROSER_SCORE_MODE,
                             PLOT_STYLE, NEAR_CLASSES, FAR_CLASSES)
from data.cifar10 import get_cifar10_loaders, get_cifar10_unaugmented_train_loader
from data.cifar100_unknowns import get_cifar100_unknown_loader
from models.resnet_cifar import ResNet18Cifar
from train import train_classifier
from evaluation.thresholds import evaluate_accuracy
from extract_outputs import extract_and_cache
from scores.msp import score_msp
from scores.mls import score_mls
from scores.energy import score_energy
from scores.mahalanobis import fit_mahalanobis, score_mahalanobis
from scores.proser_score import score_proser
from methods.proser import train_proser
from methods.rpl import train_rpl, rpl_extract, rpl_score, rpl_accuracy_from_d2
from evaluation.metrics import eval_score_vs_group, compute_threshold_95
from evaluation.evaluate_osr import plot_score_panels
from evaluation.failure_analysis import failure_analysis


def main():
    # ---------------- Data ----------------
    print("=== Building data loaders ===")
    train_loader_plain, val_loader, test_loader = get_cifar10_loaders(randaugment=False)
    train_loader_ra, _, _ = get_cifar10_loaders(randaugment=True)
    train_unaug_loader = get_cifar10_unaugmented_train_loader()

    near_loader, cifar100_classes = get_cifar100_unknown_loader(NEAR_CLASSES)
    far_loader, _ = get_cifar100_unknown_loader(FAR_CLASSES)

    # ---------------- Step 1: Vanilla ----------------
    print("\n=== Step 1: Vanilla closed-set baseline ===")
    vanilla_ckpt = os.path.join(CKPT_DIR, "vanilla_best.pt")
    vanilla_model = ResNet18Cifar(num_classes=10, num_dummy=0)
    vanilla_val_acc, _ = train_classifier(
        vanilla_model, train_loader_plain, val_loader,
        epochs=NUM_EPOCHS_MAIN, base_lr=LR_MAIN, ckpt_path=vanilla_ckpt, tag="vanilla"
    )
    vanilla_test_acc = evaluate_accuracy(vanilla_model, test_loader)
    print(f"Vanilla: best val_acc={vanilla_val_acc:.4f}, test_acc(CSA)={vanilla_test_acc:.4f}")

    # ---------------- Extract Vanilla outputs everywhere ----------------
    print("\n=== Extracting Vanilla outputs (train/val/test/near/far) ===")
    van_train = extract_and_cache(vanilla_model, train_unaug_loader, "vanilla_train_unaug")
    van_val = extract_and_cache(vanilla_model, val_loader, "vanilla_val")
    van_test = extract_and_cache(vanilla_model, test_loader, "vanilla_test")
    van_near = extract_and_cache(vanilla_model, near_loader, "vanilla_near")
    van_far = extract_and_cache(vanilla_model, far_loader, "vanilla_far")

    # ---------------- Step 2: Post-hoc scores on frozen Vanilla ----------------
    print("\n=== Step 2: Post-hoc novelty scores (MSP/MLS/Energy/Mahalanobis) ===")
    means, var = fit_mahalanobis(van_train["feat"], van_train["labels"])

    def all_scores(d):
        return {
            "MSP": score_msp(d["logits"]),
            "MLS": score_mls(d["logits"]),
            "Energy": score_energy(d["logits"]),
            "Mahalanobis": score_mahalanobis(d["feat"], means, var),
        }

    s_val, s_test, s_near, s_far = all_scores(van_val), all_scores(van_test), all_scores(van_near), all_scores(van_far)

    table1_rows = []
    for name in ["MSP", "MLS", "Energy", "Mahalanobis"]:
        metrics = eval_score_vs_group(s_val[name], s_test[name], s_near[name], s_far[name])
        metrics["score"] = name
        table1_rows.append(metrics)
    table1 = pd.DataFrame(table1_rows).set_index("score")
    print("\nTABLE 1 — post-hoc scores on frozen Vanilla model:")
    print(table1)
    table1.to_csv(os.path.join(RESULTS_DIR, "table1_posthoc_scores.csv"))

    # ---------------- Step 3: GCSC ----------------
    print("\n=== Step 3: GCSC (RandAugment) + MLS ===")
    gcsc_ckpt = os.path.join(CKPT_DIR, "gcsc_best.pt")
    gcsc_model = ResNet18Cifar(num_classes=10, num_dummy=0)
    gcsc_val_acc, _ = train_classifier(
        gcsc_model, train_loader_ra, val_loader,
        epochs=NUM_EPOCHS_MAIN, base_lr=LR_MAIN, ckpt_path=gcsc_ckpt, tag="gcsc"
    )
    gcsc_test_acc = evaluate_accuracy(gcsc_model, test_loader)
    print(f"GCSC: best val_acc={gcsc_val_acc:.4f}, test_acc(CSA)={gcsc_test_acc:.4f}")

    gcsc_val = extract_and_cache(gcsc_model, val_loader, "gcsc_val")
    gcsc_test = extract_and_cache(gcsc_model, test_loader, "gcsc_test")
    gcsc_near = extract_and_cache(gcsc_model, near_loader, "gcsc_near")
    gcsc_far = extract_and_cache(gcsc_model, far_loader, "gcsc_far")

    gcsc_mls = {
        "val": score_mls(gcsc_val["logits"]), "test": score_mls(gcsc_test["logits"]),
        "near": score_mls(gcsc_near["logits"]), "far": score_mls(gcsc_far["logits"]),
    }

    # ---------------- Step 4: PROSER ----------------
    print("\n=== Step 4: PROSER (classifier + data placeholders) ===")
    proser_ckpt = os.path.join(CKPT_DIR, "proser_best.pt")
    proser_model, proser_val_acc, _ = train_proser(
        vanilla_ckpt, train_loader_plain, val_loader,
        epochs=NUM_EPOCHS_PROSER, base_lr=LR_PROSER,
        num_dummy=NUM_DUMMY, ckpt_path=proser_ckpt, tag="proser"
    )
    proser_test_acc = evaluate_accuracy(proser_model, test_loader)  # known-logits only
    print(f"PROSER: best val_acc={proser_val_acc:.4f}, test_acc(CSA)={proser_test_acc:.4f}")

    proser_val = extract_and_cache(proser_model, val_loader, "proser_val", has_dummy=True)
    proser_test = extract_and_cache(proser_model, test_loader, "proser_test", has_dummy=True)
    proser_near = extract_and_cache(proser_model, near_loader, "proser_near", has_dummy=True)
    proser_far = extract_and_cache(proser_model, far_loader, "proser_far", has_dummy=True)

    proser_mls = {
        "val": score_mls(proser_val["logits"]), "test": score_mls(proser_test["logits"]),
        "near": score_mls(proser_near["logits"]), "far": score_mls(proser_far["logits"]),
    }
    proser_ph = {
        "val": score_proser(proser_val["logits"], proser_val["dummy"]),
        "test": score_proser(proser_test["logits"], proser_test["dummy"]),
        "near": score_proser(proser_near["logits"], proser_near["dummy"]),
        "far": score_proser(proser_far["logits"], proser_far["dummy"]),
    }

    # ---------------- Step 5 (optional): RPL ----------------
    rpl_rows = []
    if RUN_RPL:
        print("\n=== Step 5 (optional): RPL ===")
        rpl_ckpt = os.path.join(CKPT_DIR, "rpl_best.pt")
        rpl_backbone, rpl_module = train_rpl(
            train_loader_plain, val_loader, epochs=NUM_EPOCHS_MAIN,
            base_lr=LR_MAIN, ckpt_path=rpl_ckpt, tag="rpl"
        )
        d2_val, _ = rpl_extract(rpl_backbone, rpl_module, val_loader)
        d2_test, labels_test = rpl_extract(rpl_backbone, rpl_module, test_loader)
        d2_near, _ = rpl_extract(rpl_backbone, rpl_module, near_loader)
        d2_far, _ = rpl_extract(rpl_backbone, rpl_module, far_loader)

        rpl_test_acc = rpl_accuracy_from_d2(d2_test, labels_test)
        u_val_rpl, u_test_rpl = rpl_score(d2_val), rpl_score(d2_test)
        u_near_rpl, u_far_rpl = rpl_score(d2_near), rpl_score(d2_far)

        metrics = eval_score_vs_group(u_val_rpl, u_test_rpl, u_near_rpl, u_far_rpl)
        metrics.update({"method": "RPL", "score": "RPL-distance", "CSA": rpl_test_acc})
        rpl_rows.append(metrics)
        print(f"RPL: test_acc(CSA)={rpl_test_acc:.4f}")

    # ---------------- Step 6: Common evaluation table (Table 2) ----------------
    print("\n=== Step 6: Vanilla vs GCSC vs PROSER (MLS common score) ===")
    table2_rows = []

    m = eval_score_vs_group(s_val["MLS"], s_test["MLS"], s_near["MLS"], s_far["MLS"])
    m.update({"method": "Vanilla", "score": "MLS", "CSA": vanilla_test_acc})
    table2_rows.append(m)

    m = eval_score_vs_group(gcsc_mls["val"], gcsc_mls["test"], gcsc_mls["near"], gcsc_mls["far"])
    m.update({"method": "GCSC", "score": "MLS", "CSA": gcsc_test_acc})
    table2_rows.append(m)

    m = eval_score_vs_group(proser_mls["val"], proser_mls["test"], proser_mls["near"], proser_mls["far"])
    m.update({"method": "PROSER", "score": "MLS", "CSA": proser_test_acc})
    table2_rows.append(m)

    m = eval_score_vs_group(proser_ph["val"], proser_ph["test"], proser_ph["near"], proser_ph["far"])
    m.update({"method": "PROSER", "score": f"placeholder ({PROSER_SCORE_MODE})", "CSA": proser_test_acc})
    table2_rows.append(m)

    table2_rows.extend(rpl_rows)

    table2 = pd.DataFrame(table2_rows).set_index(["method", "score"])
    cols = ["CSA", "AUROC_near", "AUROC_far", "AUROC_all", "tau_95pct_val",
            "test_known_acceptance", "near_rejection_rate", "far_rejection_rate",
            "FPR@95TPR_near", "FPR@95TPR_far", "FPR@95TPR_all"]
    table2 = table2[cols]
    print("\nTABLE 2 — Vanilla vs GCSC vs PROSER (+ optional RPL):")
    print(table2)
    table2.to_csv(os.path.join(RESULTS_DIR, "table2_model_comparison.csv"))

    # ---------------- Required figure: MSP / MLS / Mahalanobis ----------------
    print("\n=== Plotting MSP / MLS / Mahalanobis figure ===")
    plot_score_panels(
        u_val={"MSP": s_val["MSP"], "MLS": s_val["MLS"], "Mahalanobis": s_val["Mahalanobis"]},
        u_test_known={"MSP": s_test["MSP"], "MLS": s_test["MLS"], "Mahalanobis": s_test["Mahalanobis"]},
        u_near={"MSP": s_near["MSP"], "MLS": s_near["MLS"], "Mahalanobis": s_near["Mahalanobis"]},
        u_far={"MSP": s_far["MSP"], "MLS": s_far["MLS"], "Mahalanobis": s_far["Mahalanobis"]},
        score_names=["MSP", "MLS", "Mahalanobis"],
        save_path=os.path.join(RESULTS_DIR, "fig_score_msp_mls_mahalanobis.png"),
    )

    # ---------------- Failure analysis (Vanilla MLS threshold) ----------------
    print("\n=== Failure analysis (Vanilla MLS threshold) ===")
    tau_vanilla_mls = compute_threshold_95(s_val["MLS"])
    near_fail = failure_analysis(van_test["logits"], tau_vanilla_mls, "near",
                                  van_near["logits"], van_near["labels"],
                                  cifar100_classes, n=3, score_fn=score_mls)
    far_fail = failure_analysis(van_test["logits"], tau_vanilla_mls, "far",
                                 van_far["logits"], van_far["labels"],
                                 cifar100_classes, n=3, score_fn=score_mls)
    failures = pd.DataFrame(near_fail + far_fail)
    print(failures)
    failures.to_csv(os.path.join(RESULTS_DIR, "failure_analysis.csv"), index=False)

    # ---------------- Save a run summary ----------------
    summary = {
        "vanilla_test_acc": vanilla_test_acc,
        "gcsc_test_acc": gcsc_test_acc,
        "proser_test_acc": proser_test_acc,
        "PROSER_SCORE_MODE": PROSER_SCORE_MODE,
        "PLOT_STYLE": PLOT_STYLE,
        "RUN_RPL": RUN_RPL,
    }
    with open(os.path.join(RESULTS_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE. Results saved under:", RESULTS_DIR, "===")
    return table1, table2, failures


if __name__ == "__main__":
    table1, table2, failures = main()