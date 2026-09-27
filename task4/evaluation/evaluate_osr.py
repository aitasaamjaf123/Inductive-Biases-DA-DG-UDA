"""
Task 4 — required compact figure: score distributions and/or ROC curves for
MSP, MLS, and Mahalanobis, with the 95th-percentile validation threshold
marked on the histogram view.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from configs.config import PLOT_STYLE
from evaluation.metrics import compute_threshold_95


def plot_score_panels(u_val, u_test_known, u_near, u_far, score_names, save_path):
    n = len(score_names)
    rows = 2 if PLOT_STYLE == "both" else 1
    fig, axes = plt.subplots(rows, n, figsize=(5 * n, 4 * rows))
    if rows == 1:
        axes = np.array([axes])

    for j, name in enumerate(score_names):
        vk, tk, un, uf = u_val[name], u_test_known[name], u_near[name], u_far[name]
        if PLOT_STYLE in ("hist", "both"):
            ax = axes[0, j]
            bins = 40
            ax.hist(tk, bins=bins, alpha=0.5, density=True, label="Known (test)")
            ax.hist(un, bins=bins, alpha=0.5, density=True, label="Near unknown")
            ax.hist(uf, bins=bins, alpha=0.5, density=True, label="Far unknown")
            tau = compute_threshold_95(vk)
            ax.axvline(tau, color="k", linestyle="--", label="tau (95% val)")
            ax.set_title(f"{name}: score distribution")
            ax.legend(fontsize=8)
        if PLOT_STYLE in ("roc", "both"):
            ax = axes[1, j] if PLOT_STYLE == "both" else axes[0, j]
            for u_unk, lbl in [(un, "near"), (uf, "far"), (np.concatenate([un, uf]), "all")]:
                y_true = np.concatenate([np.zeros_like(tk), np.ones_like(u_unk)])
                y_score = np.concatenate([tk, u_unk])
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc = roc_auc_score(y_true, y_score)
                ax.plot(fpr, tpr, label=f"{lbl} (AUROC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
            ax.set_title(f"{name}: ROC")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved figure to {save_path}")