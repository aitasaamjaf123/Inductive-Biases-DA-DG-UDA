# ══════════════════════════════════════════════════════════════════════════
# task2/evaluation/umap_visualization.py — OPTIONAL UMAP alignment plots
# ══════════════════════════════════════════════════════════════════════════
# Not required by the assignment's Suggested Repository Structure — kept as a
# convenience script for the representation-alignment visualizations
# referenced informally when discussing DAN/DANN/CDAN. Loads checkpoints from
# disk; does NOT retrain anything. Requires: pip install umap-learn
#
# Usage (after run_task2.py has completed at least once):
#     python -m task2.evaluation.umap_visualization
#
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from shared.pacs_protocol import CONFIG, SEED, DEVICE
from shared.pacs import EVAL_TRANSFORM, make_loader
from task2.models.backbone import ResNet18Backbone
from task2.evaluation.domain_separability import extract_features
from task2.train import prepare_data

# ══════════════════════════════════════════════════════════════════════════
# NEW CELL — UMAP alignment plots (baseline + each model) + class histogram
# Run AFTER the main pipeline. Loads checkpoints from disk, no retraining.
# ══════════════════════════════════════════════════════════════════════════
import umap
from matplotlib.lines import Line2D

PLOT_DIR = os.path.join(CONFIG["output_root"], "results", "umap_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_SRC_PER_DOMAIN = 300          # random source images per source domain
CLASSES = CONFIG["classes"]
NUM_CLASSES = len(CLASSES)
RNG = np.random.RandomState(SEED)

source_splits, target_samples = prepare_data()


def sample_paths(samples, n, rng):
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    return [samples[i] for i in idx]


def build_umap_set(source_splits, target_samples):
    """Random source-train images from each source domain + ALL target images."""
    src_subset = []
    for dom in CONFIG["source_domains"]:
        for p, y in sample_paths(source_splits[dom]["train"], N_SRC_PER_DOMAIN, RNG):
            src_subset.append((p, y, dom))
    tgt_subset = [(p, y, "target") for p, y in target_samples]
    return src_subset, tgt_subset


def features_for(backbone, src_subset, tgt_subset):
    src_loader = make_loader([(p, y) for p, y, _ in src_subset], EVAL_TRANSFORM,
                             64, shuffle=False, name="umap-src")
    tgt_loader = make_loader([(p, y) for p, y, _ in tgt_subset], EVAL_TRANSFORM,
                             64, shuffle=False, name="umap-tgt")
    f_s, y_s = extract_features(backbone, src_loader, DEVICE)
    f_t, y_t = extract_features(backbone, tgt_loader, DEVICE)
    feats = np.concatenate([f_s, f_t], axis=0)
    labels = np.concatenate([y_s, y_t], axis=0)
    domains = np.concatenate([[d for _, _, d in src_subset],
                              [d for _, _, d in tgt_subset]], axis=0)
    return feats, labels, domains


def umap_2d(feats, seed=SEED):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=seed)
    return reducer.fit_transform(feats)


def plot_alignment(emb, labels, domains, title, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=14)

    # Left: colored by DOMAIN (is alignment happening?)
    domain_styles = [("photo", "o"), ("art_painting", "s"),
                     ("cartoon", "^"), ("target", "x")]
    for dom, mk in domain_styles:
        m = domains == dom
        axes[0].scatter(emb[m, 0], emb[m, 1], s=8, marker=mk, alpha=0.5, label=dom)
    axes[0].set_title("colored by domain")
    axes[0].legend(markerscale=2)

    # Right: colored by CLASS, marker shows source (o) vs target (x)
    cmap = plt.get_cmap("tab10")
    for c in range(NUM_CLASSES):
        for is_tgt, mk in [(False, "o"), (True, "x")]:
            m = (labels == c) & ((domains == "target") == is_tgt)
            axes[1].scatter(emb[m, 0], emb[m, 1], s=8, marker=mk,
                            color=cmap(c), alpha=0.6)
    handles = [Line2D([], [], marker="o", color=cmap(c), ls="", label=CLASSES[c])
               for c in range(NUM_CLASSES)]
    handles += [Line2D([], [], marker="o", color="gray", ls="", label="source"),
                Line2D([], [], marker="x", color="gray", ls="", label="target")]
    axes[1].legend(handles=handles, markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].set_title("colored by class (o = source, x = target)")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path}")


def load_backbone(run_name):
    path = os.path.join(CONFIG["output_root"], "checkpoints", f"{run_name}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    bb = ResNet18Backbone().to(DEVICE)
    bb.load_state_dict(ckpt["backbone"])
    bb.eval()
    return bb


src_subset, tgt_subset = build_umap_set(source_splits, target_samples)
print(f"UMAP set: {len(src_subset)} source images, {len(tgt_subset)} target images")

# ── 1. BASELINE: pretrained ResNet-18, no PACS training ──
baseline_bb = ResNet18Backbone().to(DEVICE)
baseline_bb.eval()
feats, labels, domains = features_for(baseline_bb, src_subset, tgt_subset)
emb = umap_2d(feats)
plot_alignment(emb, labels, domains,
               "BASELINE (ImageNet-pretrained, no adaptation): source vs target features",
               "umap_baseline_pretrained.png")

# ── 2. EACH MODEL ──
for run_name in ["source_only", "dan", "dann", "cdan"]:
    bb = load_backbone(run_name)
    feats, labels, domains = features_for(bb, src_subset, tgt_subset)
    emb = umap_2d(feats)
    plot_alignment(emb, labels, domains,
                   f"{run_name}: source vs target features (UMAP)",
                   f"umap_{run_name}.png")

# ── 3. CLASS BALANCE: training (source) vs test (target) histogram ──
src_labels = [y for dom in CONFIG["source_domains"] for _, y in source_splits[dom]["train"]]
tgt_labels = [y for _, y in target_samples]
src_counts = np.bincount(src_labels, minlength=NUM_CLASSES)
tgt_counts = np.bincount(tgt_labels, minlength=NUM_CLASSES)

print("Source train counts:", dict(zip(CLASSES, src_counts.tolist())))
print("Target counts:      ", dict(zip(CLASSES, tgt_counts.tolist())))

x = np.arange(NUM_CLASSES)
w = 0.4
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w / 2, src_counts / src_counts.sum(), w, label="source train")
ax.bar(x + w / 2, tgt_counts / tgt_counts.sum(), w, label="target (sketch)")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylabel("fraction of samples")
ax.set_title("Class distribution: source training features vs target features")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "class_balance_histogram.png"), dpi=150)
plt.show()