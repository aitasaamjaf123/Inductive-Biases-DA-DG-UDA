# ══════════════════════════════════════════════════════════════════════════
# task2/evaluation/plot_summary.py — OPTIONAL extra plots from saved results
# ══════════════════════════════════════════════════════════════════════════
# Not required by the assignment's Suggested Repository Structure — kept as a
# convenience script. Reads main_comparison_table.csv and per_class_shift.json
# that task2/evaluate_final.py already saved; does NOT retrain anything.
#
# Usage (after run_task2.py has completed at least once):
#     python -m task2.evaluation.plot_summary
#
import os
from shared.pacs_protocol import CONFIG

# ══════════════════════════════════════════════════════════════════════════
# NEW CELL — paste at the end. Reads files already saved by your last run
# (main_comparison_table.csv, per_class_shift.json). No retraining required.
# ══════════════════════════════════════════════════════════════════════════

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results_dir = os.path.join(CONFIG["output_root"], "results")
comparison_df = pd.read_csv(os.path.join(results_dir, "main_comparison_table.csv"), index_col="method")

with open(os.path.join(results_dir, "per_class_shift.json")) as f:
    class_reports = json.load(f)

# ── Plot A: source-val accuracy vs. target accuracy vs. domain separability,
#    per method, side by side. This is the "test performance" summary you
#    asked for — cheap because target_acc/domain_separability were already
#    computed once at the end of your run and are just sitting in the CSV. ──
methods = list(comparison_df.index)
x = np.arange(len(methods))
width = 0.25

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(x - width, comparison_df["mean_src_val_acc"], width, label="mean source-val acc")
ax1.bar(x, comparison_df["target_acc"], width, label="target acc (oracle, eval-only)")
ax1.set_ylabel("accuracy")
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim(0, 1)
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(x, comparison_df["domain_separability"], "o--", color="black",
         label="domain separability (post-hoc probe)")
ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1)  # chance level
ax2.set_ylabel("domain separability (0.5 = fully confused)")
ax2.set_ylim(0.4, 1.05)
ax2.legend(loc="upper right")

plt.title("Source vs. target accuracy vs. domain separability, per method")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "summary_accuracy_vs_separability.png"), dpi=150)
plt.show()

# ── Plot B: per-class accuracy delta (method - source_only) for each
#    adapted method. This is the diagnostic plot for "did the model collapse
#    to predicting one or two classes on target?" — directly relevant given
#    cdan's target_acc of 0.07 (below chance for ~7 classes). ──
fig, axes = plt.subplots(1, len(class_reports), figsize=(6 * len(class_reports), 5), sharey=True)
if len(class_reports) == 1:
    axes = [axes]

for ax, (method_name, report) in zip(axes, class_reports.items()):
    deltas = report["per_class_delta"]
    classes = list(deltas.keys())
    vals = [deltas[c] for c in classes]
    colors = ["tab:red" if v < 0 else "tab:green" for v in vals]
    ax.barh(classes, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{method_name} vs source_only\nper-class accuracy delta on target")
    ax.set_xlabel("accuracy delta")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "per_class_delta.png"), dpi=150)
plt.show()

# ── Print the most-degraded classes with their dominant confusion, for the
#    method(s) that collapsed — this tells you WHAT the model is doing
#    wrong on target (e.g. "everything predicted as class X"). ──
for method_name, report in class_reports.items():
    print(f"\n=== {method_name}: most degraded classes vs source_only ===")
    for entry in report["most_degraded"]:
        conf = entry["dominant_confusion"]
        conf_str = f"confused with '{conf[0]}' ({conf[1]} times)" if conf else "no dominant confusion"
        print(f"  {entry['class']:>15s}  delta={entry['delta']:+.3f}   {conf_str}")