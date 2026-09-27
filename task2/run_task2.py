# ══════════════════════════════════════════════════════════════════════════
# task2/run_task2.py — entry point: run the whole Task 2 pipeline end to end
# ══════════════════════════════════════════════════════════════════════════
# Usage (from the repository root, so the `shared` and `task2` packages are
# importable):
#     python -m task2.run_task2
#
from shared.pacs_protocol import CONFIG, DEVICE
from task2.train import run_all_main_methods
from task2.evaluate_final import final_target_and_source_eval, plot_training_curves, controlled_design_study


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run everything (paste this as the last Kaggle cell)
# ══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Using device: {DEVICE}")
    results, source_splits, target_samples = run_all_main_methods()
    comparison_df, class_reports = final_target_and_source_eval(results, source_splits, target_samples)
    plot_training_curves(results)
    controlled_df = controlled_design_study(source_splits, target_samples)
    print("Done. Outputs are under:", CONFIG["output_root"])

if __name__ == "__main__":
    main()
