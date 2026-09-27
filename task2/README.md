# Task 2 — Unsupervised Domain Adaptation (PACS, Sketch as target)

This directory implements PA1 Task 2 exactly as specified: Source-only ERM,
DAN (MMD), DANN (adversarial), and CDAN (class-conditional adversarial),
all sharing one training/evaluation pipeline, frozen BatchNorm running
statistics, and seed 6304 throughout.

The code here is a direct, mechanical split of the original single-notebook
implementation (`task2.ipynb`) into the file layout the assignment's
"Suggested Repository Structure" describes. **No logic was changed** —
only cross-file imports were added where a function in one file now needs
a symbol defined in another.

## Directory layout

```
shared/
    pacs_protocol.py     # CONFIG dict, set_seed(), get_device(), DEVICE
    pacs.py               # PACSDataset, TRAIN/EVAL transforms, split builders, loaders
    splits/               # split-cache JSONs get written here at runtime

task2/
    configs/              # YAML records of the hyperparameters below (see note)
        base.yaml
        source_only.yaml
        dan.yaml
        dann.yaml
        cdan.yaml
    models/
        backbone.py               # ResNet18Backbone, set_bn_eval, freeze_batchnorm_running_stats
        classifier_head.py        # ClassifierHead (linear)
        domain_discriminator.py   # GradientReversalLayer/Function, grl_alpha_schedule, DomainDiscriminator
    methods/
        dan.py           # multi_kernel_rbf_mmd2()  (real DAN-specific code)
        cdan.py           # multilinear_map()  (real CDAN-specific code)
        dann.py           # thin re-export + pointer to where DANN's pieces live
        source_only.py    # thin pointer (no method-specific loss term to add)
    evaluation/
        metrics.py                # evaluate_classifier, mean_source_val_macro_f1
        domain_separability.py    # extract_features, domain_separability_score
        class_analysis.py         # per_class_accuracy, class_shift_report
        plot_summary.py           # OPTIONAL bonus — post-hoc plots from saved CSV/JSON
        umap_visualization.py     # OPTIONAL bonus — UMAP alignment plots
    train.py               # generic train_method() used by all 4 methods; prepare_data(); run_all_main_methods()
    evaluate_final.py       # final comparison table, training-curve plots, controlled design study (Step 6)
    run_task2.py            # entry point — runs the whole Task 2 pipeline end to end
    results/                # CSV/JSON/PNG outputs land here (checkpoints/logs land under CONFIG["output_root"])
```

## Why some files aren't a literal 1:1 match to the assignment's list

The assignment's suggested `methods/` folder lists one file per method
(`source_only.py dan.py dann.py cdan.py`). The working implementation trains
all four methods through **one generic loop**, `train_method()` in
`task2/train.py`, which is what the assignment's own Task 2 setup section
requires: *"all methods must run through the same training and evaluation
pipeline."* Splitting that loop into four near-duplicate per-method files
would mean four copies of the same data loading / early stopping /
checkpointing code (and a real risk of them drifting out of sync), so:

- `methods/dan.py` and `methods/cdan.py` hold the actual method-specific
  math (`multi_kernel_rbf_mmd2`, `multilinear_map`) that plugs into the
  shared loop.
- DANN's method-specific pieces (gradient-reversal layer, alpha schedule,
  domain discriminator) live in `models/domain_discriminator.py`, since
  they're an `nn.Module`/model component, not a standalone loss function.
  `methods/dann.py` re-exports them with a docstring pointing back.
- `methods/source_only.py` is a docstring pointer, since source-only ERM
  adds no alignment term at all — it's just the `method == "source_only"`
  branch of the shared loop.
- `configs/*.yaml` are **documentation records** of the hyperparameters
  used for each method, not something the code parses at runtime — the
  actual runtime configuration is the `CONFIG` dict in
  `shared/pacs_protocol.py`. Keeping the working `CONFIG`-dict-driven code
  as-is (rather than bolting on a YAML loader) avoids touching tested code,
  per the instruction to keep everything unchanged. If you want the YAML
  files to actually drive the run, wire a small loader into
  `train.py:train_method()` / `run_task2.py`.

## How to run

From the **repository root** (so `shared` and `task2` are importable as
packages — both already have `__init__.py`):

```bash
# 1. Edit the data path
#    shared/pacs_protocol.py -> CONFIG["data_root"] should point at your
#    local PACS folder (photo/, art_painting/, cartoon/, sketch/, each with
#    one subfolder per class).

# 2. Run the full pipeline: trains Source-only, DAN, DANN, CDAN; produces the
#    final comparison table, training curves, and the controlled design study.
python -m task2.run_task2
```

This is equivalent to running the original notebook's cells in order; the
entry point (`if __name__ == "__main__":`) is unchanged, just moved into
`run_task2.py` and wrapped in `main()` so it can also be imported.

Outputs land under `CONFIG["output_root"]` (defaults to
`/kaggle/working/task2_outputs`; change this in `shared/pacs_protocol.py`
for a local run), specifically:
- `checkpoints/<run_name>.pt` — best checkpoint per method (by mean
  source-validation macro-F1)
- `logs/<run_name>_history.json` — per-epoch loss/F1 history
- `splits/pacs_sketch_seed6304_sources.json`, `..._target.json` — the
  cached stratified 80/20 source splits and the full target sample list,
  seed 6304 (reuse these for Task 3, since it must use the identical
  splits)
- `results/main_comparison_table.csv`, `results/per_class_shift.json`,
  `results/training_curves.png`, `results/controlled_study_<method>.csv`

### Optional bonus scripts

These are not part of the assignment's required file list, but were in the
original notebook as extra analysis cells. Run them **after**
`run_task2.py` has produced results at least once — they don't retrain
anything, they just read saved checkpoints/CSVs:

```bash
python -m task2.evaluation.plot_summary          # extra accuracy/separability + per-class-delta plots
pip install umap-learn
python -m task2.evaluation.umap_visualization    # UMAP source/target alignment plots
```

## Reproducibility notes

- Seed 6304 is used everywhere the assignment requires it (splits,
  classifier-head training, checkpoint comparison, domain-separability
  probe).
- BatchNorm running statistics are frozen for every method
  (`set_bn_eval()` is called every epoch, right after `model.train()`, per
  the assignment's explicit instruction not to put the whole model in
  `eval()` mode).
- The **Source-only** checkpoint trained here (`checkpoints/source_only.pt`)
  is the one Task 3 must reuse unmodified as its ERM baseline — do not
  retrain it there.
