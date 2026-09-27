# Task 3 — Domain Generalization (PACS, Sketch unseen at training time)

This directory implements PA1 Task 3: ERM, DAN-DG (pairwise source-domain
MMD alignment, no target access), and SAM (sharpness-aware minimization),
compared on three observed source domains (Photo, Art Painting, Cartoon)
and finally evaluated once on the held-out Sketch domain.

Like Task 2, this is a **mechanical split** of the original single-notebook
implementation (`task3.ipynb`, itself already written with `[file/path.py]`
tags matching the assignment's suggested layout) into real files. **No
logic was changed** — only cross-file imports were added.

## Directory layout

```
task3/
    configs/                          # YAML records of the hyperparameters below (see note)
        erm.yaml
        dan_dg.yaml
        sam.yaml
    shared/                            # Task-3-local copies of "shared"-tagged code — see note below
        pacs_protocol.py               # CONFIG dict, set_seed(), DEVICE
        pacs.py                        # PACSDataset, transforms, split/loader builders
    models/
        backbone.py                    # PACSModel (backbone+head combined), freeze_bn, load_task2_checkpoint
        classifier_head.py             # thin pointer — see note below
    methods/
        erm.py                         # get_erm_model (loads real Task2 ckpt) + train_erm_from_scratch
        dan_dg.py                      # multi_kernel_rbf / mmd2 / pairwise_source_mmd + train_dan_dg
        sam.py                         # SAM optimizer (Foret et al.) + train_sam
    selection/
        source_validation.py           # EarlyStopper (patience on mean source macro-F1)
    evaluation/
        domain_metrics.py              # evaluate_domain, evaluate_all_sources
        source_domain_separability.py  # 3-class logistic-regression probe (Photo/Art/Cartoon)
        sharpness.py                   # build_fixed_sharpness_batch, sharpness_proxy (Delta_sharp)
    evaluate_sketch.py                 # THE ONLY place Sketch labels are used — final eval only
    train.py                           # main() orchestration + run_controlled_study() + plot_training_curves()
    run_task3.py                       # entry point — runs the whole Task 3 pipeline end to end
    results/                           # CSV/JSON/PNG outputs land here
```

## Two deliberate deviations from the assignment's literal file list

**1. `task3/shared/` instead of reusing the top-level `shared/` from Task 2.**
The assignment's suggested layout has Task 2 and Task 3 share one
`shared/pacs.py` + `shared/pacs_protocol.py`, and Task 3's own text says to
"reuse the shared PACS data and split code from Task 2." In the code you
gave me, the two notebooks were written independently and use **different
CONFIG schemas and a different dataset class**:

| | Task 2 (`shared/`) | Task 3 (`task3/shared/`) |
|---|---|---|
| data root key | `CONFIG["data_root"]` | `CONFIG["PACS_ROOT"]` |
| source domains key | `CONFIG["source_domains"]` | `CONFIG["SOURCE_DOMAINS"]` |
| dataset class | `PACSDataset` (retries on a corrupt file, folder-name auto-detection) | `PACSDataset` (simpler, no retry/auto-detect logic) |
| transforms | module-level `TRAIN_TRANSFORM` / `EVAL_TRANSFORM` | `build_transforms(cfg)` function |

Forcing these into one literal `shared/` would require rewriting one (or
both) implementations, which the instruction for this pass was explicitly
to avoid ("keep the code the same"). So Task 3 keeps its own
`task3/shared/pacs_protocol.py` / `task3/shared/pacs.py`, functionally
identical to what the notebook already had.

**What this means for you in practice:** the assignment requires Task 3 to
reuse Task 2's *exact* stratified 80/20 source splits (seed 6304). Point
`CONFIG["SPLITS_JSON_PATH"]` in `task3/shared/pacs_protocol.py` at the
`pacs_sketch_seed6304_sources.json` file that Task 2's run already wrote
under `shared/splits/` (or `CONFIG["output_root"]/splits/` from Task 2's
run) — `build_or_load_splits()` will load it directly instead of
regenerating a split that might not match bit-for-bit.

**2. `models/backbone.py` and `models/classifier_head.py` are not two
independent classes.** Task 3 fully fine-tunes the whole ResNet-18 (unlike
Task 2's frozen-backbone-plus-linear-head setup), so there's no natural
code seam between "backbone parameters" and "head parameters" — they're one
`nn.Module`, `PACSModel`, defined in `models/backbone.py`.
`models/classifier_head.py` re-exports `PACSModel` with a docstring
pointing back, so the file exists (matching the suggested layout) without
duplicating or diverging from the working class.

**3. `methods/erm.py` has two functions, and `train.py`'s `main()` uses the
one that does *not* match the letter of the assignment.** The assignment
requires the Task 3 ERM baseline to be *the same checkpoint as Task 2's
Source-only model, reused unmodified* — this is `get_erm_model()` /
`load_task2_checkpoint()`, requiring `CONFIG["TASK2_CHECKPOINT_PATH"]` to
point at a real Task 2 `source_only.pt`. The original notebook also
contains a fallback, `train_erm_from_scratch()`, for when no such
checkpoint path is available, and `main()` calls that fallback by default.
**Before you finalize Task 3 results**, decide which one you actually want
to report and edit `train.py:main()` accordingly — the assignment's grading
rubric expects the reused-checkpoint version.

## How to run

From the **repository root** (so `task3` is importable — it already has
`__init__.py` throughout):

```bash
# 1. Edit task3/shared/pacs_protocol.py -> CONFIG:
#      CONFIG["PACS_ROOT"]              # your local PACS folder
#      CONFIG["TASK2_CHECKPOINT_PATH"]  # path to Task 2's source_only.pt (see note above)
#      CONFIG["SPLITS_JSON_PATH"]       # path to Task 2's split JSON (see note above)
#      CONFIG["CONTROLLED_STUDY_METHOD"]  # "dan_dg" or "sam" — pick ONE per the assignment

# 2. Run the full pipeline (staged; see CONFIG's RUN_* switches to
#    skip/re-run individual stages without retraining everything):
python -m task3.run_task3
```

Outputs land under `CONFIG["OUTPUT_DIR"]` (defaults to
`/kaggle/working/task3_outputs`; change this for a local run):
- `checkpoints/<tag>.pth` — best checkpoint per method/sweep run
- `results/<tag>_history.json` — per-epoch loss/F1 history
- `results/main_comparison.json` — ERM/DAN-DG/SAM source-validation +
  separability + sharpness table
- `results/controlled_study.json` — the Step 5 bounded sweep
- `results/sketch_final_results.json` — **the only file containing Sketch
  results**, written only once `CONFIG["RUN_FINAL_SKETCH_EVAL"]` runs
- `plots/*_curves.png` — training-curve figures

## No-target-leakage discipline

Sketch images/labels are never loaded until `evaluate_sketch.py` is called
from the final stage of `train.py:main()` (gated by
`CONFIG["RUN_FINAL_SKETCH_EVAL"]`). Every training run, the early-stopping
checkpoint selection, the domain-separability probe, and the sharpness
proxy all operate on source (Photo/Art Painting/Cartoon) data only — this
mirrors the assignment's explicit leakage rule for Task 3 and is worth
double-checking before you write up results.
