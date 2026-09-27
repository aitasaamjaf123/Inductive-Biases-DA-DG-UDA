# Task 4 — Open-Set Recognition (OSR)

CIFAR-10 as the ten known classes; fixed CIFAR-100 near/far test classes as
unknowns, evaluated only after every checkpoint and score is frozen.

## Directory structure

```
task4/
  configs/config.py            # seeds, paths, hyperparameters, decision switches
  data/
    make_splits.py             # stratified 90/10 CIFAR-10 train/val split (seed 6304)
    cifar10.py                 # dataset wrapper, transforms, loaders
    cifar100_unknowns.py       # fixed near/far unknown-class loaders (eval-only)
  models/resnet_cifar.py       # CIFAR-appropriate ResNet-18 (3x3 stem, no maxpool)
  methods/
    vanilla.py                 # Step 1 — plain closed-set classifier
    gcsc.py                    # Step 3 — same recipe + RandAugment
    manifold_mixup.py          # PROSER data-placeholder loss
    proser.py                  # Step 4 — classifier + data placeholders
    rpl.py                     # Step 5 (optional) — Reciprocal Point Learning
  scores/
    msp.py mls.py energy.py mahalanobis.py proser_score.py
  evaluation/
    thresholds.py              # cosine LR schedule + closed-set accuracy
    metrics.py                 # AUROC + validation-calibrated rejection metrics
    failure_analysis.py        # incorrectly-accepted-unknown inspection
    evaluate_osr.py            # required score-distribution / ROC figure
  train.py                     # generic supervised training loop (Vanilla/GCSC)
  extract_outputs.py           # feature/logit extraction + caching
  scripts/run_task4.py         # runs Steps 1–6 end to end
  cache/                       # saved features/logits (.npz) — gitignored
  results/                     # tables, figures, run_summary.json
```

## Setup

```bash
cd task4
pip install -r requirements.txt
```

By default all paths (`DATA_ROOT`, `CKPT_DIR`, `CACHE_DIR`, `RESULTS_DIR`,
`SPLIT_DIR`) in `configs/config.py` point at `/kaggle/working/...`. If you
are not on Kaggle, edit those five constants to local paths (e.g. `./data`,
`./checkpoints`, `./cache`, `./results`, `./splits`) before running anything.
CIFAR-10/CIFAR-100 are downloaded automatically via `torchvision` on first
use (`DOWNLOAD=True`); set `USE_KAGGLE_LOCAL_DATA=True` and point `DATA_ROOT`
at a pre-extracted dataset folder to run offline.

## Running

From inside `task4/`:

```bash
python scripts/run_task4.py
```

This runs, in order:
1. **Vanilla** — trains the plain 10-class ResNet-18 (100 epochs, SGD,
   cosine decay), reports CIFAR-10 test accuracy (CSA), and caches its
   features/logits on train/val/test/near/far.
2. **Post-hoc scores** — computes MSP, MLS, Energy, and Mahalanobis on the
   frozen Vanilla outputs and writes `results/table1_posthoc_scores.csv`.
3. **GCSC** — retrains with RandAugment inserted after crop+flip, evaluated
   with MLS.
4. **PROSER** — fine-tunes from the Vanilla checkpoint with 5 dummy
   classifiers, the classifier-placeholder loss (β=1), and the
   manifold-mixup data-placeholder loss (γ=0.1, mixed after layer2/before
   layer3). Evaluated with both MLS and its own placeholder score.
5. **RPL (optional)** — only runs if `RUN_RPL = True` in `configs/config.py`.
6. **Common evaluation** — writes `results/table2_model_comparison.csv`
   (Vanilla/GCSC/PROSER [+RPL] × CSA/AUROC/rejection metrics), the required
   MSP/MLS/Mahalanobis figure (`results/fig_score_msp_mls_mahalanobis.png`),
   and `results/failure_analysis.csv` (incorrectly accepted near/far
   unknowns under the Vanilla-MLS threshold).

Checkpoints are cached under `CKPT_DIR`; re-running the script skips
training any model whose checkpoint already exists
(`SKIP_IF_CHECKPOINT_EXISTS = True`), so a run can be safely resumed across
sessions. Cached features/logits under `CACHE_DIR` are reused the same way.

## Decision switches (`configs/config.py`)

The assignment leaves a few implementation choices open; these are
resolved by explicit constants rather than being hard-coded, so you can flip
one value and re-run only the affected (cheap, cache-reusing) part of the
pipeline:

- `RUN_RPL` — whether to run the optional Step 5 extension.
- `PLOT_STYLE` — `"hist"`, `"roc"`, or `"both"` for the required Step-6 figure.
- `PROSER_SCORE_MODE` — `"dummy_minus_known"` (logit margin) or
  `"dummy_softmax"` (normalized probability mass on dummy logits) for
  PROSER's placeholder-based detection score.

## Notes

- No CIFAR-100 image is used anywhere except the two fixed evaluation
  loaders in `data/cifar100_unknowns.py`; those are only ever called after
  training, checkpoint selection, and score/threshold definitions are fixed.
- `train.py` is intentionally generic (used by both `methods/vanilla.py` and
  `methods/gcsc.py`); the only difference between the two methods is which
  loader (`randaugment=False` vs `True`) is passed in.
- Reported CSA for PROSER always uses the 10 known-class logits only, so
  closed-set classification and rejection remain measured separately.