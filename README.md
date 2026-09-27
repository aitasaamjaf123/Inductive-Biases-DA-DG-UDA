# EE-5102/CS-6304 — PA1: Beyond IID, Closed-Set Learning

Individual submission. This repository implements all four tasks of
Programming Assignment 1, studying inductive biases, unsupervised domain
adaptation, domain generalization, and open-set recognition under a shared
discipline of fixed seeds (**6304**), frozen evaluation protocols, and strict
no-leakage rules between training and held-out evaluation data.

- **Task 1 — Inductive Biases & Feature Representations.** ResNet-50, ViT-B/16,
  and CLIP ViT-B-32 compared under color, shape/texture cue-conflict,
  translation, and patch-shuffle interventions, plus representation-stability
  and t-SNE/UMAP analysis. Dataset: STL-10.
- **Task 2 — Unsupervised Domain Adaptation.** Source-only ERM, DAN, DANN, and
  CDAN on PACS (Photo/Art Painting/Cartoon → Sketch), with target images
  (unlabeled) available during adaptation.
- **Task 3 — Domain Generalization.** ERM, DAN-DG (source-only pairwise MMD),
  and SAM on the same PACS source domains, with Sketch held out entirely
  until final evaluation.
- **Task 4 — Open-Set Recognition.** CIFAR-10 known classes vs. near/far
  CIFAR-100 unknowns; MSP/MLS/Energy/Mahalanobis post-hoc scores, GCSC
  (RandAugment + MLS), and PROSER (classifier + data placeholders).

Task 5 of the assignment (Synthesis) is answered in the PDF report, not in
code — see `report/`.

## Repository layout

```
pa1-beyond-iid/
    README.md                  # this file
    requirements.txt
    .gitignore
    common/                    # utilities genuinely shared across >1 task (seeding, logging, plotting, metrics)
    task1/                     # Inductive biases & representations — see task1/README.md
    task2/                     # Unsupervised domain adaptation      — see task2/README.md
    task3/                     # Domain generalization               — see task3/README.md
    task4/                     # Open-set recognition                — see task4/README.md
    shared/                    # PACS dataset/transform/split code shared by Task 2 (see note below)
    report/
        figures/
        pa1_report.pdf         # NeurIPS-style, 8 pages main content + references
```

Each task directory has its own `README.md` with the exact commands to
reproduce that task's results, its own `configs/`, and its own `results/`.
This top-level README covers what's common across all four: setup,
reproducibility rules, and how the pieces fit together as one study.

### A note on `shared/`

The assignment's suggested layout has Task 2 and Task 3 reuse one
`shared/pacs.py` + `shared/pacs_protocol.py`. In this repo, Task 2 uses the
top-level `shared/`; Task 3 keeps a functionally equivalent but
independently-implemented copy at `task3/shared/`, because the two were
originally built with different `CONFIG` schemas and the instruction for
this pass was to preserve working code as-is rather than force a merge that
risked breaking either implementation. **Task 3 still reuses Task 2's exact
splits and ERM checkpoint** (via `SPLITS_JSON_PATH` /
`TASK2_CHECKPOINT_PATH` in `task3/shared/pacs_protocol.py`'s `CONFIG`) —
only the *code* that reads them is duplicated, not the *data* or the
*decisions*. Full detail in `task3/README.md`.

## Setup

```bash
git clone <repository-url>
cd pa1-beyond-iid
python -m venv .venv && source .venv/bin/activate   # or conda/mamba equivalent
pip install -r requirements.txt
```

Datasets are **not** committed to this repository (see `.gitignore`).
Download/prepare them per each task's README:

| Task | Dataset | Notes |
|---|---|---|
| 1 | STL-10 (or Oxford-IIIT Pets) | `torchvision.datasets` download, or local path in `task1/configs/` |
| 2 & 3 | PACS | Photo / Art Painting / Cartoon / Sketch, one subfolder per class per domain |
| 4 | CIFAR-10 + fixed CIFAR-100 classes | `torchvision.datasets`, evaluation-only for CIFAR-100 |

## Run order

Task 3 depends on Task 2's checkpoint and splits; the others are
independent. A full reproduction runs:

```bash
python -m task1.scripts.run_task1        # see task1/README.md for exact entry point
python -m task2.run_task2
python -m task3.run_task3                # after pointing task3's CONFIG at Task 2's outputs — see task3/README.md
python -m task4.evaluate_osr             # see task4/README.md for exact entry point
```

## Reproducibility & protocol discipline

These rules apply across the whole assignment, not just one task, and are
worth checking before you write up results (see also the assignment's own
"Before You Submit" checklist):

- **Seed 6304** everywhere the assignment specifies it: dataset splits,
  classifier-head/backbone training comparisons, subset selection,
  patch-shuffle permutations, domain-separability probes, and the
  sharpness proxy.
- **No target-label leakage.** Task 2's target accuracy/macro-F1 and
  per-class analysis are computed only after every checkpoint and setting
  is frozen. Task 3 never loads Sketch images or labels until the final
  evaluation stage, and Task 2's Sketch results must not be used to revise
  any Task 3 setting. Task 4 never uses CIFAR-100 images for training,
  checkpoint selection, score design, or threshold selection.
- **Fixed evaluation subsets and splits are cached to disk** (JSON) the
  first time they're built, so re-running any task later reuses the exact
  same images rather than resampling.
- **BatchNorm policy (Tasks 2 & 3):** running mean/variance are frozen at
  their pretrained ImageNet values for every method; only γ/β remain
  trainable. `model.train()` is always followed by putting BatchNorm
  modules back into `eval()` mode — the whole model is never put in
  `eval()` during training.
- **Controlled studies are bounded and pre-registered.** Where the
  assignment asks for one bounded ablation (Task 2 Step 6, Task 3 Step 5),
  the expected direction of the effect is stated in the code/README before
  the sweep is interpreted, and results from a sweep are never used to
  retroactively change the main comparison's settings.
- **Every reported number traces to a saved result file** (`results/*.csv`,
  `results/*.json`) or a documented, reproducible command — not to a number
  copied by hand from a notebook run.

## Deliverables checklist

- [ ] Public GitHub repository with this structure, `README.md`s per task,
      and no committed datasets/checkpoints (see `.gitignore`)
- [ ] `report/pa1_report.pdf` — 8 pages main content, NeurIPS format,
      GitHub link at the end of the abstract
- [ ] LMS submission: PDF + GitHub link, per the format announced on LMS
- [ ] External implementations (AdaIN, MMD/DANN/CDAN references, PROSER,
      optional RPL, etc.) attributed in this README or a `NOTICE`/`README`
      section per task
- [ ] Confirmed: no target/Sketch/CIFAR-100 leakage into any training,
      selection, or threshold-setting step (see checklist above)

## Attribution

Public implementations and library functions used as starting points for
method-specific components (e.g. gradient-reversal layers, MMD kernels,
AdaIN style transfer, PROSER placeholders) are attributed inline in the
relevant source files and summarized per task in that task's `README.md`.
Every line of submitted code is understood and owned by the author per the
assignment's AI-usage and coding-assistance policy; no generative AI was
used to write any part of the PDF report.
