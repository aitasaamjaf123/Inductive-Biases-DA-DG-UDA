# PA1 — Tasks 2 & 3 (Unsupervised Domain Adaptation / Domain Generalization)

This is the Task 2 + Task 3 portion of the PA1 repository, split out of the
original notebooks (`task2.ipynb`, `task3.ipynb`) into the file layout the
assignment's "Suggested Repository Structure" describes, with no change to
the underlying logic. Task 1 and Task 4 are assumed complete and are not
included here.

```
shared/            # Task 2's shared PACS utilities (see task2/README.md)
task2/             # Unsupervised Domain Adaptation — see task2/README.md
task3/             # Domain Generalization — see task3/README.md
```

Read `task2/README.md` and `task3/README.md` for how to run each task and
for an explanation of the couple of places where the working code didn't
map 1:1 onto the assignment's suggested file list (and why).

## Run order

Task 3 reuses Task 2's Source-only ERM checkpoint and its exact source
splits (seed 6304), so run Task 2 first:

```bash
python -m task2.run_task2
# then point task3/shared/pacs_protocol.py's CONFIG at Task 2's outputs
# (TASK2_CHECKPOINT_PATH, SPLITS_JSON_PATH) — see task3/README.md
python -m task3.run_task3
```

## Requirements

Both tasks were developed against a Kaggle Python 3.12 environment. Install:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow
# optional, for task2's bonus UMAP script only:
pip install umap-learn
```

Pin exact versions in your own `requirements.txt` / `environment.yml` per
the assignment's Git appendix before you freeze your submission.
