# ══════════════════════════════════════════════════════════════════════════
# task3/shared/pacs_protocol.py — Task-3-local CONFIG, seeding, device
# ══════════════════════════════════════════════════════════════════════════
# NOTE ON WHY THIS LIVES UNDER task3/shared/ INSTEAD OF THE TOP-LEVEL shared/:
# The assignment's suggested layout reuses one shared/ package across Task 2
# and Task 3. In practice the two notebooks were written with different
# CONFIG schemas (e.g. Task 2 uses CONFIG["data_root"]/CONFIG["source_domains"],
# Task 3 uses CONFIG["PACS_ROOT"]/CONFIG["SOURCE_DOMAINS"]) and slightly
# different PACSDataset/transform-building code. Forcing both into one
# literal shared/pacs.py + shared/pacs_protocol.py would require rewriting
# working, tested code, which the instructions for this pass were explicitly
# to avoid. So Task 3's copies of the "shared"-tagged blocks are kept at
# task3/shared/, functionally identical to what the notebook already had,
# and this deviation is called out again in task3/README.md.
"""
=================================================================================
TASK 3 — DOMAIN GENERALIZATION ON PACS  (single-file Kaggle version)
=================================================================================

HOW TO READ THIS FILE
----------------------
Every section is marked with a header like:

    # =========================================================
    # [shared/pacs.py]
    # =========================================================

That tag tells you which file, under the "Suggested Repository Structure" from
the assignment, this block belongs in when you split things up later. Just
copy each tagged block into the matching file. Nothing else needs to change
(imports at the very top are shared across every file that needs them).

WHAT YOU MUST UPLOAD / EDIT BEFORE RUNNING
--------------------------------------------
1. PACS dataset as a Kaggle input (folders: photo/, art_painting/, cartoon/,
   sketch/, each containing one sub-folder per class). Set CONFIG["PACS_ROOT"].

2. Your Task 2 Source-only (ERM) checkpoint. Task 3 REUSES this checkpoint
   unchanged as the ERM baseline — it must NOT be retrained (the manual is
   explicit about this: "load its saved checkpoint rather than retraining it
   under a different configuration"). Upload the .pth file as a Kaggle
   dataset/input and set CONFIG["TASK2_CHECKPOINT_PATH"].

   >>> IMPORTANT: the loader below assumes the checkpoint is a plain
   >>> state_dict for a model shaped exactly like `PACSModel` below
   >>> (a `feature_extractor` Sequential up to avgpool + a `fc` Linear(512,7)).
   >>> If your Task 2 code used different attribute names (e.g. `model.resnet`,
   >>> `model.backbone`, `model.classifier`), the state_dict keys won't match
   >>> and `load_state_dict` will fail or silently mismatch. Paste me your
   >>> Task 2 model class / save code and I'll adjust the loader exactly —
   >>> for now there's a lenient loader with a key-remapping hook you can edit
   >>> (see `load_task2_checkpoint`).

3. (Recommended, not strictly required) Your Task 2 splits file
   (shared/splits/pacs_sketch_seed6304.json). Task 3 must reuse the EXACT
   SAME source train/val splits as Task 2. If you upload that file, set
   CONFIG["SPLITS_JSON_PATH"] to it and this script will load it directly.
   If you leave it as None, this script regenerates the split itself using
   the same recipe (stratified 80/20 per source domain, seed 6304,
   sklearn train_test_split on a sorted file listing). This SHOULD reproduce
   your Task 2 split bit-for-bit only if Task 2 used the identical listing
   order + split recipe — safest is to just upload the real file.

4. Class order: the classifier head's 7 output indices must match Task 2's
   class order exactly, or the reloaded ERM checkpoint will silently score
   the wrong classes. By default this script auto-discovers classes as
   `sorted(os.listdir(photo_domain_folder))`. If Task 2 used a different
   order, set CONFIG["CLASS_NAMES"] explicitly to a list in that exact order.

DECISIONS THE ASSIGNMENT LEAVES OPEN (both are implemented — toggle in CONFIG)
--------------------------------------------------------------------------------
- Controlled Design Study (Step 5): the manual says "choose one of two bounded
  studies" — either sweep lambda_DG in {0.1, 1, 10} for DAN-DG, OR sweep rho in
  {0.01, 0.05, 0.1} for SAM. Both sweeps are fully implemented below.
  Set CONFIG["CONTROLLED_STUDY_METHOD"] to "dan_dg" or "sam" to pick which one
  actually runs. (The main comparison in Step 1-4 always uses lambda_DG=1 and
  rho=0.05 regardless of this toggle — the manual requires that.)

STAGED EXECUTION
------------------
Because a full run (ERM load + DAN-DG training + SAM training + both
diagnostics + controlled study + final Sketch eval) is long, CONFIG has
boolean stage switches (RUN_TRAIN_DAN_DG, RUN_TRAIN_SAM, RUN_SOURCE_DIAGNOSTICS,
RUN_CONTROLLED_STUDY, RUN_FINAL_SKETCH_EVAL) so you can run this cell-by-cell
in Kaggle and re-run only what you need. Checkpoints/results are saved to disk
after each stage so later stages can just reload them.

NO-TARGET-LEAKAGE DISCIPLINE
------------------------------
Sketch images/labels are NEVER touched until `RUN_FINAL_SKETCH_EVAL`. All
training, checkpoint selection, and the controlled study only ever look at
source (Photo/Art Painting/Cartoon) validation performance. This mirrors the
manual's leakage rules for Task 3.
"""

# =====================================================================================
# TOP-LEVEL IMPORTS (shared across every file below — put these at the top of whichever
# file needs them; shared/pacs.py and shared/pacs_protocol.py need most of them)
# =====================================================================================
import os
import io
import json
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =====================================================================================
# CONFIG  — EDIT THIS BLOCK. This whole block conceptually lives split across
# task3/configs/{erm,dan_dg,sam}.yaml once you break things into the real repo.
# =====================================================================================
CONFIG = {
    # ---- paths you MUST edit -------------------------------------------------------
    "PACS_ROOT": "/kaggle/input/datasets/nickfratto/pacs-dataset/pacs_data/pacs_data" ,  # <-- EDIT
    "TASK2_CHECKPOINT_PATH": "" ,  # <-- EDIT (REQUIRED)
    "SPLITS_JSON_PATH": None,   # <-- set to a path if you have the Task2 splits file, else None
    "OUTPUT_DIR": "/kaggle/working/task3_outputs",

    # ---- domain / class setup -------------------------------------------------------
    "DOMAIN_FOLDERS": {          # map logical domain name -> folder name on disk
        "photo": "photo",
        "art_painting": "art_painting",
        "cartoon": "cartoon",
        "sketch": "sketch",
    },
    "SOURCE_DOMAINS": ["photo", "art_painting", "cartoon"],
    "TARGET_DOMAIN": "sketch",
    "CLASS_NAMES": None,  # None -> auto-discover from photo/ subfolders, sorted().
                           # SET EXPLICITLY if Task 2 used a different class order!

    # ---- reproducibility / device ----------------------------------------------------
    "SEED": 6304,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # ---- image pipeline ---------------------------------------------------------------
    "IMG_RESIZE": 256,
    "IMG_CROP": 224,
    "IMAGENET_MEAN": [0.485, 0.456, 0.406],
    "IMAGENET_STD": [0.229, 0.224, 0.225],

    # ---- training protocol (identical to Task 2, per the manual) --------------------
    "BATCH_PER_SOURCE": 8,     # 8 examples per source domain -> 24 per batch total
    "VAL_BATCH_SIZE": 64,
    "MAX_EPOCHS": 30,
    "PATIENCE": 5,
    "LR": 1e-4,
    "WD": 1e-4,

    # ---- DAN-DG main comparison -------------------------------------------------------
    "LAMBDA_DG_MAIN": 1.0,
    "MMD_KERNEL_MULTS": (0.5, 1.0, 2.0),

    # ---- SAM main comparison -----------------------------------------------------------
    "RHO_SAM_MAIN": 0.05,

    # ---- controlled design study (Step 5) — BOTH implemented, pick which one RUNS ----
    "CONTROLLED_STUDY_METHOD": "dan_dg",   # "dan_dg" or "sam"  <-- YOUR CHOICE HERE
    "LAMBDA_DG_SWEEP": [0.1, 1.0, 10.0],
    "RHO_SAM_SWEEP": [0.01, 0.05, 0.1],

    # ---- sharpness proxy ------------------------------------------------------------
    "SHARPNESS_RHO": 0.05,
    "SHARPNESS_N_PER_SOURCE": 32,

    # ---- domain separability diagnostic ----------------------------------------------
    "SEPARABILITY_TEST_SIZE": 0.30,
    "SEPARABILITY_C": 1.0,

    # ---- stage switches: turn stages on/off for staged Kaggle execution ---------------
    "RUN_TRAIN_DAN_DG": True,
    "RUN_TRAIN_SAM": True,
    "RUN_SOURCE_DIAGNOSTICS": True,     # separability + sharpness for ERM/DAN-DG/SAM
    "RUN_CONTROLLED_STUDY": True,
    "RUN_FINAL_SKETCH_EVAL": True,      # only flip on once everything above is frozen
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["OUTPUT_DIR"], "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["OUTPUT_DIR"], "plots"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["OUTPUT_DIR"], "results"), exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["SEED"])
DEVICE = torch.device(CONFIG["DEVICE"])


