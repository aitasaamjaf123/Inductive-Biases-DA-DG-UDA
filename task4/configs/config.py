"""
Task 4 — Open-Set Recognition (OSR)
Shared configuration: paths, seeds, hyperparameters, and the "decision
switches" for the places where the assignment spec explicitly leaves a
choice open to the implementer.

Every other module in task4/ does `from configs.config import *`.
"""
import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 6304


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Paths — change these for your own machine / cluster. Defaults assume a
# Kaggle-style working directory; swap for local paths (e.g. "./data") when
# running elsewhere.
# ---------------------------------------------------------------------------
USE_KAGGLE_LOCAL_DATA = False   # True -> point DATA_ROOT at a pre-downloaded
                                 # dataset folder and set DOWNLOAD=False below.
DATA_ROOT   = "/kaggle/working/data"
CKPT_DIR    = "/kaggle/working/checkpoints"
CACHE_DIR   = "/kaggle/working/cache"        # cached logits/features (.npz)
RESULTS_DIR = "/kaggle/working/results"
SPLIT_DIR   = "/kaggle/working/splits"

for _d in [DATA_ROOT, CKPT_DIR, CACHE_DIR, RESULTS_DIR, SPLIT_DIR]:
    os.makedirs(_d, exist_ok=True)

DOWNLOAD = not USE_KAGGLE_LOCAL_DATA

# ---------------------------------------------------------------------------
# USER DECISION SWITCHES — flip these, nothing else, to pick between
# defensible readings of ambiguous spec language.
# ---------------------------------------------------------------------------
RUN_RPL = False
# Step 5 (RPL) is explicitly optional in the spec. Set True to also train
# and evaluate it, adding it as an extra row in the model-comparison table.

PLOT_STYLE = "both"
# "hist" -> score-distribution histograms only
# "roc"  -> ROC curves only
# "both" -> one figure with both (2 rows x 3 cols); satisfies the
#           "distributions OR ROC curves" requirement either way.

PROSER_SCORE_MODE = "dummy_minus_known"
# How to turn the 5 dummy logits + 10 known logits into PROSER's
# "placeholder-based detection score":
#   "dummy_minus_known": u(x) = max_dummy_logit(x) - max_known_logit(x)
#       (raw-logit margin; matches the official PROSER reference code's
#        effective test-time rejection rule)
#   "dummy_softmax": u(x) = softmax probability mass on the 5 dummy logits
#       when softmax is taken over all 15 logits together.
# Both are implemented in scores/proser_score.py; this switch selects which
# one is reported in the required tables/plots. Re-running evaluation after
# flipping this is cheap since it reuses cached logits (no retraining).

SKIP_IF_CHECKPOINT_EXISTS = True
# Resume across sessions without re-training a model whose checkpoint file
# already exists on disk.

# ---------------------------------------------------------------------------
# Fixed hyperparameters (from the spec — not switches)
# ---------------------------------------------------------------------------
BATCH_SIZE        = 128
NUM_EPOCHS_MAIN   = 100     # Vanilla / GCSC / RPL
NUM_EPOCHS_PROSER = 50
LR_MAIN           = 0.1
LR_PROSER         = 1e-3
MOMENTUM          = 0.9
WEIGHT_DECAY      = 5e-4
NUM_DUMMY         = 5       # PROSER dummy classifiers
PROSER_BETA       = 1.0     # classifier-placeholder loss weight
PROSER_GAMMA      = 0.1     # data-placeholder loss weight
MAHAL_EPS         = 1e-6

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor",
                "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock",
               "keyboard", "mushroom", "sunflower", "wardrobe"]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)