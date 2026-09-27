# ══════════════════════════════════════════════════════════════════════════
# FILE: shared/pacs_protocol.py  — GLOBAL CONFIG & SEEDING
# ══════════════════════════════════════════════════════════════════════════
import os
import io
import json
import math
import random
import warnings
import traceback
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # don't crash on slightly-truncated jpgs

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import torchvision
from torchvision.models import resnet18, ResNet18_Weights

SEED = 6304

CONFIG = {
    "data_root": "/kaggle/input/datasets/nickfratto/pacs-dataset/pacs_data/pacs_data",   # <-- EDIT to your PACS path
    "output_root": "/kaggle/working/task2_outputs",
    "classes": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
    "source_domains": ["photo", "art_painting", "cartoon"],
    "target_domain": "sketch",
    "val_fraction": 0.2,
    "seed": SEED,
    "max_epochs": 30,
    "early_stop_patience": 5,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "per_source_domain_batch": 8,   # -> 24 total source images / step
    "target_batch": 24,             # equal total source/target batch sizes
    "num_workers": 2,
    "lambda_mmd": 1.0,
    "mmd_kernel_mults": [0.5, 1.0, 2.0],
    "grl_max_alpha": 1.0,           # scales the standard DANN/CDAN schedule
    "domain_disc_hidden": 256,
    "domain_disc_dropout": 0.5,
    "grad_clip_norm": 5.0,          # safety only; applied identically to all methods
    "controlled_study": {
        "method": "dan",            # "dan" -> sweep lambda_mmd, "dann" -> sweep grl_max_alpha
        "lambda_mmd_grid": [0.1, 1.0, 10.0],
        "grl_max_alpha_grid": [0.25, 0.5, 1.0],
    },
}

os.makedirs(CONFIG["output_root"], exist_ok=True)
for sub in ["checkpoints", "logs", "results", "splits"]:
    os.makedirs(os.path.join(CONFIG["output_root"], sub), exist_ok=True)


def set_seed(seed: int = SEED):
    """Seed every RNG we touch. Called before every method's training run so
    comparisons are apples-to-apples (protocol requires seed 6304 throughout)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic where possible; don't hard-crash on ops without a
    # deterministic kernel (some CUDA ops for ResNet don't have one).
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    warnings.warn("CUDA not available — falling back to CPU. This will be slow.")
    return torch.device("cpu")


DEVICE = get_device()