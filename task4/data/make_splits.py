"""
Task 4 — stratified 90/10 split of the official CIFAR-10 training partition,
seed 6304. Cached to disk so every downstream script reuses the exact same
indices.
"""
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import CIFAR10

from configs.config import SEED, DATA_ROOT, SPLIT_DIR, DOWNLOAD


def build_cifar10_splits(save=True):
    """Returns (train_idx, val_idx) as np arrays of absolute indices into
    the base CIFAR-10 training dataset."""
    split_path = os.path.join(SPLIT_DIR, "cifar10_train_val_split.npz")
    if os.path.exists(split_path):
        d = np.load(split_path)
        return d["train_idx"], d["val_idx"]

    base_train = CIFAR10(root=DATA_ROOT, train=True, download=DOWNLOAD)
    labels = np.array(base_train.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    if save:
        np.savez(split_path, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx