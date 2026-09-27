# ══════════════════════════════════════════════════════════════════════════
# shared/pacs.py — dataset / transforms / loaders (used by Task 2)
# ══════════════════════════════════════════════════════════════════════════
import os
import json
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights

from shared.pacs_protocol import CONFIG, SEED


# ══════════════════════════════════════════════════════════════════════════
# FILE: shared/pacs.py  — DATASET / TRANSFORMS / LOADERS
# ══════════════════════════════════════════════════════════════════════════

# Auto-detect folder-name variants so a slightly different PACS mirror
# doesn't crash the whole run.
DOMAIN_DIR_CANDIDATES = {
    "photo": ["photo", "Photo", "photos"],
    "art_painting": ["art_painting", "Art Painting", "art painting", "art"],
    "cartoon": ["cartoon", "Cartoon"],
    "sketch": ["sketch", "Sketch"],
}


def _resolve_domain_dir(data_root: str, domain: str) -> str:
    """Find the actual on-disk folder for a domain, trying a couple of
    reasonable roots (PACS is sometimes nested one extra level deep on
    Kaggle, e.g. pacs_data/pacs_data/<domain>/<class>/*.jpg)."""
    roots_to_try = [data_root]
    for name in os.listdir(data_root) if os.path.isdir(data_root) else []:
        candidate = os.path.join(data_root, name)
        if os.path.isdir(candidate):
            roots_to_try.append(candidate)

    for root in roots_to_try:
        if not os.path.isdir(root):
            continue
        for cand in DOMAIN_DIR_CANDIDATES[domain]:
            full = os.path.join(root, cand)
            if os.path.isdir(full):
                return full
    raise FileNotFoundError(
        f"Could not locate domain '{domain}' under '{data_root}'. "
        f"Tried folder names {DOMAIN_DIR_CANDIDATES[domain]} under "
        f"{roots_to_try}. Edit CONFIG['data_root'] or DOMAIN_DIR_CANDIDATES."
    )


def list_domain_samples(data_root: str, domain: str, classes):
    """Returns list[(filepath, class_idx)]. Verifies every class folder
    exists; skips (with a warning) any file that isn't a readable image
    instead of crashing training later."""
    domain_dir = _resolve_domain_dir(data_root, domain)
    samples = []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(domain_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Missing class folder '{cls_dir}' for domain '{domain}'.")
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((fpath, cls_idx))
    if len(samples) == 0:
        raise RuntimeError(f"Zero images found for domain '{domain}' — check paths/classes.")
    return samples


# Pull the exact normalization used to pretrain the ResNet-18 weights.
_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
_PRETRAIN_TFMS = _WEIGHTS.transforms()
NORM_MEAN = list(_PRETRAIN_TFMS.mean)
NORM_STD = list(_PRETRAIN_TFMS.std)

import torchvision.transforms as T

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


class PACSDataset(Dataset):
    """Simple (path, label) dataset with defensive image loading: a corrupt
    file is logged once and swapped for another random sample from the same
    dataset rather than raising mid-epoch."""

    def __init__(self, samples, transform, name="dataset"):
        self.samples = samples
        self.transform = transform
        self.name = name
        self._warned = set()

    def __len__(self):
        return len(self.samples)

    def _safe_load(self, path):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as e:
            if path not in self._warned:
                warnings.warn(f"[{self.name}] Could not read image '{path}': {e}. Substituting another sample.")
                self._warned.add(path)
            return None

    def __getitem__(self, idx):
        for _ in range(5):  # bounded retries so a bad file can't infinite-loop
            path, label = self.samples[idx]
            img = self._safe_load(path)
            if img is not None:
                return self.transform(img), label
            idx = random.randrange(len(self.samples))
        raise RuntimeError(f"[{self.name}] Repeated image-load failures; check dataset integrity.")


def build_stratified_source_splits(data_root, source_domains, classes, val_fraction, seed, split_cache_path):
    """Stratified 80/20 split per source domain, seed 6304, cached to disk
    so re-runs (or later Task 3 reuse) are exactly reproducible."""
    if os.path.exists(split_cache_path):
        with open(split_cache_path, "r") as f:
            cached = json.load(f)
        return cached

    splits = {}
    for domain in source_domains:
        samples = list_domain_samples(data_root, domain, classes)
        paths = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=val_fraction, random_state=seed, stratify=labels
        )
        splits[domain] = {
            "train": list(zip(train_paths, train_labels)),
            "val": list(zip(val_paths, val_labels)),
        }
    with open(split_cache_path, "w") as f:
        json.dump(splits, f)
    return splits


def build_target_split(data_root, target_domain, classes, split_cache_path):
    """Task 2: the FULL target domain is used (unlabeled) for adaptation and
    then evaluated with labels once everything is frozen — no train/val
    split is taken out of it."""
    if os.path.exists(split_cache_path):
        with open(split_cache_path, "r") as f:
            return json.load(f)
    samples = list_domain_samples(data_root, target_domain, classes)
    with open(split_cache_path, "w") as f:
        json.dump(samples, f)
    return samples


class InfiniteCycler:
    """Cycles a DataLoader forever so we can pull fixed-size batches
    (8/domain, 24 target) every step regardless of each split's length."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            return next(self._iter)


def make_loader(samples, transform, batch_size, shuffle, name, num_workers=None, drop_last=False):
    ds = PACSDataset(samples, transform, name=name)
    nw = CONFIG["num_workers"] if num_workers is None else num_workers
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=nw,
        drop_last=drop_last, pin_memory=torch.cuda.is_available(),
        worker_init_fn=lambda wid: np.random.seed(SEED + wid),
    )

