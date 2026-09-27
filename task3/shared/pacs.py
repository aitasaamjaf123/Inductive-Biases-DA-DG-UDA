# ══════════════════════════════════════════════════════════════════════════
# task3/shared/pacs.py — dataset class + transforms + split/loader helpers
# ══════════════════════════════════════════════════════════════════════════
# See the note at the top of task3/shared/pacs_protocol.py for why this is
# task3-local rather than the top-level shared/ package.
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split

from task3.shared.pacs_protocol import CONFIG

# =====================================================================================
# [shared/pacs.py]  — dataset class + transforms
# =====================================================================================
class PACSDataset(Dataset):
    """A flat list of (path, label) pairs for one domain / one split."""

    def __init__(self, samples, transform):
        # samples: list of (filepath:str, label:int)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def build_transforms(cfg):
    mean, std = cfg["IMAGENET_MEAN"], cfg["IMAGENET_STD"]
    train_tf = transforms.Compose([
        transforms.Resize((cfg["IMG_RESIZE"], cfg["IMG_RESIZE"])),
        transforms.RandomCrop(cfg["IMG_CROP"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg["IMG_RESIZE"], cfg["IMG_RESIZE"])),
        transforms.CenterCrop(cfg["IMG_CROP"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def discover_classes(cfg):
    if cfg["CLASS_NAMES"] is not None:
        return list(cfg["CLASS_NAMES"])
    photo_dir = os.path.join(cfg["PACS_ROOT"], cfg["DOMAIN_FOLDERS"]["photo"])
    classes = sorted([d for d in os.listdir(photo_dir)
                       if os.path.isdir(os.path.join(photo_dir, d))])
    assert len(classes) == 7, f"Expected 7 PACS classes, found {len(classes)}: {classes}"
    return classes


def list_domain_files(cfg, domain, classes):
    """Sorted (path, label) list for one domain, across all classes."""
    domain_dir = os.path.join(cfg["PACS_ROOT"], cfg["DOMAIN_FOLDERS"][domain])
    samples = []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(domain_dir, cls)
        files = sorted(os.listdir(cls_dir))
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((os.path.join(cls_dir, f), label))
    return samples



# =====================================================================================
# [shared/pacs_protocol.py]  — split loading/generation, dataloader construction
# =====================================================================================
def build_or_load_splits(cfg, classes):
    """
    Returns dict: {domain: {"train": [(path,label),...], "val": [(path,label),...]}}
    for each source domain. Sketch is NOT split (used whole, only at final eval).

    If cfg["SPLITS_JSON_PATH"] is set, loads that file directly (RECOMMENDED —
    guarantees identical splits to Task 2). Otherwise regenerates deterministically
    with the same recipe the manual specifies (stratified 80/20, seed 6304).
    """
    if cfg["SPLITS_JSON_PATH"] is not None and os.path.exists(cfg["SPLITS_JSON_PATH"]):
        with open(cfg["SPLITS_JSON_PATH"], "r") as f:
            raw = json.load(f)
        splits = {}
        for domain in cfg["SOURCE_DOMAINS"]:
            splits[domain] = {
                "train": [(p, l) for p, l in raw[domain]["train"]],
                "val": [(p, l) for p, l in raw[domain]["val"]],
            }
        print("[splits] Loaded splits JSON from", cfg["SPLITS_JSON_PATH"])
        return splits

    print("[splits] No splits JSON provided/found — regenerating "
          "stratified 80/20 splits per source domain with seed", cfg["SEED"],
          "(make sure this matches your Task 2 recipe!)")
    splits = {}
    for domain in cfg["SOURCE_DOMAINS"]:
        samples = list_domain_files(cfg, domain, classes)
        paths = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        train_p, val_p, train_l, val_l = train_test_split(
            paths, labels, test_size=0.2, random_state=cfg["SEED"], stratify=labels
        )
        splits[domain] = {
            "train": list(zip(train_p, train_l)),
            "val": list(zip(val_p, val_l)),
        }

    # Save what we generated so it's reproducible / can be reused as the "real" file.
    save_path = os.path.join(cfg["OUTPUT_DIR"], "pacs_sketch_seed6304_regenerated.json")
    with open(save_path, "w") as f:
        json.dump(splits, f)
    print("[splits] Saved regenerated splits to", save_path)
    return splits


def build_source_loaders(cfg, splits, train_tf, eval_tf):
    """Per-domain train loaders (used for infinite cycling) and val loaders."""
    train_loaders, val_loaders = {}, {}
    for domain in cfg["SOURCE_DOMAINS"]:
        train_ds = PACSDataset(splits[domain]["train"], train_tf)
        val_ds = PACSDataset(splits[domain]["val"], eval_tf)
        train_loaders[domain] = DataLoader(
            train_ds, batch_size=cfg["BATCH_PER_SOURCE"], shuffle=True,
            drop_last=True, num_workers=2, pin_memory=True,
        )
        val_loaders[domain] = DataLoader(
            val_ds, batch_size=cfg["VAL_BATCH_SIZE"], shuffle=False,
            num_workers=2, pin_memory=True,
        )
    return train_loaders, val_loaders


def build_target_loader(cfg, classes, eval_tf):
    """Whole Sketch domain — ONLY call this inside the final evaluation stage."""
    samples = list_domain_files(cfg, cfg["TARGET_DOMAIN"], classes)
    ds = PACSDataset(samples, eval_tf)
    loader = DataLoader(ds, batch_size=cfg["VAL_BATCH_SIZE"], shuffle=False,
                         num_workers=2, pin_memory=True)
    return loader


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

