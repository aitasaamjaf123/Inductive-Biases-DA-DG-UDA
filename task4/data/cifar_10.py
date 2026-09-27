"""
Task 4 — CIFAR-10 dataset wrapper, transforms, and loaders.
"""
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from configs.config import MEAN, STD, DATA_ROOT, DOWNLOAD, BATCH_SIZE
from data.make_splits import build_cifar10_splits


class ArraySubset(Dataset):
    """Applies a given transform to a subset (by absolute index) of a
    torchvision dataset that exposes .data (numpy HWC uint8) and .targets."""

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.base.data[idx]
        label = int(self.base.targets[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label


def get_transforms(randaugment=False):
    train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaugment:
        train_tf.append(T.RandAugment(num_ops=2, magnitude=9))
    train_tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    train_tf = T.Compose(train_tf)
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    return train_tf, eval_tf


def get_cifar10_loaders(randaugment=False, batch_size=BATCH_SIZE):
    """Standard train/val/test loaders. train_idx gets the (possibly
    RandAugment-augmented) train transform; val/test always get eval_tf."""
    train_tf, eval_tf = get_transforms(randaugment)
    base_train = CIFAR10(root=DATA_ROOT, train=True, download=DOWNLOAD)
    train_idx, val_idx = build_cifar10_splits()

    train_set = ArraySubset(base_train, train_idx, train_tf)
    val_set = ArraySubset(base_train, val_idx, eval_tf)
    test_set = CIFAR10(root=DATA_ROOT, train=False, download=DOWNLOAD, transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=2, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_cifar10_unaugmented_train_loader(batch_size=256):
    """Unaugmented (no crop/flip) view of the TRAINING portion only —
    required for fitting the Mahalanobis class means/covariance."""
    _, eval_tf = get_transforms(False)
    base_train = CIFAR10(root=DATA_ROOT, train=True, download=DOWNLOAD)
    train_idx, _ = build_cifar10_splits()
    ds = ArraySubset(base_train, train_idx, eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)