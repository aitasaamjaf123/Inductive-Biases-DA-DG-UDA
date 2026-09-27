"""
Task 4 — fixed CIFAR-100 near/far unknown-class evaluation loaders.
Evaluation only: these images may never influence training, checkpoint
selection, score design, or threshold selection.
"""
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from configs.config import DATA_ROOT, DOWNLOAD
from data.cifar10 import ArraySubset, get_transforms


def get_cifar100_unknown_loader(class_names, batch_size=256):
    """CIFAR-100 TEST-set-only loader restricted to the given fine class
    names. Returns (loader, fine_label_ids_in_order) so callers can map
    predictions back to human-readable unknown-class names."""
    _, eval_tf = get_transforms(False)
    base = CIFAR100(root=DATA_ROOT, train=False, download=DOWNLOAD)
    name_to_idx = {c: i for i, c in enumerate(base.classes)}
    missing = [c for c in class_names if c not in name_to_idx]
    if missing:
        raise ValueError(f"CIFAR-100 class names not found: {missing}. "
                          f"Check spelling against base.classes.")
    target_ids = set(name_to_idx[c] for c in class_names)
    indices = [i for i, t in enumerate(base.targets) if t in target_ids]
    assert len(indices) == len(class_names) * 100, (
        f"Expected {len(class_names) * 100} images (100/class from the CIFAR-100 "
        f"test set), got {len(indices)}."
    )
    ds = ArraySubset(base, indices, eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return loader, base.classes