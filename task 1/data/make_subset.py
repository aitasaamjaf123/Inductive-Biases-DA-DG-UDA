import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10
from torch.utils.data import Subset, DataLoader
from configs.config import SEED, DATA_ROOT, NUM_WORKERS
from data.transforms import base_transform

def load_train_data():
    os.makedirs(DATA_ROOT, exist_ok=True)
    raw_dataset = STL10(root=DATA_ROOT, split="train", download=True)
    labels = raw_dataset.labels
    CLASS_NAMES = raw_dataset.classes
    print(f"Loaded {len(raw_dataset)} training images, classes: {CLASS_NAMES}")
    
    if len(raw_dataset) != 5000:
        print(f"WARNING: expected 5000 images in the STL-10 train split, got {len(raw_dataset)}.")

    train_idxs, val_idxs = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )
    print(f"Train: {len(train_idxs)}, Validation: {len(val_idxs)}")
    return raw_dataset, train_idxs, val_idxs, CLASS_NAMES

def build_class_balanced_test_subset(device, class_names):
    print("\n--- Building class-balanced test subset ---")
    test_dataset_meta = STL10(root=DATA_ROOT, split="test", download=True)
    test_labels = np.array(test_dataset_meta.labels)
    rng = np.random.RandomState(SEED)
    selected_indices, imbalance_report = [], {}
    
    for c in range(10):
        class_indices = np.where(test_labels == c)[0]
        n_take = min(50, len(class_indices))
        if len(class_indices) < 50:
            imbalance_report[class_names[c]] = len(class_indices)
        selected_indices.extend(rng.choice(class_indices, size=n_take, replace=False).tolist())
        
    selected_indices = np.array(selected_indices)
    np.save(os.path.join(DATA_ROOT, "selected_test_identifiers.npy"), selected_indices)
    
    with open(os.path.join(DATA_ROOT, "selected_test_identifiers.json"), "w") as f:
        json.dump({"seed": SEED, "n_selected": int(len(selected_indices)),
                   "indices": selected_indices.tolist(), "imbalance": imbalance_report}, f, indent=2)
    print(f"Saved {len(selected_indices)} balanced test identifiers.")

    test_dataset_common = STL10(root=DATA_ROOT, split="test", transform=base_transform, download=False)
    test_subset = Subset(test_dataset_common, selected_indices)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=(device.type == "cuda"))
                             
    return test_dataset_common, test_loader, selected_indices