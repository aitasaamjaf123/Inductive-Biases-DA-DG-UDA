import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from configs.config import device, SEED

def train_head(features, labels, train_idx, val_idx, num_classes=10, seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(features[train_idx], labels[train_idx]),
        batch_size=128,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(features[val_idx], labels[val_idx]),
        batch_size=128,
        shuffle=False,
    )
    classifier = nn.Linear(features.shape[1], num_classes).to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_weights, patience, p_count = 0.0, None, 5, 0

    for epoch in range(50):
        classifier.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(classifier(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

        classifier.eval()
        correct = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                correct += (classifier(xb.to(device)).argmax(dim=1) == yb.to(device)).sum().item()
        val_acc = correct / len(val_idx)

        if val_acc > best_acc:
            best_acc, best_weights, p_count = val_acc, copy.deepcopy(classifier.state_dict()), 0
        else:
            p_count += 1
            if p_count >= patience:
                break

    classifier.load_state_dict(best_weights)
    classifier.eval()
    return classifier, best_acc

def transformed_loader(base_loader, transform_fn=None):
    for images, targets in base_loader:
        yield (images if transform_fn is None else transform_fn(images)), targets
 
def evaluate_predictions(
    dataset_loader,
    predictors,
    baseline_preds=None,
    shape_labels=None,
    texture_labels=None,
    return_features=False,
):
    all_preds = {name: [] for name in predictors}
    all_confs = {name: [] for name in predictors}
    all_feats = {name: [] for name in predictors} if return_features else None
    all_targets = []
 
    with torch.inference_mode():
        for images, targets in dataset_loader:
            all_targets.append(targets.cpu() if torch.is_tensor(targets) else torch.as_tensor(targets))
            for name, predict_fn in predictors.items():
                out = predict_fn(images)
                all_preds[name].append(out["preds"])
                all_confs[name].append(out["confs"])
                if return_features:
                    all_feats[name].append(out["feats"])
 
    all_targets = torch.cat(all_targets).numpy()
 
    results = {}
    for name in predictors:
        preds = torch.cat(all_preds[name]).numpy()
        confs = torch.cat(all_confs[name]).numpy()
        metrics = {"mean_max_confidence": float(confs.mean()), "preds": preds}
 
        if shape_labels is None:
            metrics["top1_accuracy"] = float((preds == all_targets).mean())
            metrics["macro_f1"] = float(f1_score(all_targets, preds, average="macro"))
        else:
            shape_labels_arr = np.asarray(shape_labels)
            texture_labels_arr = np.asarray(texture_labels)
            is_shape = preds == shape_labels_arr
            is_texture = (preds == texture_labels_arr) & ~is_shape
            is_other = ~is_shape & ~is_texture
 
            n_shape, n_texture, n_other = int(is_shape.sum()), int(is_texture.sum()), int(is_other.sum())
            n_total = len(preds)
            denom = n_shape + n_texture
            metrics.update(
                {
                    "n_shape": n_shape,
                    "n_texture": n_texture,
                    "n_other": n_other,
                    "n_total": n_total,
                    "shape_bias_pct": float(100 * n_shape / denom) if denom > 0 else float("nan"),
                    "coverage_pct": float(100 * denom / n_total),
                }
            )
 
        if baseline_preds is not None:
            metrics["consistency"] = float((preds == baseline_preds[name]).mean())
 
        if return_features:
            metrics["feats"] = torch.cat(all_feats[name]).numpy()
 
        results[name] = metrics
 
    return results