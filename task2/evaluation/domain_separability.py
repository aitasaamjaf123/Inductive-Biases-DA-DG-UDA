# ══════════════════════════════════════════════════════════════════════════
# task2/evaluation/domain_separability.py
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from shared.pacs_protocol import SEED


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/evaluation/domain_separability.py
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        f = backbone(xb).cpu().numpy()
        feats.append(f)
        labels.append(yb.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def domain_separability_score(source_feats: np.ndarray, target_feats: np.ndarray, seed: int = SEED) -> float:
    """Equal numbers of source-val and target features -> 70/30 split ->
    balanced logistic regression (C=1) -> held-out accuracy distinguishing
    source vs. target. 50% = chance (domains fully confused)."""
    n = min(len(source_feats), len(target_feats))
    if n == 0:
        return float("nan")
    rng = np.random.RandomState(seed)
    src_idx = rng.choice(len(source_feats), size=n, replace=False)
    tgt_idx = rng.choice(len(target_feats), size=n, replace=False)

    X = np.concatenate([source_feats[src_idx], target_feats[tgt_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))

