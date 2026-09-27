# ══════════════════════════════════════════════════════════════════════════
# task3/evaluation/source_domain_separability.py
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# =====================================================================================
# [task3/evaluation/source_domain_separability.py]
# =====================================================================================
@torch.no_grad()
def extract_features(model, loader, device, max_n=None):
    model.eval()
    feats, labels = [], []
    count = 0
    for x, y in loader:
        x = x.to(device)
        _, f = model(x, return_features=True)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
        count += x.size(0)
        if max_n is not None and count >= max_n:
            break
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    if max_n is not None:
        feats, labels = feats[:max_n], labels[:max_n]
    return feats, labels


def source_domain_separability(cfg, model, val_loaders, device):
    """Balanced 3-class (Photo/Art/Cartoon) logistic regression on frozen features."""
    per_domain_feats = {}
    counts = []
    for d in cfg["SOURCE_DOMAINS"]:
        f, _ = extract_features(model, val_loaders[d], device)
        per_domain_feats[d] = f
        counts.append(f.shape[0])
    n_balanced = min(counts)  # balance across the three source domains

    X, y = [], []
    for i, d in enumerate(cfg["SOURCE_DOMAINS"]):
        idx = np.random.RandomState(cfg["SEED"]).choice(
            per_domain_feats[d].shape[0], size=n_balanced, replace=False)
        X.append(per_domain_feats[d][idx])
        y.append(np.full(n_balanced, i))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["SEPARABILITY_TEST_SIZE"],
        random_state=cfg["SEED"], stratify=y)

    clf = LogisticRegression(multi_class="multinomial", C=cfg["SEPARABILITY_C"],
                              max_iter=2000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return {"source_domain_separability_accuracy": acc, "chance_level": 1.0 / 3.0}




