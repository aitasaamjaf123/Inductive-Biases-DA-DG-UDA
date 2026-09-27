"""
Task 4 — extract and cache penultimate features + logits (+ dummy logits for
PROSER) for a frozen model on a given loader, so every score can be computed
from exactly the same saved outputs.
"""
import os
import numpy as np
import torch

from configs.config import DEVICE, CACHE_DIR


@torch.no_grad()
def extract_outputs(model, loader, has_dummy=False):
    model.eval()
    feats, logits_list, dummy_list, labels_list = [], [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        if has_dummy:
            feat, logits, dummy = model(x, return_feat=True)
            dummy_list.append(dummy.cpu())
        else:
            feat, logits = model(x, return_feat=True)
        feats.append(feat.cpu())
        logits_list.append(logits.cpu())
        labels_list.append(y)
    out = {
        "feat": torch.cat(feats).numpy(),
        "logits": torch.cat(logits_list).numpy(),
        "labels": torch.cat(labels_list).numpy(),
    }
    out["dummy"] = torch.cat(dummy_list).numpy() if has_dummy else None
    return out


def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.npz")


def save_cache(name, d):
    kwargs = {k: v for k, v in d.items() if v is not None}
    np.savez(cache_path(name), **kwargs)


def load_cache(name):
    p = cache_path(name)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def extract_and_cache(model, loader, name, has_dummy=False, force=False):
    if not force:
        cached = load_cache(name)
        if cached is not None:
            print(f"[cache] loaded {name}")
            return cached
    out = extract_outputs(model, loader, has_dummy=has_dummy)
    save_cache(name, out)
    print(f"[cache] saved {name}")
    return out