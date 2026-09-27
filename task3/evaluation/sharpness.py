# ══════════════════════════════════════════════════════════════════════════
# task3/evaluation/sharpness.py
# ══════════════════════════════════════════════════════════════════════════
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# =====================================================================================
# [task3/evaluation/sharpness.py]
# =====================================================================================
def build_fixed_sharpness_batch(cfg, splits, eval_tf, device):
    """32 examples per source domain, sampled deterministically with seed 6304
    from each source VALIDATION split, shared across ERM/DAN-DG/SAM."""
    rng = np.random.RandomState(cfg["SEED"])
    xs, ys = [], []
    for d in cfg["SOURCE_DOMAINS"]:
        val_samples = splits[d]["val"]
        n = cfg["SHARPNESS_N_PER_SOURCE"]
        idx = rng.choice(len(val_samples), size=min(n, len(val_samples)), replace=False)
        for i in idx:
            path, label = val_samples[i]
            img = Image.open(path).convert("RGB")
            img = eval_tf(img)
            xs.append(img)
            ys.append(label)
    x = torch.stack(xs, dim=0).to(device)
    y = torch.tensor(ys, dtype=torch.long).to(device)
    return x, y


def sharpness_proxy(model, x, y, rho, device):
    """Delta_sharp = L(theta + eps) - L(theta), eps = rho * grad/||grad||_2.
    Model must be in eval() mode (BN already frozen anyway since only BN2d
    running stats matter, but eval() also fixes dropout etc. deterministically).
    Restores original parameters afterwards."""
    model.eval()
    model.zero_grad()

    logits = model(x)
    loss_orig = F.cross_entropy(logits, y)
    loss_orig_val = loss_orig.item()

    loss_orig.backward()
    grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    grad_norm = torch.norm(torch.stack(
        [g.norm(p=2) for g in grads if g is not None]), p=2)

    originals = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            if g is None:
                continue
            eps = rho * g / (grad_norm + 1e-12)
            p.add_(eps)

    with torch.no_grad():
        logits_pert = model(x)
        loss_pert = F.cross_entropy(logits_pert, y).item()

    # restore
    with torch.no_grad():
        for p, orig in zip(model.parameters(), originals):
            p.copy_(orig)

    model.zero_grad()
    return loss_pert - loss_orig_val

