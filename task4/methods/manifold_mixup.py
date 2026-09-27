"""
Task 4 — PROSER data-placeholder loss via manifold mixup.
Mixes representations after layer2/before layer3 of two CIFAR-10 examples of
DIFFERENT classes (caller filters this), lambda ~ Beta(2,2), and trains the
mixed representation toward the dummy ("unknown") classifiers.
"""
import numpy as np
import torch
import torch.nn.functional as F


def data_placeholder_loss(model, x1, x2):
    h1 = model.forward_pre(x1)
    h2 = model.forward_pre(x2)
    lam = float(np.random.beta(2.0, 2.0))
    mixed_h = lam * h1 + (1.0 - lam) * h2
    _, logits_known, dummy_logits = model.forward_post(mixed_h)
    dummy_max, _ = dummy_logits.max(dim=1, keepdim=True)
    combined = torch.cat([logits_known, dummy_max], dim=1)
    target = torch.full((x1.size(0),), logits_known.size(1),
                         dtype=torch.long, device=x1.device)
    return F.cross_entropy(combined, target)