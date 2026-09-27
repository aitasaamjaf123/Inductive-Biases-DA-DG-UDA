# ══════════════════════════════════════════════════════════════════════════
# task2/methods/cdan.py — multilinear conditioning map g(x) = vec(f ⊗ p)
# ══════════════════════════════════════════════════════════════════════════
import torch


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/methods/cdan.py  — multilinear conditioning
# ══════════════════════════════════════════════════════════════════════════

def multilinear_map(feat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """g(x) = vec(f ⊗ p). No entropy conditioning, no detaching f or p, per
    protocol — gradients flow through both the feature and the softmax
    classifier output into the backbone via the GRL."""
    n, d = feat.size()
    c = probs.size(1)
    outer = torch.bmm(feat.unsqueeze(2), probs.unsqueeze(1))  # (n, d, c)
    return outer.view(n, d * c)
