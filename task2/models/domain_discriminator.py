# ══════════════════════════════════════════════════════════════════════════
# task2/models/domain_discriminator.py — GRL + domain discriminator (used by DANN & CDAN)
# ══════════════════════════════════════════════════════════════════════════
import math
import torch
import torch.nn as nn
from torch.autograd import Function


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/models/domain_discriminator.py  — GRL + DOMAIN DISCRIMINATOR
# ══════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)


def grl_alpha_schedule(p: float, max_alpha: float = 1.0) -> float:
    """Standard DANN schedule: alpha(p) = 2/(1+exp(-10p)) - 1, scaled by
    max_alpha for the controlled gradient-reversal-strength study."""
    p = min(max(p, 0.0), 1.0)
    base = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    return max_alpha * base


class DomainDiscriminator(nn.Module):
    """256-unit hidden layer, ReLU, dropout 0.5, 2-class output — used
    identically by DANN (input = 512-d feature) and CDAN (input =
    512*num_classes-d multilinear map), only in_dim differs."""

    def __init__(self, in_dim, hidden=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)

