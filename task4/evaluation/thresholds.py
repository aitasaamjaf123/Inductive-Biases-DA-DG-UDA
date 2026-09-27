"""
Task 4 — cosine LR schedule and a shared closed-set-accuracy evaluator.
(Named `thresholds.py` to match the assignment's suggested layout, which
groups this schedule helper alongside the rejection-threshold logic in
metrics.py.)
"""
import math
import torch

from configs.config import DEVICE


def cosine_lr(optimizer, base_lr, epoch, total_epochs):
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * epoch / total_epochs))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def evaluate_accuracy(model, loader):
    """Accuracy using ONLY the known-class logits (ignores dummy heads if any)."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total