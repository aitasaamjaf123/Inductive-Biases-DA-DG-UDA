"""
Task 4 — generic supervised training loop, shared by methods/vanilla.py and
methods/gcsc.py (they differ only in whether the train loader uses
RandAugment).
"""
import os
import torch
import torch.nn.functional as F

from configs.config import (DEVICE, SEED, set_seed, WEIGHT_DECAY, MOMENTUM,
                             SKIP_IF_CHECKPOINT_EXISTS)
from evaluation.thresholds import cosine_lr, evaluate_accuracy


def train_classifier(model, train_loader, val_loader, epochs, base_lr, ckpt_path,
                      weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, tag=""):
    if SKIP_IF_CHECKPOINT_EXISTS and os.path.exists(ckpt_path):
        print(f"[{tag}] checkpoint already exists at {ckpt_path}, skipping training.")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE)
        val_acc = evaluate_accuracy(model, val_loader)
        return val_acc, []

    set_seed(SEED)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                 momentum=momentum, weight_decay=weight_decay)
    best_val_acc = -1.0
    history = []
    for epoch in range(epochs):
        lr = cosine_lr(optimizer, base_lr, epoch, epochs)
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        train_acc = correct / total
        val_acc = evaluate_accuracy(model, val_loader)
        history.append({"epoch": epoch, "lr": lr, "train_loss": running_loss / total,
                         "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{tag}] epoch {epoch:3d}/{epochs} lr={lr:.4f} "
                  f"train_loss={running_loss/total:.4f} train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f} (best={best_val_acc:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return best_val_acc, history