"""
Task 4 — Step 4: PROSER (Zhou et al. 2021) — classifier and data placeholders.

Initialized from the selected Vanilla checkpoint, appends `num_dummy`
randomly initialized dummy classifiers, and fine-tunes with a combination of
the classifier-placeholder loss (beta=1) and the manifold-mixup
data-placeholder loss (gamma=0.1). Each mini-batch is split in half: the
first half drives the classifier-placeholder loss, the second half drives
the data-placeholder loss.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

from configs.config import (DEVICE, SEED, set_seed, MOMENTUM, WEIGHT_DECAY,
                             NUM_EPOCHS_PROSER, LR_PROSER, NUM_DUMMY,
                             PROSER_BETA, PROSER_GAMMA, SKIP_IF_CHECKPOINT_EXISTS)
from models.resnet_cifar import ResNet18Cifar
from evaluation.thresholds import cosine_lr, evaluate_accuracy
from methods.manifold_mixup import data_placeholder_loss


def classifier_placeholder_loss(logits_known, dummy_logits, y, beta=PROSER_BETA):
    """
    Zhou et al. (2021) classifier-placeholder loss, beta=1.
      L1: ordinary example should be classified correctly among
          {K known classes, "placeholder" = strongest dummy response}.
      L2: with the TRUE class masked out, the placeholder should now beat
          every remaining known class (dummy becomes the runner-up region).
      total = L1 + beta * L2
    """
    dummy_max, _ = dummy_logits.max(dim=1, keepdim=True)          # (B,1)
    combined = torch.cat([logits_known, dummy_max], dim=1)         # (B, K+1)
    L1 = F.cross_entropy(combined, y)

    masked = logits_known.clone()
    masked.scatter_(1, y.unsqueeze(1), float("-inf"))
    combined2 = torch.cat([masked, dummy_max], dim=1)
    target2 = torch.full_like(y, logits_known.size(1))             # placeholder index = K
    L2 = F.cross_entropy(combined2, target2)

    return L1 + beta * L2


def train_proser(vanilla_ckpt_path, train_loader, val_loader,
                  epochs=NUM_EPOCHS_PROSER, base_lr=LR_PROSER,
                  num_dummy=NUM_DUMMY, ckpt_path=None,
                  beta=PROSER_BETA, gamma=PROSER_GAMMA, tag="proser"):
    if SKIP_IF_CHECKPOINT_EXISTS and os.path.exists(ckpt_path):
        print(f"[{tag}] checkpoint already exists at {ckpt_path}, skipping training.")
        model = ResNet18Cifar(num_classes=10, num_dummy=num_dummy).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        val_acc = evaluate_accuracy(model, val_loader)
        return model, val_acc, []

    set_seed(SEED)
    model = ResNet18Cifar(num_classes=10, num_dummy=num_dummy).to(DEVICE)
    vanilla_state = torch.load(vanilla_ckpt_path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(vanilla_state, strict=False)
    print(f"[{tag}] loaded vanilla init. missing={missing} unexpected={unexpected}")

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                 momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    best_val_acc = -1.0
    history = []
    for epoch in range(epochs):
        lr = cosine_lr(optimizer, base_lr, epoch, epochs)
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            B = x.size(0)
            half = B // 2
            x1, y1 = x[:half], y[:half]
            x2, y2 = x[half:], y[half:]

            optimizer.zero_grad()

            # ---- first half: classifier-placeholder loss ----
            feat1, logits1, dummy1 = model(x1, return_feat=True)
            loss_cls = classifier_placeholder_loss(logits1, dummy1, y1, beta=beta)

            # ---- second half: manifold-mixup data-placeholder loss ----
            loss_data = torch.tensor(0.0, device=DEVICE)
            if x2.size(0) >= 2:
                perm = torch.randperm(x2.size(0), device=x2.device)
                x2b, y2b = x2[perm], y2[perm]
                valid = (y2 != y2b)
                if valid.sum().item() > 0:
                    loss_data = data_placeholder_loss(model, x2[valid], x2b[valid])

            loss = loss_cls + gamma * loss_data
            loss.backward()
            optimizer.step()

        val_acc = evaluate_accuracy(model, val_loader)
        history.append({"epoch": epoch, "lr": lr, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[{tag}] epoch {epoch:3d}/{epochs} lr={lr:.5f} "
                  f"val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model, best_val_acc, history