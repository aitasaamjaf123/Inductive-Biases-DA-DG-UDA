"""
Task 4 — Step 5 (OPTIONAL): Reciprocal Point Learning (Chen et al. 2020).

(1) Representation of reciprocal points: one learnable point R_c in the
    512-d feature space per known class, plus one learnable scalar radius
    r_c per class (the open-space margin). Both trained jointly with the
    backbone.
(2) Distance -> known-class score: for feature f(x), the "evidence for
    class c" logit is the SQUARED EUCLIDEAN DISTANCE d^2(f(x), R_c). R_c
    represents "otherness" for class c, so a true class-c example should
    lie FAR from R_c; cross-entropy over these distance-logits pushes the
    true class's distance to be the largest among the C distances.
(3) Open-space regularization: for the true class y, the distance to R_y is
    additionally penalized by a hinge loss max(0, d^2(f(x),R_y) - r_y^2),
    bounding how far into the "known" region class-y features may spread.
(4) Test-time rejection score: known-class prediction = argmax_c d^2(f(x),
    R_c) (farthest point). Unknownness score u(x) = -max_c d^2(f(x), R_c):
    a point not clearly far from ANY reciprocal point does not confidently
    belong to any known class, so a small max-distance indicates "unknown".
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import (DEVICE, SEED, set_seed, MOMENTUM, WEIGHT_DECAY,
                             NUM_EPOCHS_MAIN, LR_MAIN, SKIP_IF_CHECKPOINT_EXISTS)
from models.resnet_cifar import ResNet18Cifar
from evaluation.thresholds import cosine_lr


class ReciprocalPoints(nn.Module):
    def __init__(self, num_classes, feat_dim, gamma=1.0):
        super().__init__()
        self.R = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)
        self.log_radius = nn.Parameter(torch.zeros(num_classes))  # softplus'd for positivity
        self.gamma = gamma

    def sq_distances(self, feat):
        diff = feat.unsqueeze(1) - self.R.unsqueeze(0)   # (B,C,D)
        return (diff ** 2).sum(-1)                        # (B,C)

    def radius(self):
        return F.softplus(self.log_radius) + 1e-3


def rpl_loss(feat, y, rp_module: ReciprocalPoints, lam_open=0.1):
    d2 = rp_module.sq_distances(feat)
    logits = rp_module.gamma * d2
    ce = F.cross_entropy(logits, y)
    true_d2 = d2.gather(1, y.unsqueeze(1)).squeeze(1)
    r2 = (rp_module.radius() ** 2)[y]
    open_loss = F.relu(true_d2 - r2).mean()
    return ce + lam_open * open_loss, ce.item(), open_loss.item()


def train_rpl(train_loader, val_loader, epochs=NUM_EPOCHS_MAIN, base_lr=LR_MAIN,
              ckpt_path=None, tag="rpl"):
    backbone_ckpt = ckpt_path.replace(".pt", "_backbone.pt")
    rp_ckpt = ckpt_path.replace(".pt", "_rp.pt")
    if SKIP_IF_CHECKPOINT_EXISTS and os.path.exists(backbone_ckpt) and os.path.exists(rp_ckpt):
        print(f"[{tag}] checkpoints exist, skipping training.")
        backbone = ResNet18Cifar(num_classes=10, num_dummy=0).to(DEVICE)
        backbone.load_state_dict(torch.load(backbone_ckpt, map_location=DEVICE))
        rp = ReciprocalPoints(10, 512).to(DEVICE)
        rp.load_state_dict(torch.load(rp_ckpt, map_location=DEVICE))
        return backbone, rp

    set_seed(SEED)
    backbone = ResNet18Cifar(num_classes=10, num_dummy=0).to(DEVICE)
    rp = ReciprocalPoints(10, 512).to(DEVICE)
    params = list(backbone.parameters()) + list(rp.parameters())
    optimizer = torch.optim.SGD(params, lr=base_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    for epoch in range(epochs):
        lr = cosine_lr(optimizer, base_lr, epoch, epochs)
        backbone.train(); rp.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            feat, _ = backbone(x, return_feat=True)
            loss, ce, ol = rpl_loss(feat, y, rp)
            loss.backward()
            optimizer.step()

        # val accuracy = argmax over distances
        backbone.eval(); rp.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                feat, _ = backbone(x, return_feat=True)
                d2 = rp.sq_distances(feat)
                pred = d2.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(backbone.state_dict(), backbone_ckpt)
            torch.save(rp.state_dict(), rp_ckpt)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{tag}] epoch {epoch:3d}/{epochs} lr={lr:.4f} val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

    backbone.load_state_dict(torch.load(backbone_ckpt, map_location=DEVICE))
    rp.load_state_dict(torch.load(rp_ckpt, map_location=DEVICE))
    return backbone, rp


@torch.no_grad()
def rpl_extract(backbone, rp, loader):
    backbone.eval(); rp.eval()
    d2_list, labels_list = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        feat, _ = backbone(x, return_feat=True)
        d2 = rp.sq_distances(feat)
        d2_list.append(d2.cpu())
        labels_list.append(y)
    d2 = torch.cat(d2_list).numpy()
    labels = torch.cat(labels_list).numpy()
    return d2, labels


def rpl_score(d2):
    """u(x) = -max_c d^2(x,R_c): small max-distance -> looks unknown."""
    return -d2.max(axis=1)


def rpl_accuracy_from_d2(d2, labels):
    pred = d2.argmax(axis=1)
    return float((pred == labels).mean())