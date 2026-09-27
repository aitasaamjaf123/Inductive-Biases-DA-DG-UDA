"""
Task 4 — Step 1: Vanilla closed-set baseline.
Thin wrapper around the shared train_classifier() loop with no augmentation
beyond the standard crop+flip, and num_dummy=0 (plain 10-class classifier).
"""
import os
from configs.config import CKPT_DIR, NUM_EPOCHS_MAIN, LR_MAIN
from models.resnet_cifar import ResNet18Cifar
from train import train_classifier


def run_vanilla(train_loader_plain, val_loader):
    ckpt = os.path.join(CKPT_DIR, "vanilla_best.pt")
    model = ResNet18Cifar(num_classes=10, num_dummy=0)
    val_acc, history = train_classifier(
        model, train_loader_plain, val_loader,
        epochs=NUM_EPOCHS_MAIN, base_lr=LR_MAIN, ckpt_path=ckpt, tag="vanilla",
    )
    return model, val_acc, history, ckpt