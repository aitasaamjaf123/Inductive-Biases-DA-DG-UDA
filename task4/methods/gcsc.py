"""
Task 4 — Step 3: GCSC (stronger closed-set classifier via RandAugment).
Identical recipe to Vanilla except the train loader was built with
randaugment=True (RandAugment(num_ops=2, magnitude=9) inserted after
crop+flip); evaluated with MLS to test whether a closed-set accuracy gain
is accompanied by better rejection.
"""
import os
from configs.config import CKPT_DIR, NUM_EPOCHS_MAIN, LR_MAIN
from models.resnet_cifar import ResNet18Cifar
from train import train_classifier


def run_gcsc(train_loader_randaugment, val_loader):
    ckpt = os.path.join(CKPT_DIR, "gcsc_best.pt")
    model = ResNet18Cifar(num_classes=10, num_dummy=0)
    val_acc, history = train_classifier(
        model, train_loader_randaugment, val_loader,
        epochs=NUM_EPOCHS_MAIN, base_lr=LR_MAIN, ckpt_path=ckpt, tag="gcsc",
    )
    return model, val_acc, history, ckpt