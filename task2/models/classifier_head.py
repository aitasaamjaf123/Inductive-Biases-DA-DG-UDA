# ══════════════════════════════════════════════════════════════════════════
# task2/models/classifier_head.py
# ══════════════════════════════════════════════════════════════════════════
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/models/classifier_head.py  — CLASSIFIER HEAD
# ══════════════════════════════════════════════════════════════════════════

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feat):
        return self.fc(feat)