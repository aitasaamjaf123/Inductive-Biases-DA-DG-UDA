"""
Task 4 — CIFAR-appropriate ResNet-18.

Per spec: the ImageNet 7x7/stride-2 stem conv is replaced with a 3x3/stride-1
conv, the initial max-pool is removed, and the network operates directly on
32x32 images. Supports optional `num_dummy` extra "placeholder" classifier
heads for PROSER (Step 4); when num_dummy == 0 the model is a plain 10-class
classifier (used for Vanilla / GCSC / RPL).

forward_pre / forward_post split the network at "after layer2, before
layer3" — the exact point the spec requires for PROSER's manifold mixup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes=10, num_dummy=0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # <- CIFAR stem
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.num_dummy = num_dummy
        if num_dummy > 0:
            self.dummy_fc = nn.Linear(512, num_dummy)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward_pre(self, x):
        """Stem through layer2 (mixup point)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward_post(self, h):
        """layer3 through logits, given a (possibly mixed) layer2 feature map."""
        out = self.layer3(h)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = torch.flatten(out, 1)
        logits = self.fc(feat)
        dummy_logits = self.dummy_fc(feat) if self.num_dummy > 0 else None
        return feat, logits, dummy_logits

    def forward(self, x, return_feat=False):
        h = self.forward_pre(x)
        feat, logits, dummy_logits = self.forward_post(h)
        if self.num_dummy > 0:
            return (feat, logits, dummy_logits) if return_feat else (logits, dummy_logits)
        return (feat, logits) if return_feat else logits


def unpack_logits(out):
    """out may be logits, (logits,dummy), (feat,logits) or (feat,logits,dummy)."""
    if isinstance(out, tuple):
        if len(out) == 2:
            a, b = out
            return b if a.dim() == 2 and a.shape[-1] != 10 and b.shape[-1] == 10 else a
        elif len(out) == 3:
            return out[1]
    return out