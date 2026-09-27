# ══════════════════════════════════════════════════════════════════════════
# task2/models/backbone.py — ResNet-18 backbone (BatchNorm running stats frozen)
# ══════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/models/backbone.py  — BACKBONE
# ══════════════════════════════════════════════════════════════════════════

class ResNet18Backbone(nn.Module):
    """ImageNet-pretrained ResNet-18 with its FC head stripped off, exposing
    the 512-d pooled feature. BatchNorm running stats are frozen per the
    protocol (see freeze_batchnorm_running_stats / set_bn_eval below)."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.avgpool = net.avgpool
        self.out_dim = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (N, 512)


def set_bn_eval(module: nn.Module):
    """Call AFTER model.train(). Puts every BatchNorm submodule into eval
    mode (so running_mean/running_var stop updating) while leaving the rest
    of the network in train mode (so gamma/beta keep receiving gradients and
    dropout etc. behave normally). Required every method, every epoch."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def freeze_batchnorm_running_stats(module: nn.Module):
    """Belt-and-suspenders: also stop momentum-based updates and disable
    gradient tracking on running buffers (they're buffers, not parameters,
    so this mainly documents intent / guards against accidental momentum
    changes elsewhere)."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = 0.0

