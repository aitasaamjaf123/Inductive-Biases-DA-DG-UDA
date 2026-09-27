# ══════════════════════════════════════════════════════════════════════════
# task3/models/backbone.py — ResNet-18 backbone + 7-way head (fully fine-tuned)
# ══════════════════════════════════════════════════════════════════════════
# NOTE: the assignment's suggested layout splits this into
# task3/models/backbone.py + task3/models/classifier_head.py. The notebook
# implements the backbone and the linear head as one nn.Module (PACSModel)
# because the whole network is fully fine-tuned in Task 3 (unlike Task 2's
# frozen-backbone + linear-head setup), so there is no natural seam between
# "backbone parameters" and "head parameters" at the code level. Keeping
# PACSModel as one class avoids introducing a split that the working code
# never had. task3/models/classifier_head.py re-exports PACSModel.fc as a
# thin pointer, with an explanation, for anyone following the suggested
# layout literally.
import torch
import torch.nn as nn
from torchvision import models


# =====================================================================================
# [task3/models/backbone.py] + [task3/models/classifier_head.py]
# (kept as one class here for simplicity; split the two nn.Modules apart later)
# =====================================================================================
class PACSModel(nn.Module):
    """ResNet-18 (ImageNet1K_V1) backbone + 7-way linear head, fully fine-tuned."""

    def __init__(self, num_classes=7):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        # everything up to (and including) avgpool -> 512-dim feature
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        feat = self.feature_extractor(x)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        if return_features:
            return logits, feat
        return logits


def freeze_bn(model):
    """Freeze BatchNorm running stats (mean/var) while leaving gamma/beta trainable.
    Call this every time AFTER model.train() — never put the whole model in eval()."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def load_task2_checkpoint(model, path, device):
    """
    Loads the Task 2 Source-only ERM checkpoint into `model`.

    Assumes a plain state_dict saved with keys matching PACSModel
    (feature_extractor.*, fc.*). EDIT THE REMAP DICT BELOW if your Task 2
    checkpoint used different attribute names (e.g. saved as
    {'backbone.*':..., 'classifier.*':...}).
    """
    raw = torch.load(path, map_location=device)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    # ---- OPTIONAL KEY REMAPPING (edit if your Task2 keys differ) ----
    key_remap = {
        # "old_prefix.": "new_prefix.",   # example: "resnet.": "feature_extractor."
    }
    remapped = {}
    for k, v in state_dict.items():
        nk = k
        for old, new in key_remap.items():
            if nk.startswith(old):
                nk = new + nk[len(old):]
        remapped[nk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        print("[WARNING] load_task2_checkpoint: missing keys:", missing)
        print("[WARNING] load_task2_checkpoint: unexpected keys:", unexpected)
        print("          -> If this list is non-empty, your Task 2 checkpoint's "
              "naming doesn't match PACSModel. Fix `key_remap` above.")
    else:
        print("[checkpoint] Task 2 ERM checkpoint loaded cleanly.")
    return model

