import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import random
from configs.config import SEED

base_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

resnet_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
vit_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
clip_normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

def grayscale_batch(images: torch.Tensor) -> torch.Tensor:
    return TF.rgb_to_grayscale(images, num_output_channels=3)

def palette_transfer_reinhard(content_img: np.ndarray, palette_img: np.ndarray) -> np.ndarray:
    c = cv2.cvtColor(content_img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    p = cv2.cvtColor(palette_img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    out = np.empty_like(c)
    for i in range(3):
        c_std = max(float(c[..., i].std()), 1e-5)
        out[..., i] = (c[..., i] - c[..., i].mean()) * (p[..., i].std() / c_std) + p[..., i].mean()
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    out[..., 1:] = np.clip(out[..., 1:], -128, 127)
    rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)

def generate_balanced_palette_pairs(dataset_dict, num_pairs_target=500, seed=SEED):
    rng = random.Random(seed)
    classes = list(dataset_dict.keys())
    assert len(classes) == 10
    pairs = []
    for c_class in classes:
        other_classes = [oc for oc in classes if oc != c_class]
        for idx, content_img in enumerate(dataset_dict[c_class]):
            p_class = other_classes[idx % len(other_classes)]
            pairs.append({"content_image": content_img,
                          "palette_image": rng.choice(dataset_dict[p_class]),
                          "content_class": c_class, "palette_class": p_class})
    return pairs[:num_pairs_target]

CLASS_PAIRS = [(5, 6), (4, 6), (3, 5), (1, 7), (2, 9)]

def make_cue_pairs(class_to_images, class_pairs=CLASS_PAIRS, n_per_direction=40, seed=SEED):
    rng = random.Random(seed)
    out = []
    for a, b in class_pairs:
        for c, s in ((a, b), (b, a)):
            for _ in range(n_per_direction):
                out.append(dict(content_image=rng.choice(class_to_images[c]),
                                palette_image=rng.choice(class_to_images[s]),
                                content_class=c, palette_class=s))
    return out

def calc_mean_std(features, eps=1e-6):
    b, c = features.size()[:2]
    mean = features.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    std = features.reshape(b, c, -1).std(dim=2).reshape(b, c, 1, 1) + eps
    return mean, std
 
def adain(content_features, style_features):
    c_mean, c_std = calc_mean_std(content_features)
    s_mean, s_std = calc_mean_std(style_features)
    return s_std * (content_features - c_mean) / c_std + s_mean

def translate_batch(images: torch.Tensor, pixels: int, direction: str) -> torch.Tensor:
    if pixels == 0:
        return images
    _, _, H, W = images.shape
    if direction == "right":
        padded = F.pad(images, (pixels, 0, 0, 0), mode="reflect")
        return padded[:, :, :, :W]
    elif direction == "left":
        padded = F.pad(images, (0, pixels, 0, 0), mode="reflect")
        return padded[:, :, :, -W:]
    elif direction == "down":
        padded = F.pad(images, (0, 0, pixels, 0), mode="reflect")
        return padded[:, :, :H, :]
    elif direction == "up":
        padded = F.pad(images, (0, 0, 0, pixels), mode="reflect")
        return padded[:, :, -H:, :]
    else:
        raise ValueError(f"direction must be one of up/down/left/right, got {direction!r}")

DIRECTIONS = ("up", "down", "left", "right")
DISPLACEMENTS = (0, 8, 16, 32)

def _non_identity_permutation(rng: torch.Generator, n: int) -> torch.Tensor:
    identity = torch.arange(n)
    while True:
        perm = torch.randperm(n, generator=rng)
        if not torch.equal(perm, identity):
            return perm

def shuffle_patches_batch(images: torch.Tensor, grid_size: int = 4, seed: int = SEED) -> torch.Tensor:
    B, C, H, W = images.shape
    assert H % grid_size == 0 and W % grid_size == 0, "H, W must be divisible by grid_size"
    ph, pw = H // grid_size, W // grid_size
    n_patches = grid_size * grid_size

    out = images.clone()
    for i in range(B):
        gen = torch.Generator().manual_seed(seed + i)
        perm = _non_identity_permutation(gen, n_patches)
        patches = []
        for k in range(n_patches):
            r, c = divmod(k, grid_size)
            patches.append(images[i, :, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw])
        for k in range(n_patches):
            r, c = divmod(int(perm[k]), grid_size)
            out[i, :, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = patches[k]
    return out