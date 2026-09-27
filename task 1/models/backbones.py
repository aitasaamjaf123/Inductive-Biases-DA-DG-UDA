import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import open_clip
from torch.utils.data import DataLoader
from configs.config import device, NUM_WORKERS, BATCH_SIZE_EXTRACT
from data.transforms import resnet_normalize, vit_normalize, clip_normalize, base_transform
import torchvision.transforms as T

def extract_features(dataset, transform, extractor_fn, batch_size=BATCH_SIZE_EXTRACT):
    dataset.transform = transform
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    feats, lbls = [], []
    with torch.inference_mode():
        for images, targets in loader:
            feats.append(extractor_fn(images.to(device, non_blocking=True)).cpu())
            lbls.append(targets)
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)

def vit_cls_token(model, x):
    x = model._process_input(x)
    x = torch.cat([model.class_token.expand(x.shape[0], -1, -1), x], dim=1)
    return model.encoder(x)[:, 0]

def setup_backbones_and_extract(raw_dataset):
    print("Extracting features...")
    resnet_w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    resnet_full = torchvision.models.resnet50(weights=resnet_w).to(device).eval()
    resnet_bb = nn.Sequential(*list(resnet_full.children())[:-1])
    for p in resnet_bb.parameters():
        p.requires_grad_(False)

    resnet_feats, resnet_labels = extract_features(
        raw_dataset,
        T.Compose([base_transform, resnet_normalize]),
        lambda x: resnet_bb(x).squeeze(-1).squeeze(-1),
    )

    vit_w = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    vit_m = torchvision.models.vit_b_16(weights=vit_w).to(device).eval()
    for p in vit_m.parameters():
        p.requires_grad_(False)

    vit_feats, vit_labels = extract_features(
        raw_dataset,
        T.Compose([base_transform, vit_normalize]),
        lambda x: vit_cls_token(vit_m, x),
    )

    clip_m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_m = clip_m.to(device).eval()
    for p in clip_m.parameters():
        p.requires_grad_(False)

    clip_feats, clip_labels = extract_features(
        raw_dataset, T.Compose([base_transform, clip_normalize]), lambda x: F.normalize(clip_m.encode_image(x), dim=-1)
    )

    feature_sets = {
        "ResNet-50": (resnet_feats, resnet_labels),
        "ViT-B/16": (vit_feats, vit_labels),
        "OpenCLIP ViT-B-32": (clip_feats, clip_labels),
    }
    
    return feature_sets, resnet_bb, vit_m, clip_m

def get_model_predictors(clip_m, text_features, resnet_bb, classifier_resnet, vit_m, classifier_vit, classifier_clip):
    logit_scale = clip_m.logit_scale.exp()
 
    def resnet_predictor(images):
        x = resnet_normalize(images).to(device)
        feats = resnet_bb(x).squeeze(-1).squeeze(-1)
        probs = F.softmax(classifier_resnet(feats), dim=-1)
        return {
            "preds": probs.argmax(dim=1).cpu(),
            "confs": probs.max(dim=1).values.cpu(),
            "feats": feats.cpu(),
        }
 
    def vit_predictor(images):
        x = vit_normalize(images).to(device)
        feats = vit_cls_token(vit_m, x)
        probs = F.softmax(classifier_vit(feats), dim=-1)
        return {
            "preds": probs.argmax(dim=1).cpu(),
            "confs": probs.max(dim=1).values.cpu(),
            "feats": feats.cpu(),
        }
 
    def clip_probe_predictor(images):
        x = clip_normalize(images).to(device)
        feats = F.normalize(clip_m.encode_image(x), dim=-1)
        probs = F.softmax(classifier_clip(feats), dim=-1)
        return {
            "preds": probs.argmax(dim=1).cpu(),
            "confs": probs.max(dim=1).values.cpu(),
            "feats": feats.cpu(),
        }
 
    def clip_zeroshot_predictor(images):
        x = clip_normalize(images).to(device)
        feats = F.normalize(clip_m.encode_image(x), dim=-1)
        scaled_sim = logit_scale * (feats @ text_features.T)
        probs = F.softmax(scaled_sim, dim=-1)
        return {
            "preds": probs.argmax(dim=1).cpu(),
            "confs": probs.max(dim=1).values.cpu(),
            "feats": feats.cpu(),
        }
 
    return {
        "ResNet-50 (linear probe)": resnet_predictor,
        "ViT-B/16 (linear probe)": vit_predictor,
        "OpenCLIP (linear probe)": clip_probe_predictor,
        "CLIP (zero-shot)": clip_zeroshot_predictor,
    }