import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import STL10
import open_clip
import numpy as np
import os

from configs.config import SEED, DATA_ROOT, NUM_WORKERS, device
from data.make_subset import load_train_data, build_class_balanced_test_subset
from data.transforms import base_transform, clip_normalize, grayscale_batch, generate_balanced_palette_pairs, palette_transfer_reinhard
from models.backbones import setup_backbones_and_extract, get_model_predictors
from analysis.evaluate_bias import train_head, evaluate_predictions, transformed_loader

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Load Data and Extract Backbone Features
    raw_dataset, train_idxs, val_idxs, CLASS_NAMES = load_train_data()
    feature_sets, resnet_bb, vit_m, clip_m = setup_backbones_and_extract(raw_dataset)

    # 2. Train Linear Heads
    print("\n--- Training Linear Heads ---")
    trained_classifiers = {}
    val_accuracies = {}
    for name, (feats, lbls) in feature_sets.items():
        clf, acc = train_head(feats, lbls, train_idxs, val_idxs)
        trained_classifiers[name] = clf
        val_accuracies[name] = acc
        print(f"{name} Validation Accuracy: {acc * 100:.2f}%")

    classifier_resnet = trained_classifiers["ResNet-50"]
    classifier_vit = trained_classifiers["ViT-B/16"]
    classifier_clip = trained_classifiers["OpenCLIP ViT-B-32"]

    # 3. OpenCLIP Zero-Shot Setup
    print("\n--- Evaluating OpenCLIP Zero-Shot ---")
    prompts = [f"a photo of a {c}." for c in CLASS_NAMES]
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.inference_mode():
        text_features = F.normalize(clip_m.encode_text(tokenizer(prompts).to(device)), dim=-1)

    zs_dataset = STL10(root=DATA_ROOT, split="train", transform=torchvision.transforms.Compose([base_transform, clip_normalize]), download=False)
    val_loader_zs = DataLoader(Subset(zs_dataset, val_idxs), batch_size=256, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    
    correct, total = 0, 0
    with torch.inference_mode():
        for images, targets in val_loader_zs:
            sim = F.normalize(clip_m.encode_image(images.to(device)), dim=-1) @ text_features.T
            correct += (sim.argmax(dim=1) == targets.to(device)).sum().item()
            total += targets.size(0)
    print(f"OpenCLIP Zero-Shot Validation Accuracy: {(correct / total) * 100:.2f}%")

    # 4. Building Subsets
    test_dataset_common, test_loader, selected_indices = build_class_balanced_test_subset(device, CLASS_NAMES)
    predictors = get_model_predictors(clip_m, text_features, resnet_bb, classifier_resnet, vit_m, classifier_vit, classifier_clip)
    
    all_experiment_features, all_experiment_preds, all_experiment_metrics = {}, {}, {}

    # 5. Clean Baseline (Step 1)
    print("\n==================================================")
    print("STARTING EXPERIMENTAL EVALUATION PIPELINE")
    print("==================================================")
    print("\n[Step 1/5] Clean Baseline (500 fixed test images)...")
    clean_results = evaluate_predictions(test_loader, predictors, return_features=True)

    clean_preds = {name: res["preds"] for name, res in clean_results.items()}
    all_experiment_features["clean"] = {name: res["feats"] for name, res in clean_results.items()}
    all_experiment_preds["clean"] = clean_preds
    all_experiment_metrics["clean"] = {
        name: {k: v for k, v in res.items() if k not in ["feats", "preds"]} for name, res in clean_results.items()
    }
    for name, m in all_experiment_metrics["clean"].items():
        print(f"  {name} -> Acc: {m['top1_accuracy']*100:.2f}% | Macro-F1: {m['macro_f1']:.4f} | Conf: {m['mean_max_confidence']:.4f}")

    # 6. Color Bias (Step 2)
    print("\n[Step 2/5] Color Bias...")
    print("  -> Grayscale...")
    gray_results = evaluate_predictions(transformed_loader(test_loader, grayscale_batch), predictors, baseline_preds=clean_preds, return_features=True)
    all_experiment_features["grayscale"] = {name: res["feats"] for name, res in gray_results.items()}
    all_experiment_preds["grayscale"] = {name: res["preds"] for name, res in gray_results.items()}
    all_experiment_metrics["grayscale"] = {
        name: {k: v for k, v in res.items() if k not in ["feats", "preds"]} for name, res in gray_results.items()
    }

    # Palette transfer logic (abbreviated inline to execute transforms)
    print("  -> Palette transfer...")
    fixed_class_to_images = {c: [] for c in range(10)}
    for idx in selected_indices:
        img, target = test_dataset_common[idx]
        fixed_class_to_images[target].append(img)
    palette_pairs = generate_balanced_palette_pairs(fixed_class_to_images, num_pairs_target=500, seed=SEED)

    # Simplified inline Dataset class to keep script localized
    class PaletteTransferDataset(torch.utils.data.Dataset):
        def __init__(self, pairs): self.pairs = pairs
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            p = self.pairs[i]
            c_np = (p["content_image"].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            p_np = (p["palette_image"].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            stylized = palette_transfer_reinhard(c_np, p_np)
            return torch.from_numpy(stylized.astype("float32") / 255.0).permute(2, 0, 1), p["content_class"], p["palette_class"]

    palette_loader_raw = DataLoader(PaletteTransferDataset(palette_pairs), batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    palette_imgs, content_lbls, palette_lbls = [], [], []
    for images, c_cls, p_cls in palette_loader_raw:
        palette_imgs.append(images); content_lbls.append(c_cls); palette_lbls.append(p_cls)
    all_palette_images = torch.cat(palette_imgs, dim=0)
    all_content_labels = torch.cat(content_lbls, dim=0).numpy()
    all_palette_labels = torch.cat(palette_lbls, dim=0).numpy()

    palette_eval_loader = DataLoader(TensorDataset(all_palette_images, torch.from_numpy(all_content_labels)), batch_size=128, shuffle=False)
    palette_std = evaluate_predictions(palette_eval_loader, predictors, baseline_preds=clean_preds, return_features=True)
    palette_bias = evaluate_predictions(palette_eval_loader, predictors, shape_labels=all_content_labels, texture_labels=all_palette_labels, baseline_preds=None, return_features=False)
    
    for name in predictors:
        palette_std[name]["color_bias_pct"] = 100.0 - palette_bias[name]["shape_bias_pct"]
        palette_std[name]["coverage_pct"] = palette_bias[name]["coverage_pct"]
        palette_std[name]["n_content"] = palette_bias[name]["n_shape"]
        palette_std[name]["n_color"] = palette_bias[name]["n_texture"]
        palette_std[name]["n_other"] = palette_bias[name]["n_other"]

    all_experiment_features["palette"] = {name: res["feats"] for name, res in palette_std.items()}
    all_experiment_preds["palette"] = {name: res["preds"] for name, res in palette_std.items()}
    all_experiment_metrics["palette"] = {
        name: {k: v for k, v in res.items() if k not in ["feats", "preds"]} for name, res in palette_std.items()
    }
    
    print("\n[Step 2/5] Complete.")

if __name__ == "__main__":
    import torchvision
    main()