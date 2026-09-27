Task 1 - Inductive Biases and Feature Representations

This repository contains the code for Task 1 of Programming Assignment 1, investigating the inductive biases (shape, texture, color) of ResNet-50, ViT-B/16, and OpenCLIP ViT-B-32.

## Repository Structure
- `configs/`: Hyperparameters, random seeds, and global configurations.
- `data/`: Dataset loading, balanced subset generation, cue conflict generation, and image transforms.
- `models/`: Pretrained backbone initialization, feature extraction, and predictor wrappers.
- `analysis/`: Linear head training and evaluation metrics/logic.
- `scripts/`: The main orchestrator script to run the experiments.

## Setup & Execution
1. Install requirements: `pip install torch torchvision open_clip_torch opencv-python scikit-learn pandas matplotlib`
2. Run the main pipeline from the root of the repository:
   ```bash
   python -m scripts.run_task1