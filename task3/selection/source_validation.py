# ══════════════════════════════════════════════════════════════════════════
# task3/selection/source_validation.py — early stopping on mean source macro-F1
# ══════════════════════════════════════════════════════════════════════════
import copy
import numpy as np

# =====================================================================================
# [task3/selection/source_validation.py]  — early stopping on mean source macro-F1
# =====================================================================================
class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self.best_score = -np.inf
        self.best_state = None
        self.counter = 0

    def step(self, score, model):
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        stop = self.counter >= self.patience
        return improved, stop



