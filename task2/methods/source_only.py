# ══════════════════════════════════════════════════════════════════════════
# task2/methods/source_only.py — Source-only ERM baseline
# ══════════════════════════════════════════════════════════════════════════
"""
Source-only ERM has no method-specific loss term (no alignment term is
added to the cross-entropy classification loss). The assignment requires
every method to share one training/evaluation pipeline, so the branch that
runs plain cross-entropy over the three labeled source domains lives inside
the shared generic loop in task2/train.py -> train_method(method="source_only", ...).

Run it with:
    from task2.train import train_method
    train_method("source_only", "source_only", source_splits, target_samples)

This checkpoint is also reused, unmodified, as the Task 3 ERM baseline
(see task3/methods/erm.py).
"""
