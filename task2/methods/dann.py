# ══════════════════════════════════════════════════════════════════════════
# task2/methods/dann.py — DANN (adversarial alignment)
# ══════════════════════════════════════════════════════════════════════════
"""
DANN has no method-specific math of its own beyond the gradient-reversal
layer and the domain discriminator, which the assignment's own repository
layout places in models/ (see task2/models/domain_discriminator.py:
GradientReversalLayer, DomainDiscriminator, grl_alpha_schedule).

The assignment's protocol requires every method to "run through the same
training and evaluation pipeline" (Task 2, Dataset/Models/Experimental Setup),
so the DANN-specific assembly (L_cls + domain loss, alpha schedule, only
source examples contribute to the class loss) lives inside the shared
generic training loop in task2/train.py -> train_method(method="dann", ...).

This file re-exports the pieces DANN needs so `from task2.methods import dann`
gives you everything relevant to the method in one place.
"""
from task2.models.domain_discriminator import (
    GradientReversalFunction,
    GradientReversalLayer,
    grl_alpha_schedule,
    DomainDiscriminator,
)

__all__ = [
    "GradientReversalFunction",
    "GradientReversalLayer",
    "grl_alpha_schedule",
    "DomainDiscriminator",
]
