# ══════════════════════════════════════════════════════════════════════════
# task3/models/classifier_head.py
# ══════════════════════════════════════════════════════════════════════════
"""
See the note at the top of task3/models/backbone.py: because the whole
network is fully fine-tuned in Task 3, the backbone and the 7-way linear
classifier head are implemented as one nn.Module, PACSModel, defined in
task3/models/backbone.py. Its final layer, `PACSModel.fc`
(nn.Linear(512, num_classes)), is what plays the "classifier head" role
called out by the assignment's suggested repository structure.

No separate class is defined here to avoid duplicating / diverging from the
working PACSModel implementation.
"""
from task3.models.backbone import PACSModel  # noqa: F401  (re-exported for convenience)
