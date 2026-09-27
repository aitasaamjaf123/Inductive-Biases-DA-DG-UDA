# ══════════════════════════════════════════════════════════════════════════
# task3/run_task3.py — entry point: run the whole Task 3 pipeline end to end
# ══════════════════════════════════════════════════════════════════════════
# Usage (from the repository root, so the `task3` package is importable):
#     python -m task3.run_task3
#
# Before running: edit task3/shared/pacs_protocol.py's CONFIG (PACS_ROOT,
# and TASK2_CHECKPOINT_PATH if you want main() to reuse the real Task 2
# checkpoint instead of the train_erm_from_scratch fallback — see the note
# at the top of task3/methods/erm.py).
from task3.train import main

if __name__ == "__main__":
    main()
