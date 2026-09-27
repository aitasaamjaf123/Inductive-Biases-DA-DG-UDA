### `task1/configs/config.py`
import torch

SEED = 6304
NUM_WORKERS = 4
BATCH_SIZE_EXTRACT = 256
DATA_ROOT = "./stl10_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("WARNING: no CUDA device found — this run will be slow.")