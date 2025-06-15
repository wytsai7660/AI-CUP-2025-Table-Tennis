from pathlib import Path
import sys
import torch

# ===== Paths =====
BASE_PATH = Path(__file__).parent

TRAIN_DATA_DIR = BASE_PATH / "39_Training_Dataset" / "train_data"
TRAIN_INFO = BASE_PATH / "39_Training_Dataset" / "train_info.csv"

TEST_DATA_DIR = BASE_PATH / "39_Test_Dataset" / "test_data"
TEST_INFO = BASE_PATH / "39_Test_Dataset" / "test_info.csv"

# === Pre-Computed Values ===
# Run `helper.pre-compute` to recompute them if needed
MEAN = torch.Tensor(
    [-641.6706, -2168.8293, -106.0658, 2665.5098, 4612.9365, -1239.6418]
)
STD = torch.Tensor(
    [4746.2964, 3781.6528, 2564.6877, 20106.4824, 15148.3379, 21313.9180]
)

# Check if any of the paths are absent
for path in [TRAIN_DATA_DIR, TRAIN_INFO, TEST_DATA_DIR, TEST_INFO]:
    if not path.exists():
        sys.exit(f"{path}: No such file or directory")


# ===== Constants =====
FIELDS = [
    "gender",
    "hold racket handed",
    "play years",
    "level",
]

POSSIBLE_VALUES = [
    [1, 2],
    [1, 2],
    [0, 1, 2],
    [2, 3, 4, 5],
]
