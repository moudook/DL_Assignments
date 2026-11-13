"""All settings for Assignment 6 in one place.

Every value here is a knob: change it here (or via run.py options) and re-run.
"""
import re
from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
# Data and outputs live next to the code folder, so the submitted zip stays small.
DATA_DIR = CODE_DIR.parent / "data"
OUT_DIR = CODE_DIR.parent / "out"

GROUP = 11

# ---------------- Data ----------------
VAL_FRAC = 0.2          # fraction of each class's train files held out for model selection
SPLIT_SEED = 11         # seed for the train/val split (fixed, independent of the training seeds)
HW_NORM = "minmax_axis"  # handwriting normalisation, see data.HW_NORMALISERS
CV_NORM = "zscore_train"  # CV normalisation, see data.Standardiser
N_GALLERY = 6           # handwriting images per class in the report (sheet: at least 5)

# ---------------- Training ----------------
SEEDS = [11]            # one run per seed for every config
TOL = 1e-4              # sheet: stop when |E(k) - E(k-1)| < 1e-4
MAX_EPOCHS = 2000       # safety cap only, not the stopping criterion (pilot runs needed <= 499 epochs)
BATCH_SIZE = 32
OPTIMIZER = "adam"      # "adam" or "sgd"
LR = 1e-3               # decided after the pilot (1e-4 and 3e-4 stopped undertrained)
CLIP = None             # gradient-norm clipping; None = off (not in the slides)
EPOCH_ERROR = "running"  # mean error of the batches seen during the epoch (sheet, as written)

# ---------------- Hyperparameter grid ----------------
# One entry per network: (dense_sizes, hidden_sizes)
#   dense_sizes:  neurons in each hidden layer of the network at a time step, before the recurrent layer
#                 (RNN slide 20: s_t is the "hidden layer (last)"); [] = the input goes straight in
#   hidden_sizes: neurons in each stacked recurrent (RNN/LSTM) layer
GRID = [
    ([], [32]), ([], [64]), ([], [128]),                # 1 recurrent layer, three widths
    ([], [32, 32]), ([], [64, 64]), ([], [128, 128]),   # 2 recurrent layers, three widths
    ([], [64, 64, 64]),                                 # 3 recurrent layers
    ([], [128, 64]),                                    # different width per recurrent layer
    ([32], [64]), ([64], [64]),                         # 1 hidden layer at each time step, two widths
    ([64, 64], [64]),                                   # 2 hidden layers at each time step
]
CELLS = ["rnn", "lstm"]


@dataclass
class DatasetSpec:
    name: str          # short name used in output paths: "hw" or "cv"
    folder: str        # sub-folder of DATA_DIR
    mapping_file: str  # file listing the classes of every group
    train_dir: str     # name of the train sub-folder inside each class folder
    test_dir: str      # name of the test sub-folder inside each class folder
    pattern: str       # file pattern of one sample
    input_dim: int     # size of one time step


DATASETS = {
    "hw": DatasetSpec("hw", "hw", "Mapping_Handwritten.txt", "train", "dev", "*.txt", 2),
    "cv": DatasetSpec("cv", "cv", "Mapping_CV.txt", "Train", "Test", "*.mfcc", 39),
}


def parse_group_classes(text, group=GROUP):
    """Return the class names listed for `group` in a mapping file, sorted.

    Handles both formats used in the two mapping files:
        "Group 11: a, ai, chA, dA, tA"
        "Group-11 ba, pa, bhA, paa, hai"
    """
    for line in text.splitlines():
        match = re.match(r"\s*Group[\s-]*(\d+)\s*:?\s*(.+)", line)
        if match and int(match.group(1)) == group:
            names = [name.strip() for name in match.group(2).split(",")]
            return sorted(name for name in names if name)
    raise ValueError(f"Group {group} not found in mapping file")


def get_classes(spec):
    mapping_path = DATA_DIR / spec.folder / spec.mapping_file
    return parse_group_classes(mapping_path.read_text(encoding="utf-8"))


def dataset_out_dir(spec):
    """Output folder of one dataset. The normalisation scheme and learning rate are part of the name,
    so runs with different settings never overwrite each other."""
    norm = HW_NORM if spec.name == "hw" else CV_NORM
    return OUT_DIR / f"{spec.name}_{norm}_lr{LR:g}"
