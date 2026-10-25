"""
config.py - single source of truth for paths, seeds, hyperparameters and the
fixed image/filter selections every question reuses.

Every rule here is fixed BEFORE training (see README.md, Methodology) so no choice
can be tuned on validation or test results.
"""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
OUT_DIR = CODE_DIR / "out"
CACHE_DIR = OUT_DIR / "cache"

SEED = 11
# Every trained model (A3, Arch-1/2/3, VGG19 head) runs once per seed. One
# seed by default; add more (e.g. [11, 12, 13]) and results become mean +- std
# with selection on the mean. SEEDS[0] is the designated seed - fixed in
# advance, never "the best one" - and supplies every figure, the confusion
# matrices, and the A2 filters A3b is compared against.
SEEDS = [11]
assert SEEDS[0] == SEED

# ---------------------------------------------------------------- datasets
# Both datasets ship as folders named "Group_11", so the resolver identifies
# them by their class folders, not by name. Override with the env vars (or the
# --mnist-dir / --caltech-dir flags of run.py) on a machine with another layout.
MNIST_CLASSES = ["0", "4", "5", "6", "7"]
CALTECH_CLASSES = ["bonsai", "buddha", "car_side", "scorpion", "starfish"]
SPLITS = ("train", "val", "test")

MNIST_CANDIDATES = [
    CODE_DIR / "data" / "mnist",
    CODE_DIR.parent / "Data" / "MNIST",
    CODE_DIR.parent.parent / "Assignment4" / "Data" / "Group_11",
    CODE_DIR.parent.parent / "Assignment3" / "Group_11",
]
CALTECH_CANDIDATES = [
    CODE_DIR / "data" / "caltech",
    CODE_DIR.parent / "Group_11",
    CODE_DIR.parent / "Data" / "Group_11",
]
MNIST_ENV = "A5_MNIST_DIR"
CALTECH_ENV = "A5_CALTECH_DIR"

# Expected split sizes per class, checked by the smoke test.
MNIST_PER_CLASS = {"train": 2277, "val": 759, "test": 759}
CALTECH_PER_CLASS = {"train": 50, "val": 10, "test": 20}

IMG_SIZE_B = 224

# ---------------------------------------------------------------- selections
# Images: first file (sorted by name) of each class in the train split. The
# B1c / B2c single image is the first file of the first colour class (bonsai),
# so colour feature maps and guided backprop have colour to show; car_side is
# entirely greyscale.
SINGLE_IMAGE_CLASS_B = "bonsai"

# A2 shows "any 3 filters" per layer; A3b must show the same indices after
# training. 0-based in code, printed 1-based in figures.
A2_FILTERS_L1 = [0, 2, 4]
A2_FILTERS_L2 = [0, 5, 10]
# A3b: all 6 conv1 filters (the sheet says 10, conv1 only has 6) and 10 conv2
# filters that include the A2 indices.
A3B_FILTERS_L1 = list(range(6))
A3B_FILTERS_L2 = [0, 5, 10, 1, 2, 3, 4, 6, 7, 8]

N_TRACE_NEURONS_A = 5     # A3c: neurons per image
N_GUIDED_NEURONS = 5      # B2c
N_MAPS_B1C = 8            # B1c: maps per later conv layer

# ---------------------------------------------------------------- training
# Mini-batch SGD with momentum (the AlexNet/VGG recipe the notes' case studies
# cite), no weight decay, no schedule. Batch size differs only between
# datasets, which are never compared; every Caltech model uses BATCH_SIZE_B.
BATCH_SIZE_A = 32        # MNIST: ~356 updates/epoch
BATCH_SIZE_B = 16        # Caltech: ~16 updates/epoch (32 would give only 8)
MOMENTUM = 0.9
# Measured epoch times (8-core laptop): Part A 0.81/0.58/0.48/1.07 s and
# Caltech 0.23/0.12/0.08/0.08-0.37 s at 1/2/4/8 threads -> 4 threads.
TORCH_THREADS = min(4, os.cpu_count() or 1)

# Learning rate: each model (A3, Arch-1/2/3, VGG19 head) runs the SAME grid and
# keeps the rate with the lowest early-stopped val loss. Fairness = identical
# procedure, not identical values. Rules fixed before any run:
#   - criterion is val LOSS, never switched to val accuracy afterwards
#     (architecture selection in B1b stays on val accuracy, as the sheet says)
#   - if the winner sits at a grid edge, add exactly one more step in that
#     direction (LR_EDGE_STEP), once, for that model only; nothing else is
#     ever added after seeing results
#   - every run of the grid is reported, not only the winners
LR_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]
LR_EDGE_STEP = {"low": 0.0003, "high": 0.3}
# Early stopping on mean validation cross-entropy: stop after ES_PATIENCE
# epochs without a new minimum, then restore the best epoch's weights. Same
# rule for every model. Neither the sheet nor the notes give a criterion.
ES_PATIENCE = 15
MAX_EPOCHS_A = 200       # safety cap, logged if hit
MAX_EPOCHS_B = 500

QUICK = {                # run.py --quick: pipeline check, numbers meaningless
    "max_epochs": 2,
    "per_class_train": 20,
    "per_class_eval": 10,
}


def resolve_dir(candidates, env_var, classes):
    """First candidate whose train/ holds exactly `classes` as subfolders."""
    tried = []
    env = os.environ.get(env_var)
    paths = ([Path(env)] if env else []) + list(candidates)
    for p in paths:
        tried.append(str(p))
        train = p / "train"
        if train.is_dir():
            found = sorted(d.name for d in train.iterdir() if d.is_dir())
            if found == sorted(classes):
                return p.resolve()
    raise FileNotFoundError(
        "dataset with classes %s not found; tried:\n  %s\nset %s to override"
        % (classes, "\n  ".join(tried), env_var))
