"""
data.py — dataset loading for Assignment-4.

MNIST 5-class subset (Group 11): digits {0, 4, 5, 6, 7}, 28x28 images flattened
to 784-d vectors, pre-split train/val/test.

Split sizes (verified from disk, balanced across all 5 classes):
    train 11,385 (2,277 per class)
    val    3,795 (759 per class)
    test   3,795 (759 per class)

Design notes
------------
Whole dataset is preloaded to the target device once and reused by every model.
Rationale: total footprint is ~56 MB in fp32 (train 34 MB, val 11 MB, test 11 MB),
which is trivially small against 4 GB of VRAM, so there is no reason to re-decode
JPEGs per epoch or per architecture. Across ~60 training runs, reloading would be
pure waste.

num_workers=0 is inherited from Assignment-3 and is required on Windows: with
workers > 0, full-batch collation routes the entire dataset through one named
shared-memory mapping, which previously produced "Couldn't open shared file
mapping ... error code 1455" and MemoryError crashes. Single-process loading
avoids both.
"""

import os

import torch
from torchvision import datasets, transforms

CLASS_NAMES = ["0", "4", "5", "6", "7"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_DIM = 784
IMG_SIZE = 28


class FlattenTransform:
    """(C,H,W) -> (784,). Row-major, so pixel (r,c) lands at index r*28+c."""

    def __call__(self, x):
        return x.view(-1)


def _transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        FlattenTransform(),
    ])


def resolve_data_dir(name="Group_11"):
    """
    Locate the dataset folder.

    Prefers ./<name> (the copy that ships with Assignment-4), then falls back to
    ../Assignment3/<name> since Assignment-3's code references a path that no
    longer exists under its own ./Data directory.
    """
    candidates = [
        name,
        os.path.join("..", name),
        os.path.join("..", "Assignment3", name),
        os.path.join("..", "..", name),
        os.path.join("..", "..", "Assignment3", name),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "train")):
            return os.path.normpath(path)
    raise FileNotFoundError(
        f"Could not find dataset. Tried: {candidates}\n"
        "Expected a folder containing train/ val/ test/ with class subfolders."
    )


def _drain(loader, device):
    """
    Collapse a DataLoader into one (X, y) tensor pair on `device`.

    X: (N, 784) float32, y: (N,) int64.
    """
    xs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xs.append(xb)
            ys.append(yb)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def load_splits(data_dir="Group_11", device=None, seed=42):
    """
    Load train/val/test fully onto `device`.

    Returns a dict with X_train/y_train, X_val/y_val, X_test/y_test, plus the
    class names and split sizes.

    `device=None` resolves to cuda when available, else cpu. Callers pass the
    device explicitly so there is one source of truth for it.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = resolve_data_dir(data_dir)
    tf = _transform()
    pin = device.type == "cuda"

    split_tensors = {}
    for split in ("train", "val", "test"):
        ds = datasets.ImageFolder(root=os.path.join(root, split), transform=tf)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=len(ds), shuffle=False,
            num_workers=0, pin_memory=pin,
        )
        X, y = _drain(loader, device)
        split_tensors[f"X_{split}"] = X
        split_tensors[f"y_{split}"] = y
        split_tensors[f"n_{split}"] = X.size(0)

    split_tensors["classes"] = CLASS_NAMES
    split_tensors["num_classes"] = NUM_CLASSES
    split_tensors["input_dim"] = INPUT_DIM
    split_tensors["device"] = device
    split_tensors["data_dir"] = root
    return split_tensors


def one_per_class(data, split):
    """
    First sample of each class, as (indices, labels).

    Used for the reconstruction figures the assignment requires: one image from
    each class, from each of train/val/test.
    """
    y = data[f"y_{split}"]
    idx, labels = [], []
    for c in range(NUM_CLASSES):
        hits = (y == c).nonzero(as_tuple=True)[0]
        if hits.numel() == 0:
            raise ValueError(f"class {c} absent from {split} split")
        idx.append(hits[0].item())
        labels.append(c)
    return idx, labels


def class_counts(data):
    """Per-class sample counts, for the report's dataset table."""
    out = {}
    for split in ("train", "val", "test"):
        y = data[f"y_{split}"]
        out[split] = [int((y == c).sum().item()) for c in range(NUM_CLASSES)]
    return out
