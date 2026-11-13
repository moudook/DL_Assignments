"""Reading, normalising and batching the two sequence datasets."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import config


# ---------------- Reading one file ----------------

def read_hw_file(path):
    """Handwriting file: first number N, then N (x, y) pairs. Returns an (N, 2) array."""
    values = Path(path).read_text(encoding="utf-8").split()
    n_points = int(float(values[0]))
    coords = np.array(values[1:], dtype=np.float64)
    if coords.size != 2 * n_points:
        raise ValueError(f"{path}: header says {n_points} points but file has {coords.size} numbers")
    return coords.reshape(n_points, 2)


def read_cv_file(path):
    """CV file: one 39-dimensional MFCC vector per row. Returns a (T, 39) array."""
    frames = np.loadtxt(path, ndmin=2)
    if frames.shape[1] != 39:
        raise ValueError(f"{path}: expected 39 columns, found {frames.shape[1]}")
    if not np.isfinite(frames).all():
        raise ValueError(f"{path}: contains NaN or inf")
    return frames


# ---------------- Handwriting normalisation ----------------

def minmax_per_axis(points):
    """Scale x and y of one character separately to [0, 1] (the sheet's normalisation)."""
    low = points.min(axis=0)
    span = points.max(axis=0) - low
    out = np.full_like(points, 0.5)  # an axis with no spread is placed in the middle
    for axis in range(2):
        if span[axis] > 0:
            out[:, axis] = (points[:, axis] - low[axis]) / span[axis]
    return out


def minmax_joint(points):
    """Alternative: one scale for both axes, so the character keeps its shape."""
    low = points.min(axis=0)
    span = (points.max(axis=0) - low).max()
    if span == 0:
        return np.full_like(points, 0.5)
    return (points - low) / span


def no_normalisation(points):
    return points


HW_NORMALISERS = {
    "minmax_axis": minmax_per_axis,
    "minmax_joint": minmax_joint,
    "none": no_normalisation,
}


# ---------------- Loading a split ----------------

@dataclass
class Sample:
    x: np.ndarray  # (T, input_dim)
    label: int
    path: Path


def read_sample(spec, path):
    if spec.name == "hw":
        return HW_NORMALISERS[config.HW_NORM](read_hw_file(path))
    return read_cv_file(path)


def load_split(spec, split):
    """Load every file of one split ("train" or "test"). The label comes from the class folder name."""
    classes = config.get_classes(spec)
    split_dir = spec.train_dir if split == "train" else spec.test_dir
    samples = []
    for label, class_name in enumerate(classes):
        folder = config.DATA_DIR / spec.folder / class_name / split_dir
        files = sorted(folder.glob(spec.pattern))
        if not files:
            raise FileNotFoundError(f"No {spec.pattern} files in {folder}")
        for path in files:
            samples.append(Sample(read_sample(spec, path), label, path))
    return samples


def stratified_split(samples, frac, seed):
    """Hold out `frac` of each class (chosen at random with `seed`) as the validation set."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    labels = sorted({s.label for s in samples})
    for label in labels:
        class_samples = [s for s in samples if s.label == label]
        order = rng.permutation(len(class_samples))
        n_val = int(round(frac * len(class_samples)))
        for rank, index in enumerate(order):
            if rank < n_val:
                val.append(class_samples[index])
            else:
                train.append(class_samples[index])
    return train, val


def save_split(train, val, path):
    """Record which files went to train and val (paths relative to the data folder)."""
    def names(samples):
        return sorted(s.path.relative_to(config.DATA_DIR).as_posix() for s in samples)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": names(train), "val": names(val)}, indent=1), encoding="utf-8")


# ---------------- Feature standardisation (CV) ----------------

class Standardiser:
    """Per-dimension z-score using statistics of the training split only.

    mode "zscore_train": x -> (x - mean) / std, mean and std over all training frames.
    mode "none":         x unchanged.
    """

    def __init__(self, mode):
        if mode not in ("zscore_train", "none"):
            raise ValueError(f"Unknown normalisation mode {mode}")
        self.mode = mode
        self.mean = None
        self.std = None

    def fit(self, samples):
        if self.mode == "zscore_train":
            frames = np.concatenate([s.x for s in samples], axis=0)
            self.mean = frames.mean(axis=0)
            self.std = frames.std(axis=0)
            self.std[self.std == 0] = 1.0
        return self

    def transform(self, x):
        if self.mode == "none":
            return x
        return (x - self.mean) / self.std

    def state_dict(self):
        return {
            "mode": self.mode,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state):
        standardiser = cls(state["mode"])
        if state["mean"] is not None:
            standardiser.mean = np.array(state["mean"])
            standardiser.std = np.array(state["std"])
        return standardiser


# ---------------- PyTorch dataset and batching ----------------

class SeqDataset(Dataset):
    def __init__(self, samples, standardiser):
        self.xs = [torch.tensor(standardiser.transform(s.x), dtype=torch.float32) for s in samples]
        self.labels = [s.label for s in samples]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.labels[index]


def collate(batch):
    """Pad the sequences of a batch to the same length and keep their true lengths.

    The padding values are never read by the model: it packs the batch using `lengths`.
    """
    xs = [x for x, _ in batch]
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    padded = pad_sequence(xs, batch_first=True)
    return padded, lengths, labels


def make_loader(dataset, batch_size, shuffle, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator, num_workers=0)


def prepare_data(spec):
    """Load one dataset and return train/val/test datasets plus everything needed to describe them."""
    all_train = load_split(spec, "train")
    test = load_split(spec, "test")
    train, val = stratified_split(all_train, config.VAL_FRAC, config.SPLIT_SEED)
    save_split(train, val, config.dataset_out_dir(spec) / "split.json")

    mode = config.CV_NORM if spec.name == "cv" else "none"
    standardiser = Standardiser(mode).fit(train)
    return {
        "classes": config.get_classes(spec),
        "train_samples": train,
        "val_samples": val,
        "test_samples": test,
        "train": SeqDataset(train, standardiser),
        "val": SeqDataset(val, standardiser),
        "test": SeqDataset(test, standardiser),
        "standardiser": standardiser,
    }


# ---------------- Dataset statistics ----------------

def dataset_stats(spec):
    """Class counts and sequence lengths per split (files are validated while reading)."""
    classes = config.get_classes(spec)
    stats = {"classes": classes, "splits": {}}
    for split in ("train", "test"):
        samples = load_split(spec, split)
        per_class = {}
        for label, class_name in enumerate(classes):
            lengths = [len(s.x) for s in samples if s.label == label]
            per_class[class_name] = {
                "count": len(lengths),
                "min_len": int(min(lengths)),
                "median_len": float(np.median(lengths)),
                "max_len": int(max(lengths)),
                "lengths": lengths,
            }
        stats["splits"][split] = per_class
    return stats
