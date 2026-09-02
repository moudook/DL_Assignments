from __future__ import annotations

import os
from typing import Sequence

import numpy as np


DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "Group11",
)


def load_ls_data(data_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    data_dir = data_dir or os.path.join(DATA_ROOT, "Classification", "LS_Group11")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}. Did you extract the data?"
        )

    X_parts, y_parts = [], []
    for label, filename in enumerate(files):
        data = np.loadtxt(os.path.join(data_dir, filename))
        X_parts.append(data)
        y_parts.append(np.full(len(data), label))

    return np.vstack(X_parts), np.concatenate(y_parts)


def load_nls_data(
    filepath: str | None = None,
    class_sizes: Sequence[int] = (500, 500, 700),
) -> tuple[np.ndarray, np.ndarray]:
    filepath = filepath or os.path.join(
        DATA_ROOT, "Classification", "NLS_Group11.txt"
    )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.loadtxt(filepath, skiprows=1)
    if data.shape[0] != sum(class_sizes):
        raise ValueError(
            f"Expected {sum(class_sizes)} rows in {filepath}, got {data.shape[0]}"
        )

    X_parts, y_parts = [], []
    start = 0
    for label, size in enumerate(class_sizes):
        X_parts.append(data[start : start + size])
        y_parts.append(np.full(size, label))
        start += size

    return np.vstack(X_parts), np.concatenate(y_parts)


def load_univariate_data(filepath: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    filepath = filepath or os.path.join(
        DATA_ROOT, "Regression", "UnivariateData", "11.csv"
    )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.loadtxt(filepath, delimiter=",")
    return data[:, :-1].reshape(-1, 1), data[:, -1]


def load_bivariate_data(filepath: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    filepath = filepath or os.path.join(
        DATA_ROOT, "Regression", "BivariateData", "11.csv"
    )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.loadtxt(filepath, delimiter=",")
    return data[:, :-1], data[:, -1]


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray | None = None,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    stratify: bool = True,
) -> tuple:
    train_r, val_r, test_r = ratios
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if not stratify or y is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_tr = int(round(n * train_r))
        n_va = int(round(n * val_r))
        train_idx = idx[:n_tr]
        val_idx = idx[n_tr : n_tr + n_va]
        test_idx = idx[n_tr + n_va :]
        if y is None:
            return (
                X[train_idx],
                X[val_idx],
                X[test_idx],
            )
        return (
            X[train_idx],
            X[val_idx],
            X[test_idx],
            y[train_idx],
            y[val_idx],
            y[test_idx],
        )

    train_idx_list, val_idx_list, test_idx_list = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * train_r))
        n_va = int(round(len(idx) * val_r))
        train_idx_list.append(idx[:n_tr])
        val_idx_list.append(idx[n_tr : n_tr + n_va])
        test_idx_list.append(idx[n_tr + n_va :])

    train_idx = np.concatenate(train_idx_list)
    val_idx = np.concatenate(val_idx_list)
    test_idx = np.concatenate(test_idx_list)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    out = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out