from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 9,
})


CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


_fig_counter = itertools.count(1)


def _next_fig_number() -> int:
    return next(_fig_counter)


def _save(save_path: str | None) -> None:
    if not save_path:
        return
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_error_curve(
    train_mse: list[float],
    val_mse: list[float] | None,
    title: str,
    save_path: str | None = None,
    ylabel: str = "Mean Squared Error",
) -> None:
    plt.figure(figsize=(8.0, 4.8))
    epochs = np.arange(len(train_mse))
    plt.plot(epochs, train_mse, lw=2.0, color="#4C72B0", label="Train MSE")
    if val_mse is not None and len(val_mse) == len(train_mse):
        plt.plot(epochs, val_mse, lw=2.0, color="#DD8452", label="Validation MSE")
        best_epoch = int(np.argmin(val_mse))
        plt.axvline(best_epoch, ls=":", color="#55A868", lw=1.2,
                    label=f"Best val @ epoch {best_epoch} (mse={val_mse[best_epoch]:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlim(0, max(1, len(train_mse) - 1))
    plt.tight_layout()
    _save(save_path)


def plot_decision_regions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: str | None = None,
    margin: float = 0.5,
    resolution: int = 400,
) -> None:
    if X.shape[1] != 2:
        raise ValueError(f"X must be 2-D for decision-region plot, got {X.shape[1]}")

    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    classes = np.unique(y)
    cmap = ListedColormap(CLASS_COLORS[: len(classes)])

    plt.figure(figsize=(8.0, 6.0))
    plt.contourf(xx, yy, Z, alpha=0.30,
                 cmap=cmap, levels=np.arange(len(classes) + 1) - 0.5)

    markers = itertools.cycle(["o", "s", "^", "D"])
    for idx, cls in enumerate(classes):
        m = y == cls
        plt.scatter(X[m, 0], X[m, 1],
                    color=CLASS_COLORS[idx],
                    marker=next(markers),
                    edgecolor="black", linewidth=0.5,
                    s=28, alpha=0.85,
                    label=f"Class {cls} (n={m.sum()})")

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(title)
    plt.legend(loc="best", fontsize=8.5)
    plt.tight_layout()
    _save(save_path)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    title: str,
    save_path: str | None = None,
    normalize: bool = True,
) -> None:
    n = cm.shape[0]
    cm_norm = cm / cm.sum(axis=1, keepdims=True) if normalize and cm.sum(axis=1).all() else cm

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())

    for i in range(n):
        for j in range(n):
            row_sum = cm[i, :].sum()
            pct = cm[i, j] / row_sum * 100 if row_sum else 0
            ax.text(j, i,
                    f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=10,
                    fontweight="bold" if i == j else "normal",
                    color="white" if (normalize and cm_norm[i, j] > 0.5) else "#222222")

    ax.set_xticks(range(n), [f"Pred {i}" for i in range(n)])
    ax.set_yticks(range(n), [f"True {i}" for i in range(n)])
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="Row-normalised proportion" if normalize else "Count")
    plt.tight_layout()
    _save(save_path)


def plot_node_surfaces(
    model,
    X: np.ndarray,
    y: np.ndarray | None,
    layer_idx: int,
    node_indices: list[int],
    title_prefix: str,
    save_dir: str,
    resolution: int = 60,
    kind: str = "node",
) -> list[str]:
    """3-D plot of every requested node's output as a surface over (x1, x2).

    `layer_idx=1` means the first hidden layer (just after the input).
    `kind='hidden'` or `'output'` controls only the figure title.
    Returns the list of saved file paths.
    """
    if X.shape[1] != 2:
        raise ValueError(f"X must be 2-D for node-surface plot, got {X.shape[1]}")

    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.c_[xx1.ravel(), xx2.ravel()]

    layer_acts = model.forward(grid)[layer_idx]

    saved = []
    for node_idx in node_indices:
        zz = layer_acts[:, node_idx].reshape(xx1.shape)

        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(xx1, xx2, zz, cmap="viridis",
                               alpha=0.85, edgecolor="k", linewidth=0.1)
        if y is not None:
            layer_at_data = model.forward(X)[layer_idx]
            ax.scatter(X[:, 0], X[:, 1], layer_at_data[:, node_idx],
                       color="#C44E52", s=8, alpha=0.6,
                       label="Data points")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel("Output")
        title = f"{title_prefix} — {kind} node {node_idx} (layer {layer_idx})"
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.6, label="Activation")
        ax.view_init(elev=28, azim=-55)
        plt.tight_layout()

        out_path = os.path.join(
            save_dir,
            f"surface_{kind}_l{layer_idx}_n{node_idx}.png",
        )
        _save(out_path)
        saved.append(out_path)

    return saved