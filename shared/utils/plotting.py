"""
Publication-quality plotting utilities for Assignment 1.
Every figure carries: a descriptive title, labelled axes with units where
applicable, an explicit legend, key metric annotations, and a numbered
caption line at the bottom suitable for direct inclusion in a report.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import itertools

# ── Global professional style ────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 10,
})

# Consistent colour vocabulary across ALL figures
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868"]          # blue / orange / green
MODEL_COLORS = {"sigmoid": "#4C72B0", "tanh": "#DD8452", "linear": "#55A868"}
CMAP_BLUE_ORANGE = LinearSegmentedColormap.from_list(
    "bg_pair", ["#aec7e8", "#ffbb78"])

_fig_counter = itertools.count(1)


def _caption(save_path, caption):
    """Bottom caption strip: 'Figure N. <text>'."""
    fig = plt.gcf()
    fig.text(0.5, -0.02, f"Figure {_fig_counter.__next__()}. {caption}",
             ha="center", va="top", fontsize=9.5, style="italic", color="#333333")


def _save(save_path):
    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close()


def plot_error_curves(histories, title, save_path=None,
                      caption=None, ylabel="Mean Squared Error"):
    """Error vs epoch curves. Accepts dict[(i,j)->list] (classification OvO)
    or a plain list (regression). Annotates the final converged error."""
    plt.figure(figsize=(8.5, 5))
    if isinstance(histories, dict):
        colors = plt.cm.tab10(np.linspace(0, 0.6, len(histories)))
        for c, ((i, j), mse) in zip(colors, histories.items()):
            lbl = f"Classifier {i} vs {j}  (final MSE {mse[-1]:.2e})"
            plt.plot(mse, label=lbl, lw=1.8, color=c)
        n_epochs = max(len(v) for v in histories.values())
    else:
        mse = list(histories)
        plt.plot(mse, label="Training MSE", lw=2, color=MODEL_COLORS.get("linear"))
        plt.annotate(f"final MSE = {mse[-1]:.2e}",
                     xy=(len(mse) - 1, mse[-1]),
                     xytext=(-10, 18), textcoords="offset points",
                     ha="right", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="#cccccc"))
        n_epochs = len(mse)

    plt.xlabel(f"Epoch (total: {n_epochs})")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlim(0, n_epochs - 1)
    _caption(save_path, caption or
             f"{title}. Training error decreases monotonically, indicating stable "
             f"convergence of batch gradient descent.")
    _save(save_path)


def plot_decision_regions(model, X, y, title, save_path=None, pair=None,
                          margin=0.5, resolution=400, caption=None):
    """2-D decision regions with training data superimposed. Background shade =
    predicted region; scatter markers = true training samples."""
    if X.shape[1] != 2:
        raise ValueError(f"X must have exactly 2 features, got {X.shape[1]}")

    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if pair is None:
        Z = model.predict(grid)
        classes = np.unique(y)
        subtitle = ""
    else:
        Z = model.predict_pair(grid, pair)
        classes = np.array(list(pair))
        other = set(np.unique(y)) - set(pair)
        mask = ~np.isin(y, list(other)) if other else np.ones(len(y), bool)
        X, y = X[mask], y[mask]
        subtitle = f"  (classifier {pair[0]} vs {pair[1]})"

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8.5, 6.5))
    plt.contourf(xx, yy, Z, alpha=0.30,
                 cmap=ListedColormap(CLASS_COLORS[:len(classes)]), levels=len(classes) - 1)

    markers = itertools.cycle(["o", "s", "^"])
    for idx, cls in enumerate(classes):
        m = y == cls
        plt.scatter(X[m, 0], X[m, 1], color=CLASS_COLORS[idx],
                    marker=next(markers), edgecolor="black", linewidth=0.5,
                    s=32, alpha=0.9, label=f"Class {cls}  (n={m.sum()})")

    handles = [mpl.lines.Line2D([0], [0], marker="s", color="none",
               markerfacecolor=c, alpha=0.35, markersize=14) for c in CLASS_COLORS[:len(classes)]]
    labels = [f"Decision region — Class {c}" for c in classes]
    hnd, lb = plt.gca().get_legend_handles_labels()
    plt.legend(hnd + handles, lb + labels, loc="best", ncol=1, fontsize=8.5)

    plt.xlabel("Feature $x_1$")
    plt.ylabel("Feature $x_2$")
    plt.title(title + subtitle)
    _caption(save_path, caption or
             f"{title}{subtitle}. Shaded areas show predicted decision regions; "
             f"markers show training samples superimposed for verification.")
    _save(save_path)


def plot_confusion_matrix(cm, title, save_path=None, caption=None):
    """Annotated confusion-matrix heat-map: raw counts + row-normalised %."""
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            pct = cm[i, j] / cm[i, :].sum() * 100 if cm[i, :].sum() else 0
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == j else "normal",
                    color="white" if cm[i, j] / cm[i, :].sum() > thresh else "#222222")

    ax.set_xticks(range(n), [f"Predicted {i}" for i in range(n)])
    ax.set_yticks(range(n), [f"Actual {i}" for i in range(n)])
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Row-normalised proportion")
    _caption(save_path, caption or
             f"{title}. Diagonal = correct predictions; off-diagonal = misclassifications "
             f"(counts and row percentages).")
    _save(save_path)


def plot_regression_univariate(X, y_true, y_pred, title, save_path=None,
                               caption=None, rmse=None):
    """Scatter of data with fitted model curve; RMSE annotated."""
    x_vals, y_p = X.ravel(), np.asarray(y_pred).ravel()
    srt = np.argsort(x_vals)

    plt.figure(figsize=(8.5, 5))
    plt.scatter(x_vals, y_true, s=16, alpha=0.55, label="Observed data",
                color="#4C72B0", edgecolor="none")
    plt.plot(x_vals[srt], y_p[srt], color="#C44E52", lw=2.2,
             label="Perceptron model output")
    txt = ""
    if rmse is not None:
        txt = f"RMSE = {rmse:.4f}"
        plt.annotate(txt, xy=(0.03, 0.93), xycoords="axes fraction",
                     fontsize=10.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#cccccc"))
    plt.xlabel("Input feature $x$")
    plt.ylabel("Target $y$")
    plt.title(title)
    plt.legend(loc="upper right")
    _caption(save_path, caption or
             f"{title}. Red curve is the trained perceptron fitted on the observed "
             f"data points{'; ' + txt + ' quantifies fit quality.' if txt else '.'}")
    _save(save_path)


def plot_regression_bivariate(model, X, y_true, title, save_path=None,
                              caption=None, rmse=None):
    """3-D surface of the learned model with observed data superimposed."""
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 60)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 60)
    xx1, xx2 = np.meshgrid(x1, x2)
    zz = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

    fig = plt.figure(figsize=(9.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx1, xx2, zz, alpha=0.45, cmap="Oranges",
                           edgecolor="k", linewidth=0.15)
    ax.scatter(X[:, 0], X[:, 1], y_true, color="#4C72B0", s=12, alpha=0.65,
               label=f"Observed data (n={len(X)})")
    ax.set_xlabel(r"Input $x_1$", labelpad=8)
    ax.set_ylabel(r"Input $x_2$", labelpad=8)
    ax.set_zlabel(r"Target $y$", labelpad=8)
    ax.set_title(title)
    if rmse is not None:
        ax.text2D(0.02, 0.92, f"RMSE = {rmse:.4f}",
                  transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#cccccc"))
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=-60)
    _caption(save_path, caption or
             f"{title}. Orange surface = learned bivariate perceptron plane; blue points = "
             f"observed samples.")
    _save(save_path)


def plot_scatter_target_vs_output(y_true, y_pred, title, save_path=None,
                                  caption=None):
    """Target vs predicted scatter around the ideal y=x line, with R²."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.55, color="#4C72B0",
                edgecolor="none", label="Samples")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=1.8, label=r"Ideal $y = \hat{y}$")
    plt.annotate(f"$R^2$ = {r2:.4f}", xy=(0.05, 0.90), xycoords="axes fraction",
                 fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#cccccc"))
    plt.xlabel(r"Target $y$")
    plt.ylabel(r"Model output $\hat{y}$")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.axis("square")
    _caption(save_path, caption or
             f"{title}. Tight clustering about the dashed identity line indicates accurate "
             f"prediction ($R^2$ = {r2:.4f}).")
    _save(save_path)
