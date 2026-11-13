"""Every figure of the assignment. Each function draws one figure and saves it as PNG and PDF."""
import matplotlib

matplotlib.use("Agg")  # no display needed; same output on Linux and Windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_error_curve(errors, title, path):
    """Average training error vs epoch for one hyperparameter set (sheet item a)."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    epochs = np.arange(1, len(errors) + 1)
    ax.plot(epochs, errors, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training error")
    ax.set_title(title)
    ax.annotate(f"epoch {len(errors)}\nE = {errors[-1]:.4f}", xy=(epochs[-1], errors[-1]),
                xytext=(-10, 25), textcoords="offset points", ha="right", fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.grid(alpha=0.3)
    save_fig(fig, path)


def plot_curve_overlay(curves, title, path):
    """Several error curves on one axis. curves: list of (label, errors)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colours = plt.cm.tab20(np.linspace(0, 1, max(len(curves), 2)))
    for (label, errors), colour in zip(curves, colours):
        ax.plot(np.arange(1, len(errors) + 1), errors, label=label, color=colour, lw=1.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training error")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    save_fig(fig, path)


def _draw_confusion(ax, matrix, classes, title):
    matrix = np.array(matrix)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percent = 100 * matrix / np.maximum(row_totals, 1)
    ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            colour = "white" if percent[i, j] > 60 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n{percent[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color=colour)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)


def plot_confusion_pair(train_matrix, test_matrix, classes, title, path, train_title="Train"):
    """Train and test confusion matrices of one model: counts and row percentages."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    _draw_confusion(axes[0], train_matrix, classes, train_title)
    _draw_confusion(axes[1], test_matrix, classes, "Test")
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, path)


def plot_length_hist(lengths_by_class, title, xlabel, path):
    """Distribution of sequence lengths per class."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    all_lengths = np.concatenate([np.array(v) for v in lengths_by_class.values()])
    bins = np.linspace(all_lengths.min(), all_lengths.max(), 25)
    for class_name, lengths in lengths_by_class.items():
        ax.hist(lengths, bins=bins, histtype="step", lw=1.5, label=class_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of files")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, path)


def plot_hw_gallery(strokes_by_class, title, path, unit_square=True):
    """Grid of handwritten characters: one row per class, one stroke per panel.

    The line follows the pen; dot colour goes from dark (start) to light (end),
    and the green marker is the pen-down point.
    """
    class_names = list(strokes_by_class)
    n_cols = max(len(v) for v in strokes_by_class.values())
    fig, axes = plt.subplots(len(class_names), n_cols, figsize=(1.7 * n_cols, 1.8 * len(class_names)),
                             squeeze=False)
    for row, class_name in enumerate(class_names):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(strokes_by_class[class_name]):
                ax.axis("off")
                continue
            points = strokes_by_class[class_name][col]
            ax.plot(points[:, 0], points[:, 1], color="0.6", lw=0.8)
            ax.scatter(points[:, 0], points[:, 1], c=np.arange(len(points)), cmap="viridis", s=4)
            ax.plot(points[0, 0], points[0, 1], "o", color="tab:green", ms=4)
            ax.set_aspect("equal")
            if unit_square:
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
            if col == 0:
                ax.set_ylabel(class_name, fontsize=11)
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, path)


def plot_acc_vs_length(models, bucket_edges, title, path):
    """Test accuracy per sequence-length bucket for several models.

    models: list of (label, lengths, correct) where correct is a list of 0/1.
    bucket_edges: [e0, e1, ..., ek]; bucket i holds lengths in [e_i, e_{i+1}) (the last bucket includes e_k).
    The tick labels show the number of test sequences n in each bucket; an empty bucket gets no bar.
    """
    n_buckets = len(bucket_edges) - 1
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    counts = []
    for m, (label, lengths, correct) in enumerate(models):
        lengths = np.array(lengths)
        correct = np.array(correct)
        accs = []
        counts = []
        for i in range(n_buckets):
            last = i == n_buckets - 1
            upper_ok = lengths <= bucket_edges[i + 1] if last else lengths < bucket_edges[i + 1]
            in_bucket = (lengths >= bucket_edges[i]) & upper_ok
            counts.append(int(in_bucket.sum()))
            accs.append(100 * correct[in_bucket].mean() if in_bucket.any() else np.nan)
        positions = np.arange(n_buckets) + (m - (len(models) - 1) / 2) * width
        bars = ax.bar(positions, accs, width, label=label)
        ax.bar_label(bars, labels=["" if np.isnan(a) else f"{a:.0f}" for a in accs], fontsize=7)
    bucket_names = []
    for i in range(n_buckets):
        closing = "]" if i == n_buckets - 1 else ")"
        bucket_names.append(f"[{bucket_edges[i]:.0f}, {bucket_edges[i + 1]:.0f}{closing}\nn = {counts[i]}")
    ax.set_xticks(range(n_buckets), bucket_names)
    ax.set_xlabel("Sequence length (time steps)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper center", ncol=len(models))
    save_fig(fig, path)
