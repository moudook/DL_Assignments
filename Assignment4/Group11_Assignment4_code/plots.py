"""
plots.py — milestone plotting for Assignment-4.

PLOTTING POLICY (per constraints.md): milestone-driven, NOT exhaustive.
A4 runs ~60 training jobs. Plotting every curve wastes the time budget the
tracker exists to protect. So:

  - Plot at TASK boundaries, for the SELECTED / best configuration only.
  - Plot comparison BARS built from results tables (nearly free - no training).
  - Never defer a milestone plot to a final "make all plots" pass.
  - Losing configurations get TABLE rows, not figures.

Every figure is a milestone artefact for the report, so filenames are
descriptive and never overwrite each other: a multi-day run whose figures
collide is unrecoverable.

Every save is ATOMIC (write .tmp, then os.replace). A truncated PNG from an
interrupted save is an unreadable image behind a valid-looking filename - the
worst possible failure, because it looks fine in a directory listing.

Figures required by the assignment (non-optional, plotted per architecture):
  reconstruction_grid     Task-2d, Task-5c - one image per class per split
  maxact_grid             Task-6 - all three variants (plain AE + both denoisers)
Plus analytical figures built from results tables:
  dimension_bars          which bottleneck / PCA dimension is best
  loss_curves             selected runs only, with the tolerance line drawn
  confusion_matrix        best architecture per task
"""

import os

import matplotlib
matplotlib.use("Agg")   # headless: no display on a background runner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import CLASS_NAMES

# A4 requires deep visual analysis of every figure, which needs legible output.
# These sizes are chosen so a single figure fills a reasonable block of an A4
# report page without becoming unreadable when scaled.
FIGSIZE_STD = (7, 5)
FIGSIZE_WIDE = (11, 4.5)
DPI = 150

# Consistent palette across the report so series keep their identity between
# figures. "colorblind" keeps the 5-class confusion matrix readable for the most
# common forms of colour vision deficiency.
PALETTE = "colorblind"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def fig_root(outdir):
    """Root of every figure. All task folders live directly beneath it."""
    return os.path.join(outdir, "plots")


def task_dirs(outdir, task):
    """
    Numbered subfolder layout for one task, returned as a dict.

    Requirement: figures must be organised well enough to browse, because the
    assignment will have 100+ of them by the time every task, architecture,
    bottleneck, noise level and split has one. Numeric prefixes keep related
    figures visibly grouped in a directory listing, and grouping by
    CONCEPT first (loss, accuracy, confusion) means the same question is answered
    the same way across all tasks.

    Callers use the keys; nothing here writes to disk until ensure_dir runs.
    """
    root = fig_root(outdir)
    task = str(task).replace("task", "")
    layout = {
        "root":         f"{root}/task{task}",
        "training":     f"{root}/task{task}/01_training_curves",
        "accuracy":     f"{root}/task{task}/02_accuracy",
        "confusion":    f"{root}/task{task}/03_confusion_matrices",
        "recon":        f"{root}/task{task}/04_reconstructions",
        "recon_error":  f"{root}/task{task}/05_reconstruction_error",
        "weights":      f"{root}/task{task}/06_weights_and_activations",
        "represent":    f"{root}/task{task}/07_representations",
        "comparison":   f"{root}/task{task}/08_comparisons",
    }
    return {k: ensure_dir(v) for k, v in layout.items()}


def prepare_all_dirs(outdir, tasks):
    """
    Create the figure tree for every task plus the cross-task summary up front.

    Done once so the structure exists (and is therefore browsable) even if a run
    is interrupted before it produces its first figure. prune_empty_dirs runs at
    the end to remove subfolders that ended up unused.
    """
    for t in tasks:
        task_dirs(outdir, t)
    return ensure_dir(os.path.join(fig_root(outdir), "00_summary"))


def prune_empty_dirs(outdir):
    """
    Remove figure subfolders that ended up with no figures in them.

    The full numbered tree is created up front so it exists (and is browsable)
    from the very first moment, but a task that cannot populate a subfolder -
    Task-1 has no autoencoder reconstructions, Task-3 has no weight visualisation
    - would otherwise leave a permanent empty directory. Once the figure count is
    in the hundreds, empty folders are browsing noise.

    Task roots and 00_summary survive even when empty, so the top level still
    reads as task1..task6 plus summary.
    """
    root = fig_root(outdir)
    if not os.path.isdir(root):
        return []

    removed = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            continue
        depth = rel.count(os.sep)
        # Keep task roots (depth 0) and 00_summary; only prune deeper subfolders.
        if depth == 0:
            continue
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            removed.append(rel.replace(os.sep, "/"))
    return removed


def save_atomic(fig, path):
    """
    Save via a temp file then rename.

    os.replace is atomic, so a reader or the next run never sees a partial PNG.
    """
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp.png"
    fig.savefig(tmp, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)
    return path


def _axes_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# ── additional primitives (exhaustive figure set) ─────────────────────────

def accuracy_curve(history, title, save_path, ylabel="accuracy (%)",
                   train_key="train_acc", val_key="val_acc"):
    """
    Train and validation accuracy per epoch, on one axes.

    Distinct from loss_curve in what it makes visible: loss saturates long before
    accuracy does, so a loss plot can look converged while accuracy is still
    climbing. Plotting accuracy per run means the report never has to infer an
    accuracy trend from a loss curve.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    eps = [h["epoch"] for h in history]
    if history and train_key in history[0]:
        ax.plot(eps, [100 * h[train_key] for h in history], lw=1.2, label="train")
    if history and val_key in history[0]:
        ax.plot(eps, [100 * h[val_key] for h in history], lw=1.2, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 102)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_atomic(fig, save_path)


def loss_and_accuracy(history, title, save_path, tol=1e-4):
    """
    Two-panel run summary: loss on top, train/val accuracy below.

    One figure per run answers both "did it converge" and "did it actually learn".
    Exported for every run so no figure needs a re-run to produce later.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    # A resumed run can legitimately have no history if it did zero epochs and
    # the caller passed an empty list; indexing history[0] would then raise.
    if not history:
        ax1.text(0.5, 0.5, "no training history", ha="center", va="center")
        ax1.set_title(title)
        return save_atomic(fig, save_path)
    eps = [h["epoch"] for h in history]
    loss = [h.get("loss", float("nan")) for h in history]
    finite = [v for v in loss if np.isfinite(v)]
    # Log scale only when the dynamic range justifies it; linear is better when
    # loss barely moves, which is itself a finding worth seeing plainly.
    if finite and max(finite) / max(min(finite), 1e-12) > 20:
        ax1.set_yscale("log")
    ax1.plot(eps, loss, lw=1.3, label="training loss")
    ax1.axhline(tol, ls="--", lw=1, color="crimson", label=f"tolerance {tol:g}")
    ax1.set_ylabel("training loss")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    if "train_acc" in history[0]:
        ax2.plot(eps, [100 * h["train_acc"] for h in history], lw=1.2, label="train")
    if "val_acc" in history[0]:
        ax2.plot(eps, [100 * h["val_acc"] for h in history], lw=1.2, label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_atomic(fig, save_path)


def per_class_accuracy(per_class, title, save_path, ylim=(0, 100)):
    """
    Bar chart of per-class accuracy from a confusion matrix.

    Aggregate accuracy hides which digit is hardest, and the assignment asks for
    genuine inference about the errors - so this is emitted for every selected
    configuration, not just the best one.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    labels = [p["digit"] for p in per_class]
    accs = [100 * p["accuracy"] for p in per_class]
    bars = ax.bar(labels, accs, color="tab:blue")
    best, worst = int(np.argmax(accs)), int(np.argmin(accs))
    # Mark the extremes; comparing five bars by eye is error-prone.
    bars[best].set_color("tab:green")
    bars[worst].set_color("tab:red")
    for b, a, p in zip(bars, accs, per_class):
        ax.annotate(f"{a:.2f}%\n({p['errors']} err)",
                    xy=(b.get_x() + b.get_width() / 2, a), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_xlabel("digit")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return save_atomic(fig, save_path)


def accuracy_heatmap(rows, cols, values, title, save_path,
                     rowlabel="architecture", collabel="dimension"):
    """
    Architecture (rows) x dimension (cols) grid of one metric.

    These are the comparison tables in figure form and answer "which combination
    wins" in one glance. Colour scale is fixed to 0-100 so a given score means
    the same thing across every heatmap in the report; a per-figure auto-range
    would make the same value look good or bad depending on its neighbours.
    """
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(cols)),
                                    max(4, 0.8 * len(rows)) + 1))
    arr = np.asarray(values, dtype=float)
    sns.heatmap(arr, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=100,
                xticklabels=cols, yticklabels=rows, ax=ax,
                cbar_kws={"label": "score"})
    ax.set_xlabel(collabel)
    ax.set_ylabel(rowlabel)
    ax.set_title(title)
    fig.tight_layout()
    return save_atomic(fig, save_path)

# ── reconstruction grids (Task-2d, Task-5c) ────────────────────────────────

def reconstruction_grid(originals, reconstructions, labels, title,
                        save_path, ncols=None):
    """
    Two-row grid: originals on top, reconstructions below, one column per class.

    A4 requires "one image from each class, from the training, validation and
    test set... their reconstructed images for each of the architectures (along
    with original images)". Stacking original above reconstruction in the SAME
    figure is deliberate: the comparison the reader must make is per-column, and
    splitting originals and reconstructions into two figures would force a
    lookup across pages.

    labels: per-column labels (class digits here).
    """
    n = len(originals)
    ncols = ncols or n
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.5, nrows * 1.7 + 0.4))
    # Force a strict (nrows, ncols) view. np.atleast_2d only ADDS a leading
    # axis when one is missing, so with ncols == 1 the (2,) array becomes (1,2)
    # and indexing axes[1, i] is out of bounds. reshape always yields the 2-D
    # shape the loops below assume, for any width.
    axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)

    for i in range(n):
        axes[0, i].imshow(originals[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(reconstructions[i], cmap="gray", vmin=0, vmax=1)
        # shared colour scale across the whole grid: a per-image autoscale would
        # make a blurry reconstruction look as contrasty as a sharp original and
        # hide exactly the degradation this figure exists to show.
        _axes_off(axes[0, i])
        _axes_off(axes[1, i])
        if i < len(labels):
            axes[0, i].set_title(str(labels[i]), fontsize=11)

    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("reconstructed", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_atomic(fig, save_path)


# ── Task-6 maximally-activating grids ───────────────────────────────────────

def maxact_grid(images, weights, title, save_path, unit_labels=None):
    """
    Maximally-activating inputs and matching encoder weight vectors.

    Two stacked rows: the training image that fires each unit hardest, and the
    unit's encoder weight vector reshaped to 28x28. A4 Task-6 asks for the
    inputs that maximally activate each neuron and glosses it as "plot of weights
    from the input layer to the compressed layer" - showing both rows makes the
    comparison explicit instead of guessing which was intended.

    When weights are unavailable (multi-layer encoder) only the input row shows.
    """
    n = len(images)
    has_w = weights is not None and len(weights) > 0
    nrows = 2 if has_w else 1
    fig, axes = plt.subplots(nrows, n, figsize=(n * 1.05, nrows * 1.25 + 0.5))
    # reshape (not atleast_2d): with a single column atleast_2d turns a (rows,)
    # array into (1, rows), which then indexes wrongly. reshape always gives the
    # 2-D (nrows, ncols) view the loops below assume.
    axes = np.asarray(axes, dtype=object).reshape(nrows, -1)

    for i in range(n):
        axes[0, i].imshow(np.clip(images[i].reshape(28, 28), 0, 1),
                          cmap="gray", vmin=0, vmax=1)
        _axes_off(axes[0, i])
        if unit_labels is not None and i < len(unit_labels):
            axes[0, i].set_title(str(unit_labels[i]), fontsize=6)
        if has_w:
            # Weights are signed, so use a diverging map centred on zero. A
            # grayscale image map would show half the weights as pure black and
            # destroy the structure Task-6 is asking us to compare.
            #
            # Scale to the per-figure maximum magnitude, NOT a hardcoded +-1.
            # Encoder weights are small (Xavier init gives roughly +-0.03 on a
            # 784-input layer), so a fixed +-1 range renders every weight as
            # near-white and the figure shows nothing. Scaling by the observed
            # max keeps the structure visible at any weight magnitude.
            w = np.asarray([np.asarray(x) for x in weights])
            wmax = float(np.abs(w).max())
            wmax = wmax if wmax > 0 else 1.0
            wi = np.clip(weights[i].reshape(28, 28), -wmax, wmax)
            axes[1, i].imshow(wi, cmap="RdBu_r", vmin=-wmax, vmax=wmax)
            _axes_off(axes[1, i])

    axes[0, 0].set_ylabel("max-act input", fontsize=8)
    if has_w:
        axes[1, 0].set_ylabel("encoder weight", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return save_atomic(fig, save_path)


# ── loss / convergence curves (selected runs only) ──────────────────────────

def loss_curve(history, title, save_path, tol=1e-4, ylabel="training loss",
               val_key="val_acc", second_ylabel=None):
    """
    Loss vs epoch with the early-stopping tolerance drawn in.

    The tolerance line is included deliberately: A3's failure mode was a model
    satisfying |dL| < 1e-4 while sitting at chance accuracy. Drawing the
    threshold makes it visible whether a run converged on merit or merely
    stopped moving, which is exactly the judgement the report must make.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    if not history:
        ax.text(0.5, 0.5, "no history", ha="center", va="center")
        return save_atomic(fig, save_path)

    eps = [h["epoch"] for h in history]
    loss = [h.get("loss", np.nan) for h in history]

    # Log scale only when the range is wide; linear reads better when it is not.
    finite = [v for v in loss if np.isfinite(v)]
    if finite and max(finite) / max(min(finite), 1e-12) > 50:
        ax.set_yscale("log")

    ax.plot(eps, loss, label="training loss", lw=1.2)
    ax.axhline(tol, ls="--", lw=1, color="crimson",
               label=f"stopping tolerance {tol:g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if second_ylabel and val_key and val_key in history[0]:
        ax2 = ax.twinx()
        ax2.plot(eps, [h[val_key] for h in history], color="tab:green",
                 lw=1.0, alpha=0.7, label=second_ylabel)
        ax2.set_ylabel(second_ylabel)

    fig.tight_layout()
    return save_atomic(fig, save_path)


def superimposed_curves(histories, labels, title, save_path, ylabel="loss",
                        log=True):
    """
    Several runs on one axes. Milestone use only: the best run per task, or the
    runs the report will actually discuss - not every run that exists.

    A3's figures superimposed all 7 optimizers this way; A4 reuses the FORM but
    restricts WHICH curves appear, per the milestone plotting policy.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    for h, lab in zip(histories, labels):
        if not h:
            continue
        eps = [e["epoch"] for e in h]
        vals = [e.get("loss", np.nan) for e in h]
        if log:
            ax.plot(eps, vals, lw=1.2, label=lab)
        else:
            ax.plot(eps, vals, lw=1.2, label=lab)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if labels:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return save_atomic(fig, save_path)


# ── comparison bars (built from result tables, nearly free) ─────────────────

def dimension_bars(values_by_series, x_labels, title, save_path,
                   ylabel="accuracy (%)", ref=None, ref_label=None):
    """
    Grouped bars comparing methods across dimensions.

    This is the highest-value figure per task: it is what answers A4's "observe
    the best reduced dimension", and it is built from a results table rather than
    from training, so plotting every one of them costs essentially nothing.

    ref/ref_label: optional horizontal reference line, used to show the A3
    baseline (98.76% test) or the raw-784 control, so the compressed-representation
    results are read against the thing they are compared to in the text.
    """
    n = len(x_labels)
    width = 0.8 / max(1, len(values_by_series))
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(n)

    for i, (name, vals) in enumerate(values_by_series.items()):
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    if ref is not None:
        ax.axhline(ref, ls="--", lw=1.2, color="black",
                   label=ref_label or "reference")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return save_atomic(fig, save_path)


def recon_error_bars(recon_by_arch, title, save_path):
    """
    Reconstruction error per split, grouped by architecture.

    A4 Task-2c and Task-5b ask for average reconstruction error on train, val and
    test. Showing all three together exposes the generalisation gap directly:
    a large train/test spread would mean overfitting the encoder.
    """
    splits = ["train", "val", "test"]
    n = len(recon_by_arch)
    width = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(splits))

    for i, (name, vals) in enumerate(recon_by_arch.items()):
        ax.bar(x + i * width - 0.4 + width / 2, [vals[s] for s in splits],
               width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean reconstruction error (MSE)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_atomic(fig, save_path)


# ── confusion matrix ────────────────────────────────────────────────────────

def confusion_matrix(cm, title, save_path, normalize=False):
    """
    Confusion matrix in fixed {0,4,5,6,7} order.

    The class order is forced rather than left to whatever the labels happen to
    be, so every confusion matrix in the report has rows and columns aligned and
    can be compared by eye against the others.
    """
    arr = np.asarray(cm, dtype=float)
    if normalize:
        arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(arr, annot=True, fmt=".0f" if not normalize else ".2f",
                cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar=not normalize)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    return save_atomic(fig, save_path)


def eigen_spectrum(eigenvalues, title, save_path, marks=(32, 64, 128, 256)):
    """
    Scree plot: eigenvalue magnitude vs component index, log-scaled.

    The variance-retained curve answers "how much signal do the first k keep".
    This answers a related but different question: which components actually
    matter. The spectrum decays steeply, so a handful of components carry most of
    the variance and the rest are near-zero - which explains directly why k=32
    already classifies well.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ev = np.asarray(eigenvalues, dtype=float)
    ev = ev[ev > 0]
    ax.plot(np.arange(1, len(ev) + 1), ev, lw=1.4)
    ax.set_yscale("log")
    for m in marks:
        if m <= len(ev):
            ax.axvline(m, ls=":", lw=1, color="grey")
            ax.annotate(f"k={m}", xy=(m, ev[m - 1]), xytext=(5, 0),
                        textcoords="offset points", fontsize=8, va="center")
    ax.set_xlabel("component index (descending eigenvalue)")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return save_atomic(fig, save_path)


def denoise_triptych(clean, noisy, reconstructed, title, save_path):
    """
    Clean target -> corrupted input -> reconstruction, stacked per column.

    The central evidence figure for Task-5: it shows what the network actually
    saw, what it was asked to produce, and what it produced. Without the middle
    row the grid is ambiguous about the input, which is the whole point of a
    denoising autoencoder.
    """
    n = len(clean)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.4, 3 * 1.7))
    # reshape (not atleast_2d): see maxact_grid. Guarantees (nrows, ncols).
    axes = np.asarray(axes, dtype=object).reshape(3, -1)

    for i in range(n):
        for r, (arr, lab) in enumerate(((clean, "clean target"),
                                        (noisy, "corrupted input"),
                                        (reconstructed, "reconstruction"))):
            im = axes[r, i]
            im.imshow(np.clip(arr[i].reshape(28, 28), 0, 1),
                      cmap="gray", vmin=0, vmax=1)
            _axes_off(im)
            if i == 0:
                im.set_ylabel(lab, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return save_atomic(fig, save_path)


def activation_histogram(values, title, save_path, bins=60, xlabel=None):
    """
    Distribution of a per-unit quantity (activation or weight).

    Summary companion to the per-unit bar charts: where those show individual
    units, this shows the overall spread, including how many units are near-silent
    - the actual evidence for whether the bottleneck is used efficiently or left
    partly idle.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    v = np.asarray(values, dtype=float).ravel()
    ax.hist(v, bins=bins, color="tab:blue", alpha=0.85)
    ax.axvline(0, ls="--", lw=1, color="black")
    ax.set_xlabel(xlabel or "value")
    ax.set_ylabel("number of bottleneck units")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return save_atomic(fig, save_path)


def series_curve(series, title, save_path, xlabel, ylabel,
                 ref=None, ref_label=None, margins=None):
    """
    Plot one or more named x -> y series as lines with markers.

    Distinct from superimposed_curves, which takes per-epoch training histories
    (a list of dicts with "epoch" and "loss"). This takes explicit coordinate
    mappings, which is what the cross-task and comparison figures need: "test
    accuracy by representation size" is a metric against a hyperparameter, not a
    metric over training epochs, and forcing it into the epoch-history shape
    fails on the key lookup.

    margins: optional per-x offset applied to the LAST series, so a second
    method's points are readable when both sit on top of each other.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    for name, xy in series.items():
        xs = sorted(xy)
        ys = [xy[x] for x in xs]
        ax.plot(xs, ys, lw=1.6, marker="o", ms=4, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if ref is not None:
        ax.axhline(ref, ls="--", lw=1.2, color="black",
                   label=ref_label or "reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_atomic(fig, save_path)


def generalisation_gap(gaps, title, save_path, ylabel="train - test (pp)"):
    """
    Per-configuration generalisation gap: train accuracy minus test accuracy.

    The assignment asks for observations about generalisation, and this is the
    figure that makes the claim falsifiable. A near-zero gap supports "no
    overfitting"; a gap that grows with capacity shows the opposite. Emitted for
    every task's selected configuration.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    names = list(gaps)
    vals = [gaps[n] for n in names]
    xs = np.arange(len(names))
    colors = ["tab:green" if abs(v) < 2 else ("tab:orange" if abs(v) < 5
                                              else "tab:red") for v in vals]
    ax.bar(xs, vals, width=0.6, color=colors)
    ax.axhline(0, color="black", lw=1)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:+.2f}", xy=(x, v),
                    xytext=(0, 3 if v >= 0 else -13),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return save_atomic(fig, save_path)


def comparison_bars(series, title, save_path, ylabel="test accuracy (%)",
                    labels=None):
    """
    One labelled bar per entry; the best bar is highlighted.

    Used for every cross-configuration comparison: PCA against each AE kind and
    the denoiser, all against the A3 baseline.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    names = list(series)
    vals = [100 * series[n] for n in names]
    xs = np.arange(len(names))
    bars = ax.bar(xs, vals, width=0.6, color="tab:blue")
    bars[int(np.argmax(vals))].set_color("tab:green")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels or names, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return save_atomic(fig, save_path)


# ── PCA-specific ────────────────────────────────────────────────────────────

def pca_variance_curve(cum_retained, title, save_path, marks=(32, 64, 128, 256)):
    """
    Cumulative variance retained vs retained dimension.

    Gives the report its explanation for WHY accuracy rises with dimension
    rather than merely asserting it: the curve shows how much information each
    component preserves, and the markers show where the four tested dimensions
    sit on it.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    xs = np.arange(1, len(cum_retained) + 1)
    ax.plot(xs, 100 * np.asarray(cum_retained), lw=1.5, label="cumulative")
    for m in marks:
        if m <= len(cum_retained):
            ax.axvline(m, ls=":", lw=1, color="grey")
            ax.annotate(f"k={m}\n{100 * cum_retained[m - 1]:.1f}%",
                        xy=(m, 100 * cum_retained[m - 1]),
                        xytext=(4, -22), textcoords="offset points", fontsize=8)
    ax.set_xlabel("retained dimension k")
    ax.set_ylabel("variance retained (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_atomic(fig, save_path)


def pca_reconstruction_grid(originals, k_recons, title, save_path):
    """
    Original image beside its PCA reconstruction at each tested dimension.

    Makes the information loss visible: as k grows the reconstruction sharpens
    toward the original. This is the visual evidence behind the variance-retained
    numbers, and it is the PCA analogue of the autoencoder reconstruction grids.
    """
    # Axes count is driven by the number of RECONSTRUCTIONS, not the number of
    # originals: the figure is one original plus one panel per k, and those are
    # independent counts. Sizing on len(originals) breaks whenever a caller
    # passes a single original against several k values, which is exactly how
    # Task-1 calls it.
    n_rec = len(k_recons)
    fig, axes = plt.subplots(1, n_rec + 1, figsize=((n_rec + 1) * 1.5, 1.9))
    axes = np.atleast_1d(axes)
    axes[0].imshow(originals[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=9)
    _axes_off(axes[0])
    for i, (k, rec) in enumerate(k_recons.items()):
        ax = axes[i + 1]
        ax.imshow(rec, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"k={k}", fontsize=9)
        _axes_off(ax)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return save_atomic(fig, save_path)


def weight_image_grid(weights_2d, title, save_path, ncols=8):
    """
    Plain grid of encoder weight vectors as 28x28 images.

    Complement to maxact_grid: this shows ALL units at once for architecture-level
    inspection, whereas maxact_grid pairs each unit with its best input.
    Diverging colormap centred on zero, because the linear bottleneck makes these
    weights signed.
    """
    n = len(weights_2d)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 0.85, nrows * 0.9 + 0.4))
    # reshape (not atleast_2d): guarantees (nrows, ncols) for any unit count,
    # including a single row that atleast_2d would leave 1-D.
    axes = np.asarray(axes, dtype=object).reshape(nrows, -1)
    flat = axes.flatten()
    # Per-figure symmetric scale from the observed maximum, so the structure is
    # visible regardless of how small the trained weights are.
    wmax = float(np.abs(np.asarray([np.asarray(w) for w in weights_2d])).max())
    wmax = wmax if wmax > 0 else 1.0
    for i in range(n):
        im = axes.flat[i]
        w = np.clip(np.asarray(weights_2d[i]).reshape(28, 28), -wmax, wmax)
        im.imshow(w, cmap="RdBu_r", vmin=-wmax, vmax=wmax)
        _axes_off(im)
    for j in range(n, len(flat)):
        flat[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return save_atomic(fig, save_path)