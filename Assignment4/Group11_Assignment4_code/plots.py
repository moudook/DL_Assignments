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
    axes = np.atleast_2d(axes)

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
    axes = np.atleast_2d(axes)

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
    axes = np.atleast_2d(axes)
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