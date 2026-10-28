"""
plots.py - every figure is drawn by a helper here and saved by save_fig(),
which lints it before writing: matplotlib warnings are errors, text must lie
inside the canvas, text must not overlap other text or another axes, and the
figure must not be blank.

Colormaps follow the lecture notes (slides 107-110, 125, 141-144) and are the
same everywhere:
  filters (weights):  gray, symmetric about 0 (mid-grey = 0, white = most positive)
  feature maps:       viridis (post-ReLU from 0; the un-activated A1 map min..max)
  gradients:          on a mid-grey background (part_b2_vgg.to_display)
  MNIST images: gray;  Grad-CAM overlay: jet, alpha 0.4
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from protocol import EXPLODE  # noqa: E402

DPI = 200
SIGNED = "gray"
POSITIVE = "viridis"
BOX_COLOURS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
               "#911eb4", "#42d4f4", "#f032e6"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "image.interpolation": "nearest",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class FigureLintError(RuntimeError):
    pass


# ---------------------------------------------------------------- lint

def _texts(fig):
    """Text artists that are actually drawn.

    Axes keep tick objects for positions outside the view limits; they stay
    "visible" but are never drawn, so tick labels are taken only from the ticks
    the axis will draw (Axis._update_ticks, what Axis.draw itself uses).
    """
    tick_texts, drawn = set(), []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_texts.update((id(tick.label1), id(tick.label2)))
            for tick in axis._update_ticks():
                drawn += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
    out = [t for t in fig.findobj(Text) if id(t) not in tick_texts] + drawn
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def _overlap(a, b, tol):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol and
            min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def lint(fig, tol=1.0):
    """List of problems; empty means the figure is clean."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    problems = []
    texts = _texts(fig)
    boxes = [t.get_window_extent(r) for t in texts]
    axes = [ax for ax in fig.axes if ax.get_visible()]
    ax_boxes = [ax.get_window_extent(r) for ax in axes]

    for t, b in zip(texts, boxes):
        if b.x0 < -tol or b.y0 < -tol or b.x1 > W + tol or b.y1 > H + tol:
            problems.append("text outside canvas: %r" % t.get_text())

    # text vs text: in-cell annotations of one axes are compared too, which
    # catches value labels too big for their cells.
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if _overlap(boxes[i], boxes[j], tol):
                problems.append("overlapping text: %r / %r"
                                % (texts[i].get_text(), texts[j].get_text()))

    # text vs a foreign axes (titles running into the panel above, tick labels
    # into the neighbour). Text inside its own axes is fine.
    for t, b in zip(texts, boxes):
        own = t.axes
        for ax, ab in zip(axes, ax_boxes):
            if ax is own:
                continue
            if _overlap(b, ab, tol):
                problems.append("text %r overlaps another axes" % t.get_text())
                break

    img = np.asarray(r.buffer_rgba())
    if img[..., :3].std() < 1e-3:
        problems.append("figure is blank")
    return problems


def save_fig(fig, path, pdf=False):
    """Lint, then write PNG (and optionally PDF). Raises on any problem."""
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        problems = lint(fig)
        if problems:
            plt.close(fig)
            raise FigureLintError("%s:\n  %s" % (path.name, "\n  ".join(problems)))
        fig.savefig(path, dpi=DPI)
        if pdf:
            fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# ---------------------------------------------------------------- primitives

def new_fig(nrows, ncols, panel=1.6, extra_w=0.0, extra_h=0.4, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel + extra_w,
                                                    nrows * panel + extra_h),
                             squeeze=False, layout="constrained", **kw)
    return fig, axes


def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def show_image(ax, img, title=None):
    """uint8/float image, greyscale (H,W) or RGB (H,W,3)."""
    img = np.asarray(img)
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255 if img.dtype == np.uint8 else 1)
    else:
        ax.imshow(img)
    clean_ax(ax)
    if title:
        ax.set_title(title)


def show_signed(ax, m, lim=None, title=None):
    lim = lim if lim is not None else float(np.abs(m).max()) or 1.0
    im = ax.imshow(m, cmap=SIGNED, vmin=-lim, vmax=lim)
    clean_ax(ax)
    if title:
        ax.set_title(title)
    return im


def show_positive(ax, m, vmax=None, title=None, vmin=0.0):
    vmax = vmax if vmax is not None else float(m.max()) or 1.0
    im = ax.imshow(m, cmap=POSITIVE, vmin=vmin, vmax=vmax)
    clean_ax(ax)
    if title:
        ax.set_title(title)
    return im


def annotate_cells(ax, m, fmt="{:.2f}", lim=None, fontsize=None):
    """Write each value in its cell; text colour flips on dark cells (the
    symmetric gray scale: -lim black, 0 mid-grey, +lim white)."""
    lim = lim if lim is not None else float(np.abs(m).max()) or 1.0
    for (i, j), v in np.ndenumerate(m):
        dark = (v / lim + 1) / 2 < 0.45
        ax.text(j, i, fmt.format(v), ha="center", va="center",
                fontsize=fontsize or 6, color="white" if dark else "black")


def draw_box(ax, box, colour, lw=1.5, label=None):
    """Inclusive pixel box (r0, r1, c0, c1), drawn on pixel edges."""
    r0, r1, c0, c1 = box
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1,
                           fill=False, edgecolor=colour, linewidth=lw, label=label))


def confusion_matrix_plot(cm, classes, title, path):
    """Counts and row-% per cell; accuracy in the title."""
    cm = np.asarray(cm)
    n = len(classes)
    acc = np.trace(cm) / max(cm.sum(), 1)
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    # Width has slack on purpose: with an aspect-locked heatmap plus a labelled
    # colorbar, constrained layout pushed the y label off the canvas at n + 1.8.
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.6, 1.0 * n + 1.2), layout="constrained")
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, "%d\n%.0f%%" % (v, 100 * row[i, j]), ha="center", va="center",
                fontsize=8, color="white" if row[i, j] > 0.6 else "black")
    ax.set_xticks(range(n), classes, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("%s\naccuracy %.2f%% (%d/%d)" % (title, 100 * acc, np.trace(cm), cm.sum()))
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    return save_fig(fig, path)


def image_maps_grid(images, row_labels, columns, title, path, panel=1.35, note=None):
    """Rows = input images; first column the image, then one column per map.

    `columns` is a list of (column title, maps (N, h, w)). Post-ReLU maps use
    viridis from 0 with the scale SHARED down a column (one filter across all
    images), so the same colour means the same activation within a filter.
    """
    n, ncol = len(images), 1 + len(columns)
    fig, axes = new_fig(n, ncol, panel=panel, extra_h=0.6)
    for r in range(n):
        show_image(axes[r, 0], images[r])
        axes[r, 0].set_ylabel(row_labels[r])
    axes[0, 0].set_title("input")
    for c, (ctitle, maps) in enumerate(columns, 1):
        maps = np.asarray(maps)
        vmax = float(maps.max())
        for r in range(n):
            show_positive(axes[r, c], maps[r], vmax=vmax if vmax > 0 else 1.0)
            if not maps[r].any():
                axes[r, c].text(0.5, 0.5, "all 0", transform=axes[r, c].transAxes,
                                ha="center", va="center", color="white", fontsize=7)
        axes[0, c].set_title(ctitle)
        axes[n - 1, c].set_xlabel("max %.2f" % vmax, fontsize=7)
    fig.suptitle(title + ("\n" + note if note else ""))
    return save_fig(fig, path)


def filter_row(filters, titles, title, path, annotate=True, lim=None, panel=1.9):
    """2-D filters side by side, gray scale symmetric about 0, values printed in the cells."""
    filters = [np.asarray(f) for f in filters]
    lim = lim if lim is not None else max(float(np.abs(f).max()) for f in filters)
    fig, axes = new_fig(1, len(filters), panel=panel, extra_h=0.7)
    for ax, f, t in zip(axes[0], filters, titles):
        im = show_signed(ax, f, lim=lim, title=t)
        if annotate:
            annotate_cells(ax, f, lim=lim)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="weight")
    fig.suptitle(title)
    return save_fig(fig, path)


def filter_slices(rows, row_labels, title, path, lim=None, panel=1.0, annotate=False):
    """Multi-channel filters: one row per filter, one column per input channel
    (the notes draw VGG's 3x3x3 filters as per-channel slices)."""
    rows = [np.asarray(r) for r in rows]            # each (C, k, k)
    C = rows[0].shape[0]
    lim = lim if lim is not None else max(float(np.abs(r).max()) for r in rows)
    fig, axes = new_fig(len(rows), C, panel=panel, extra_w=1.1, extra_h=0.6)
    for i, (r, lab) in enumerate(zip(rows, row_labels)):
        for c in range(C):
            im = show_signed(axes[i, c], r[c], lim=lim)
            if annotate:
                annotate_cells(axes[i, c], r[c], lim=lim, fontsize=4.5)
            if i == 0:
                axes[i, c].set_title("in-ch %d" % (c + 1))
        axes[i, 0].set_ylabel(lab)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="weight")
    fig.suptitle(title)
    return save_fig(fig, path)


def curves_plot(hist, best_epoch, title, path):
    """Loss and accuracy vs epoch; the restored best epoch is marked."""
    ep = hist["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for ax, key, label in ((axes[0], "loss", "cross-entropy loss"),
                           (axes[1], "acc", "accuracy")):
        tr = [(e, v) for e, v in zip(ep, hist["train_" + key]) if v is not None]
        ax.plot(*zip(*tr), label="train (running, per epoch)", color="#4363d8")
        ax.plot(ep, hist["val_" + key], label="validation", color="#e6194b")
        ax.axvline(best_epoch, color="0.4", ls="--", lw=1,
                   label="restored best epoch (%d)" % best_epoch)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_yscale("log")
    axes[1].legend(loc="lower right")
    fig.suptitle(title)
    return save_fig(fig, path)


def grid_plot(record, title, path):
    """Validation loss vs epoch for every learning rate tried."""
    # A run that blows up (val loss 1e14 seen at lr 0.1) would flatten every
    # other curve, so the y range covers the runs that stayed in range and the
    # exploded ones are drawn off-scale and labelled as such.
    fig, ax = plt.subplots(figsize=(6.8, 3.6), layout="constrained")
    cmap = plt.get_cmap("viridis")
    lrs = sorted(record["histories"], key=float)
    curves = {}
    for lr in lrs:
        h = record["histories"][lr]
        curves[lr] = np.array([v if v is not None and np.isfinite(v) else np.nan
                               for v in h["val_loss"]], dtype=float)
    sane = [lr for lr in lrs if np.nanmax(curves[lr]) <= EXPLODE * curves[lr][0]]
    for k, lr in enumerate(lrs):
        h = record["histories"][lr]
        tags = []
        if float(lr) == record["winner_lr"]:
            tags.append("selected")
        if lr not in sane:
            tags.append("exploded, off scale")
        ax.plot(h["epoch"], curves[lr], color=cmap(k / max(len(lrs) - 1, 1)),
                lw=2.2 if "selected" in tags else 1.2,
                label="lr %s%s" % (lr, " (%s)" % ", ".join(tags) if tags else ""))
    ax.set_yscale("log")
    if sane:
        lo = min(np.nanmin(curves[lr]) for lr in sane)
        hi = max(np.nanmax(curves[lr]) for lr in sane)
        ax.set_ylim(lo / 1.3, hi * 1.3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(title)
    return save_fig(fig, path)


def contact_sheet(paths, out_path, cols=4):
    """All figures on one PDF page grid for a final visual review."""
    paths = [Path(p) for p in paths]
    rows = max(1, int(np.ceil(len(paths) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        clean_ax(ax)
    for ax, p in zip(axes.flat, paths):
        ax.imshow(plt.imread(p))
        ax.set_title(p.name, fontsize=6)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
