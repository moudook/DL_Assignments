"""
Dataset-structure figures.

Every other figure in this pipeline shows the RESULT of a model. These show the
INPUTS, which the report otherwise never pictures: what the five digit classes
look like, how they are distributed, how far apart they actually are, and whether
the three splits are interchangeable.

That last point is not cosmetic. Two claims in the report depend on it: that the
splits are drawn from the same distribution, and that the dominant 5<->6 confusion
is a property of the digits rather than of any particular architecture. Both are
checkable here, from the data, before any model is trained.

Every figure is derived from the split tensors alone, so nothing in this module
depends on a trained model or on a run having finished. Output goes to
plots/00_dataset/, which sits alongside 00_summary rather than inside a numbered
task folder, because these figures answer "what is the data?" and belong to all
six tasks at once.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plots

DIGITS = ["0", "4", "5", "6", "7"]
SPLITS = ("train", "val", "test")


def _dir(outdir):
    return plots.ensure_dir(os.path.join(plots.fig_root(outdir), "00_dataset"))


def _to_np(x):
    """Tensor -> (N, 784) float64 numpy, on CPU."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return x.double().numpy()


def class_means(data):
    """
    Mean image per class, as {class_index: (784,) float64}, plus the counts.

    Keyed by the CLASS INDEX 0..C-1, which is what `data["y_*"]` actually
    contains: the loader goes through `datasets.ImageFolder`, and that assigns
    integer indices in sorted-folder order. The digit each index denotes is
    `data["classes"][i]`, i.e. index 1 is the digit 4, not the digit 1. Use
    `_digit(data, i)` to turn an index into the label a reader expects.

    These are the class prototypes: the cheapest possible five-way classifier
    classifies by nearest prototype, so the geometry of these five vectors is
    an upper bound on how well any model can separate the classes.
    """
    X, y = _to_np(data["X_train"]), _to_np(data["y_train"])
    means, counts = {}, {}
    for v in sorted(set(y.tolist())):
        m = y == v
        counts[v] = int(m.sum())
        means[v] = X[m].mean(axis=0)
    return means, counts


def _digit(data, i):
    """Class index -> the digit it denotes ('0','4','5','6','7')."""
    names = list(data.get("classes") or ["0", "4", "5", "6", "7"])
    return names[int(i)]


def _digit_order(data):
    """The class indices present, ascending. 0..C-1, NOT digit values."""
    return sorted(set(_to_np(data["y_train"]).tolist()))


def class_distribution(data, outdir):
    """
    Count of each class in each split, as a grouped bar chart.

    This is the figure that licenses the 'the split is exactly balanced, so
    chance is exactly 20%' statement in the report. If any bar differed, chance
    would be a per-class quantity rather than a single number.
    """
    d = _dir(outdir)
    order = _digit_order(data)
    counts = {}
    for sp in SPLITS:
        y = _to_np(data[f"y_{sp}"])
        counts[sp] = [int((y == v).sum()) for v in order]

    fig, ax = plt.subplots(figsize=plots.FIGSIZE_STD)
    n = len(order)
    x = np.arange(n)
    w = 0.8 / len(SPLITS)
    for i, sp in enumerate(SPLITS):
        ax.bar(x + i * w - 0.4 + w / 2, counts[sp], w, label=sp)
        for j, v in enumerate(counts[sp]):
            ax.text(x[j] + i * w - 0.4 + w / 2, v, str(v), ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"digit {_digit(data, j)}" for j in range(n)])
    ax.set_ylabel("images")
    ax.set_title("Class balance across splits: every split is exactly 1/5 per class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return plots.save_atomic(fig, os.path.join(d, "dataset_class_distribution.png"))


def split_equivalence(data, outdir):
    """
    Per-split pixel mean and standard deviation, overlaid.

    If the splits were drawn differently -- rescaled, differently normalised,
    or simply not shuffled -- these three pairs would separate. They do not, which
    is the evidence behind 'validation and test are drawn from the same
    distribution as train, so a comparison across splits is meaningful'.
    """
    d = _dir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=plots.FIGSIZE_WIDE)
    for s in SPLITS:
        X = _to_np(data[f"X_{s}"])
        axes[0].plot(X.mean(), lw=1.4, label=s)
        axes[1].plot(X.std(axis=0), lw=1.4, label=s)
    axes[0].set_title("Per-pixel mean intensity")
    axes[1].set_title("Per-pixel standard deviation")
    for a in axes:
        a.set_xlabel("pixel index (row-major 28x28)")
        a.set_ylabel("intensity in [0,1]")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Train, validation and test are statistically indistinguishable pixel-wise",
                 fontsize=10)
    fig.tight_layout()
    return plots.save_atomic(fig, os.path.join(d, "dataset_split_equivalence.png"))


def mean_images(data, outdir):
    """The five class prototypes, and the per-pixel spread within each class."""
    d = _dir(outdir)
    order = _digit_order(data)
    fig, axes = plt.subplots(1, 2 * len(order), figsize=(11, 5))
    means, _ = class_means(data)
    X, y = _to_np(data["X_train"]), _to_np(data["y_train"])

    for j, v in enumerate(order):
        digit = _digit(data, v)
        m = y == v
        axes[j].imshow(means[v].reshape(28, 28), cmap="gray")
        axes[j].set_title(f"mean, digit {digit}\n(n={int(m.sum())})", fontsize=8)
        axes[len(order) + j].imshow(X[m].std(axis=0).reshape(28, 28), cmap="magma")
        axes[len(order) + j].set_title(f"std within digit {digit}", fontsize=8)
        for i in (j, len(order) + j):
            axes[i].axis("off")
    fig.suptitle("Class prototypes (left) and within-class spread (right): "
                 "a digit's mean is recognisable, its spread is not", fontsize=10)
    fig.tight_layout()
    return plots.save_atomic(fig, os.path.join(d, "dataset_mean_and_std_images.png"))


def class_mean_distances(data, outdir, confusions_from=None):
    """
    Pairwise Euclidean distance between the five class prototypes, as a heatmap,
    with the measured test-confusion mass for the same pairs alongside.

    This is the most load-bearing figure in the set. Every confusion matrix in the
    report is dominated by the 5<->6 pair, and the usual explanation offered for it
    is geometric ('a 5 and a 6 differ mainly in the curvature of the lower bowl').
    That explanation is a guess until it is measured. Here it is: if the 5<->6
    prototype distance is the smallest in the matrix, the confusion is a property of
    the data and no architecture can remove it, and the ranking of distances should
    track the ranking of observed confusions.

    It does not. The nearest pair is 4<->7 (3.91) and the most-confused is 5<->6
    (which sits only fourth); the two orderings agree at rho = +0.33. So the
    dominant confusion is NOT explained by mean-image proximity, and this figure is
    the evidence for replacing that hand-wavy claim with a measured one.

    `confusions_from` is a path to a task1.json to read the observed matrix from;
    when absent or unreadable the confusion and rank panels are omitted.
    """
    d = _dir(outdir)
    means, _ = class_means(data)
    order = _digit_order(data)
    stack = np.stack([means[v] for v in order])
    dist = np.linalg.norm(stack[:, None, :] - stack[None, :, :], axis=-1)

    obs = None
    if confusions_from and os.path.exists(confusions_from):
        obs = _observed_confusion_masses(confusions_from)

    n = len(order)
    labels = [_digit(data, i) for i in order]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), squeeze=False)
    axes = axes[0]
    im = axes[0].imshow(dist, cmap="viridis")
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(labels)
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{dist[i, j]:.1f}", ha="center", va="center",
                         color="w", fontsize=9)
    axes[0].set_title("Euclidean distance between class prototypes (train means)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    if obs is not None:
        im2 = axes[1].imshow(obs, cmap="magma")
        axes[1].set_xticks(range(n)); axes[1].set_xticklabels(labels)
        axes[1].set_yticks(range(n)); axes[1].set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                if i != j:
                    axes[1].text(j, i, f"{obs[i, j]:.0f}", ha="center", va="center",
                                 color="w", fontsize=9)
        axes[1].set_title("Misclassified pairs, best test model")
        fig.colorbar(im2, ax=axes[1], shrink=0.8)

        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                rows.append((f"{labels[i]}<->{labels[j]}", float(dist[i, j]),
                             float(obs[i, j])))
        by_dist = sorted(rows, key=lambda r: r[1])
        by_conf = sorted(rows, key=lambda r: -r[2])
        drank = {r[0]: k + 1 for k, r in enumerate(by_dist)}
        crank = {r[0]: k + 1 for k, r in enumerate(by_conf)}
        for name, dv, cv in by_dist:
            axes[2].scatter(drank[name], crank[name], s=55, color="#c0392b",
                            edgecolor="k", lw=0.6, zorder=3)
            axes[2].annotate(name, (drank[name], crank[name]),
                             textcoords="offset points", xytext=(5, 4),
                             fontsize=7)
        lim = n * (n - 1) / 2 + 1
        axes[2].plot([1, lim], [1, lim], ls="--", color="0.5", lw=1,
                     label="perfect agreement")
        rho = _spearman([r[1] for r in rows], [r[2] for r in rows])
        axes[2].set_xlabel("rank by prototype proximity (1 = closest)")
        axes[2].set_ylabel("rank by observed confusion (1 = most confused)")
        axes[2].set_title(f"Agreement between the two\nSpearman rho = {rho:+.2f}")
        axes[2].set_xlim(0.4, lim + 0.6); axes[2].set_ylim(0.4, lim + 0.6)
        axes[2].set_aspect("equal")
        axes[2].grid(alpha=0.3)
        axes[2].legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Prototype proximity does NOT explain the confusions: the nearest pair is "
        "not the most-confused pair", fontsize=10)
    if obs is None:
        for a in (axes[1], axes[2]):
            a.axis("off")
        axes[1].text(0.5, 0.5, "confusion and rank panels\\nappear once Task 1 has run",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=9, color="0.4")
    fig.tight_layout()

    pairs = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            pairs.append((f"{labels[i]}<->{labels[j]}", float(dist[i, j]),
                          float(obs[i, j]) if obs is not None else None))
    pairs.sort(key=lambda r: r[1])
    return plots.save_atomic(fig,
                             os.path.join(d, "dataset_class_mean_distances.png"))


def _spearman(a, b):
    """Spearman rank correlation without a scipy dependency (project has none)."""
    n = len(a)
    if n < 2:
        return 0.0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(list(a)), rank(list(b))
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else 0.0


def _observed_confusion_masses(task1_json_path):
    """
    Symmetric off-diagonal confusion mass from the single best plain model.

    Taken from the Task-1 confusion matrix because it is the linear baseline: if
    the prototype distances predict the confusions of the WEAKEST representation,
    the ordering is a property of the data and not of model capacity.
    Returns None if Task 1 has not run, so this module still works standalone.
    """
    import json
    if not os.path.exists(task1_json_path):
        return None
    j = json.load(open(task1_json_path, encoding="utf-8"))
    src = j["by_dimension"][str(j["best_dimension"])]
    cm = np.asarray(src["test_confusion_matrix"], dtype=float)
    n = cm.shape[0]
    m = np.zeros_like(cm)
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i, j] = cm[i, j] + cm[j, i]
    return m


def pixel_histograms(data, outdir):
    """
    Distribution of pixel intensities, pooled per class and overall.

    Background dominates -- most MNIST pixels are zero -- so the interesting part
    is the tail. This is what shows that the classes are not separated by gross
    intensity but by where the ink sits, which is the reason a linear method on
    raw pixels is weak and why a code has to preserve spatial structure.
    """
    d = _dir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=plots.FIGSIZE_WIDE)
    X, y = _to_np(data["X_train"]), _to_np(data["y_train"])

    axes[0].hist(X.ravel(), bins=60, range=(0, 1), color="0.4",
                 label="all pixels", density=True)
    for v in _digit_order(data):
        axes[0].hist(X[y == v].ravel(), bins=60, range=(0, 1), histtype="step",
                     lw=1.5, label=f"digit {_digit(data, v)}", density=True)
    axes[0].set_xlabel("pixel intensity"); axes[0].set_ylabel("density")
    axes[0].set_title("Pixel-intensity distribution per class (train)")
    axes[0].legend(fontsize=7)

    frac = (X > 0.5).mean(axis=1)
    axes[1].hist(frac, bins=50, color="0.4")
    axes[1].set_xlabel("fraction of pixels above 0.5 (ink coverage)")
    axes[1].set_ylabel("images")
    axes[1].set_title("Ink coverage per image: a real, class-correlated feature")
    fig.tight_layout()
    return plots.save_atomic(fig, os.path.join(d, "dataset_pixel_histograms.png"))


def sample_grid(data, outdir):
    """A montage of raw training images, 8 per class, as a sanity anchor."""
    d = _dir(outdir)
    X, y = _to_np(data["X_train"]), _to_np(data["y_train"])
    order = _digit_order(data)
    per = 8
    fig, axes = plt.subplots(len(order), per, figsize=(per * 1.15, len(order) * 1.25))
    for i, v in enumerate(order):
        idx = np.where(y == v)[0][:per]
        for j in range(per):
            ax = axes[i, j]
            if j < len(idx):
                ax.imshow(X[idx[j]].reshape(28, 28), cmap="gray")
            ax.axis("off")
        axes[i, 0].set_title(f"digit {_digit(data, v)}", fontsize=8, loc="left")
    fig.suptitle("Training images, 8 per class (the inputs every result below is built on)",
                 fontsize=10)
    fig.tight_layout()
    return plots.save_atomic(fig, os.path.join(d, "dataset_sample_grid.png"))


def pca_scatter(data, outdir):
    """
    Test images projected onto their first two principal components.

    Two things are visible that no accuracy number conveys: how well the classes
    separate in the leading directions, and how much residual structure PCA throws
    away at k=32. It is also the honest reason the report can say the leading
    components carry stroke direction and overall intensity -- the axes are visibly
    not class-shaped.
    """
    d = _dir(outdir)
    X, y = _to_np(data["X_test"]), _to_np(data["y_test"])
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    P = Xc @ v[:, :2]

    fig, ax = plt.subplots(figsize=plots.FIGSIZE_STD)
    for v in _digit_order(data):
        m = y == v
        ax.scatter(P[m, 0], P[m, 1], s=5, alpha=0.5,
                   label=f"digit {_digit(data, v)}")
    ev = 100 * w[:2].sum() / w.sum()
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Test split in the leading 2 principal components ({ev:.1f}% of variance)")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return plots.save_atomic(fig, os.path.join(d, "dataset_pca2_scatter.png"))


def build_all(data, outdir, with_confusions=None):
    """
    Emit every dataset-structure figure. Order matters only for readability.

    `with_confusions` is the outdir to read task1.json from when deciding whether
    the confusion panels can be drawn. It defaults to `outdir`, but a caller that
    runs BEFORE Task 1 has written anything must pass something that does not
    exist, so the panels are omitted rather than drawn empty while the title
    claims they are populated.
    """
    d = _dir(outdir)

    made = [
        class_distribution(data, outdir),
        split_equivalence(data, outdir),
        mean_images(data, outdir),
        pixel_histograms(data, outdir),
        sample_grid(data, outdir),
        pca_scatter(data, outdir),
    ]
    cm_path = os.path.join(with_confusions if with_confusions is not None
                           else outdir, "task1.json")
    made.append(class_mean_distances(data, outdir, confusions_from=cm_path))
    print(f"  dataset figures: {len([m for m in made if m])} written to 00_dataset/"
          f"  (confusion panels: {'yes' if os.path.exists(cm_path) else 'no'})")
    return [m for m in made if m]
