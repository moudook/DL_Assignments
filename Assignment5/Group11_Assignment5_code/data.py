"""
data.py - framework-free dataset loading (PIL + numpy).

Both the torch code (Part A, B1) and the TensorFlow code (B2) load pixels
through here, so B1 and B2 see byte-identical 224x224 images and the B2a vs
B1b comparison differs only in the model.

Arrays are uint8 and cached to an .npz keyed by the file list; scaling and
normalisation happen in the consumer.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_split(root, split, classes):
    """Sorted (relative paths, labels) for one split. Label = index in `classes`."""
    rels, labels = [], []
    for c, name in enumerate(classes):
        d = Path(root) / split / name
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        rels += [p.relative_to(root).as_posix() for p in files]
        labels += [c] * len(files)
    return rels, np.asarray(labels, dtype=np.int64)


def _load_mnist_image(path):
    im = Image.open(path).convert("L")
    if im.size != (28, 28):
        raise ValueError("%s is %s, expected 28x28" % (path, im.size))
    return np.asarray(im, dtype=np.uint8)


def _load_caltech_image(path, size):
    """RGB (greyscale replicated to 3 channels), resized to size x size.

    Plain resize as the sheet says: aspect ratio is not preserved. PIL's
    bilinear filter widens its support when downscaling, so it antialiases.
    Returns (array, was_greyscale).
    """
    im = Image.open(path)
    grey = im.mode in ("L", "LA", "1", "I", "I;16", "F")
    im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8), grey


def _cache_key(root, rels, extra):
    h = hashlib.sha1()
    h.update(str(extra).encode("utf-8"))
    for r in rels:
        st = (Path(root) / r).stat()
        h.update(("%s|%d|%d\n" % (r, st.st_size, int(st.st_mtime))).encode("utf-8"))
    return h.hexdigest()[:16]


def _load(name, root, classes, loader, extra, cache=True):
    out = {}
    for split in config.SPLITS:
        rels, y = list_split(root, split, classes)
        key = _cache_key(root, rels, extra)
        f = config.CACHE_DIR / ("%s_%s_%s.npz" % (name, split, key))
        if cache and f.exists():
            z = np.load(f, allow_pickle=False)
            out[split] = {"X": z["X"], "y": z["y"], "files": list(z["files"]),
                          "grey": z["grey"]}
            continue
        xs, greys = [], []
        for r in rels:
            res = loader(Path(root) / r)
            if isinstance(res, tuple):
                xs.append(res[0])
                greys.append(res[1])
            else:
                xs.append(res)
                greys.append(True)
        X = np.stack(xs)
        grey = np.asarray(greys, dtype=bool)
        if cache:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, X=X, y=y, files=np.asarray(rels), grey=grey)
        out[split] = {"X": X, "y": y, "files": rels, "grey": grey}
    out["classes"] = list(classes)
    out["root"] = Path(root)
    return out


def load_mnist(root=None, cache=True):
    """{split: {X uint8 (N,28,28), y, files, grey}, classes, root}."""
    root = root or config.resolve_dir(config.MNIST_CANDIDATES, config.MNIST_ENV,
                                      config.MNIST_CLASSES)
    return _load("mnist", root, config.MNIST_CLASSES, _load_mnist_image,
                 extra="mnist28", cache=cache)


def load_caltech(root=None, size=config.IMG_SIZE_B, cache=True):
    """{split: {X uint8 (N,size,size,3), y, files, grey}, classes, root}."""
    root = root or config.resolve_dir(config.CALTECH_CANDIDATES, config.CALTECH_ENV,
                                      config.CALTECH_CLASSES)
    return _load("caltech", root, config.CALTECH_CLASSES,
                 lambda p: _load_caltech_image(p, size),
                 extra="caltech%d_bilinear" % size, cache=cache)


def subset_per_class(split_data, n):
    """First n samples of each class (for --quick and overfit tests)."""
    y = split_data["y"]
    idx = np.concatenate([np.flatnonzero(y == c)[:n] for c in np.unique(y)])
    return {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
            for k, v in split_data.items()}


def first_per_class(split_data, n_classes):
    """Index of the first (sorted-name) sample of each class."""
    y = split_data["y"]
    return [int(np.flatnonzero(y == c)[0]) for c in range(n_classes)]


def select_a(mnist):
    """A1/A2/A3b/A3c images: first train file of each class."""
    a = first_per_class(mnist["train"], len(mnist["classes"]))
    return {"A_images": a, "A_files": [mnist["train"]["files"][i] for i in a]}


def select_b(caltech):
    """B1d/B2b/B2d images (first train file per class) and the B1c/B2c image."""
    b = first_per_class(caltech["train"], len(caltech["classes"]))
    single = b[caltech["classes"].index(config.SINGLE_IMAGE_CLASS_B)]
    return {"B_images": b, "B_files": [caltech["train"]["files"][i] for i in b],
            "B_single": single, "B_single_file": caltech["train"]["files"][single]}


def selections(mnist, caltech):
    """The fixed images every question reuses (README.md, Methodology)."""
    return {**select_a(mnist), **select_b(caltech)}


def write_selection(sel, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel, indent=2), encoding="utf-8")


def file_hashes(root, rels):
    """md5 of raw file bytes, for the split-leakage check."""
    out = {}
    for r in rels:
        out[r] = hashlib.md5((Path(root) / r).read_bytes()).hexdigest()
    return out
