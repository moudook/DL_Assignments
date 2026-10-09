"""
evaluate.py — metrics, confusion matrices, and reconstruction figures for A4.

Contents
--------
  evaluate_classifier      accuracy + confusion matrix + per-class breakdown
  reconstruction_grid      one image per class per split, original vs reconstructed
  maximally_activating     the training input that maximises each bottleneck neuron
  format_table             markdown-ready table for the report

All numerics are fp32 and full-batch, matching A3 so the comparison tables are
like-for-like. Confusion matrices use the fixed class order {0,4,5,6,7} rather
than whatever order the model happened to see, so rows and columns line up
across every figure in the report.
"""

import os

import numpy as np
import torch

from data import CLASS_NAMES, NUM_CLASSES

IMG_SIZE = 28


# ── classification metrics ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate_classifier(model, data, device=None, split="test"):
    """
    Accuracy, confusion matrix and predictions for one split.

    Returns dict with targets, preds, accuracy, and the confusion matrix as a
    list-of-lists (so it is JSON-serialisable without a numpy encoder).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X = data[f"X_{split}"]
    y = data[f"y_{split}"]

    preds = []
    for i in range(0, X.size(0), 4096):
        preds.append(model(X[i:i + 4096]).argmax(1))
    preds = torch.cat(preds)

    targets = y.to(preds.device)
    acc = (targets == preds).float().mean().item()

    # Explicit labels list: guarantees the matrix is 5x5 with rows/cols in
    # CLASS_NAMES order even if some class is absent from this split.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
        cm[t, p] += 1

    per_class = []
    for c in range(NUM_CLASSES):
        total = int(cm[c].sum())
        correct = int(cm[c, c])
        per_class.append({
            "digit": CLASS_NAMES[c],
            "correct": correct,
            "errors": total - correct,
            "accuracy": (correct / total) if total else float("nan"),
            "support": total,
        })

    return {
        "split": split,
        "accuracy": acc,
        "targets": targets.cpu().numpy().tolist(),
        "preds": preds.cpu().numpy().tolist(),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "n": int(X.size(0)),
    }


def most_confused(cm, top=3):
    """
    Off-diagonal pairs with the most errors, largest first.

    Used for the report's prose analysis: the assignment wants genuine inference
    about WHY classes are confused, and the dominant off-diagonal pairs are the
    factual basis for it.
    """
    pairs = []
    arr = np.asarray(cm)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if i != j and arr[i, j] > 0:
                pairs.append({
                    "true": CLASS_NAMES[i],
                    "pred": CLASS_NAMES[j],
                    "count": int(arr[i, j]),
                })
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top]


# ── reconstruction figures ──────────────────────────────────────────────────

def _to_grid_images(vecs):
    """
    (N, 784) float tensor -> list of N 28x28 numpy arrays, values clipped to [0,1].

    Clipping matters: sigmoid output is already in [0,1], but float round-trip can
    leave values a hair outside, and matplotlib's default colour limits would then
    stretch the colour scale and misrepresent contrast.
    """
    arr = vecs.detach().cpu().numpy().reshape(-1, IMG_SIZE, IMG_SIZE)
    return [np.clip(a, 0.0, 1.0) for a in arr]


@torch.no_grad()
def reconstruction_grid(model, data, split, indices, device=None):
    """
    Reconstructed versions of specific samples, for A4 Task-2d / Task-5c.

    A4 requires one image from each class, for train/val/test, shown alongside
    its reconstruction. `indices` comes from data.one_per_class().

    Returns (originals, reconstructions, indices) as lists of 28x28 arrays plus
    the sample indices used. The third element previously evaluated the tensor's
    `.shape` for truthiness and returned a list of ints, which was meaningless;
    every caller discarded it, so nothing broke, but it would mislead anyone who
    started using the return value.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X = data[f"X_{split}"]
    idx = torch.as_tensor(indices, device=X.device)
    sample = X[idx]
    rec = model(sample)

    return _to_grid_images(sample), _to_grid_images(rec), list(indices)


# ── Task-6 weight visualisation ─────────────────────────────────────────────

@torch.no_grad()
def maximally_activating(model, X, n_units=None, device=None):
    """
    For each bottleneck neuron, the training image that activates it most.

    A4 Task-6 asks to "plot the inputs as images that maximally activate each of
    the neurons of the hidden representations (plot of weights from the input
    layer to the compressed layer)". Two readings, and the honest reading is
    both:

      (a) the training input that maximises each unit's activation, and
      (b) the corresponding ENCODER WEIGHT VECTOR reshaped to 28x28.

    The assignment names (a) and glosses it with (b). Both are informative and
    they disagree when a unit fires on a motif that recurs across many images, so
    returning both lets the report make that comparison explicitly rather than
    guessing which was meant.

    Returns dict with:
        inputs   list of (n_units, 784) arrays - the maximally activating images
        weights  list of (784,) arrays        - encoder weight rows, as 28x28
        acts     list of (n_units,) activations at those inputs
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    Z = model.encode(X)                      # (N, k)
    k = Z.size(1) if n_units is None else min(n_units, Z.size(1))

    acts, inputs = [], []
    for j in range(k):
        col = Z[:, j]
        # The bottleneck is LINEAR (A4 mandate), so activations are signed and a
        # unit may be reliably NEGATIVE. Taking the plain argmax would then
        # select the input that pushes a unit furthest POSITIVE, which for a
        # negative-preferring unit is not the "maximally activating" image in any
        # useful sense - it just reports the least-negative case.
        #
        # Select by largest ABSOLUTE activation instead: that is the input the
        # unit responds to most strongly in either direction, which is what
        # "maximally activate" means for a signed code. Weights are reported
        # alongside activations so the report can note sign convention.
        best = int(col.abs().argmax().item())
        acts.append(float(col[best].item()))
        inputs.append(X[best].detach().cpu().numpy())

    # Encoder weight rows. For the 1-hidden AE the encoder is a single
    # Linear(784, k), so weight[j] is exactly the 784->j connection. For the
    # 3-hidden AE the bottleneck is reached through two layers, so the effective
    # input-layer contribution is the product W1 @ W2 row j; we still expose the
    # first-layer rows since A4 says "input layer to the compressed layer".
    W1 = None
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.Linear):
            W1 = m.weight.detach().cpu().numpy()
            break
    if W1 is None:
        return {"inputs": inputs, "weights": [], "acts": acts, "k": k}
    W1 = W1[:k]                             # (k, 784)

    return {
        "inputs": inputs,
        "weights": [W1[j] for j in range(k)],
        "acts": acts,
        "k": k,
    }


# ── report helpers ──────────────────────────────────────────────────────────

def format_table(rows, headers):
    """Markdown table, for pasting into the report or a results summary."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def accuracy_row(label, val_acc, test_result=None, extra=""):
    """One row of a comparison table: label, val%, test%, then extra."""
    test_s = f"{100 * test_result['accuracy']:.2f}" if test_result else "-"
    return f"| {label} | {100 * val_acc:.2f} | {test_s} | {extra} |"