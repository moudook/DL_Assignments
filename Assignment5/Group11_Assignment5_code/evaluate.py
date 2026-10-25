"""
evaluate.py - accuracy, confusion matrix and per-class recall from label
arrays. Framework-free (numpy), used by every part.
"""

import numpy as np


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def summary(y_true, y_pred, classes, loss=None):
    """JSON-ready metrics for one split. Rows of `cm` are true classes."""
    cm = confusion(y_true, y_pred, len(classes))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    out = {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "correct": int(np.trace(cm)),
        "total": int(cm.sum()),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": dict(zip(classes, recall.round(4).tolist())),
    }
    if loss is not None:
        out["loss"] = float(loss)
    return out
