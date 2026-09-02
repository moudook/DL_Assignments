from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_per_class(cm: np.ndarray) -> np.ndarray:
    n = cm.shape[0]
    out = np.zeros(n)
    for i in range(n):
        col_sum = cm[:, i].sum()
        out[i] = cm[i, i] / col_sum if col_sum > 0 else 0.0
    return out


def recall_per_class(cm: np.ndarray) -> np.ndarray:
    n = cm.shape[0]
    out = np.zeros(n)
    for i in range(n):
        row_sum = cm[i, :].sum()
        out[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0
    return out


def f1_per_class(precisions: np.ndarray, recalls: np.ndarray) -> np.ndarray:
    out = np.zeros_like(precisions)
    for i in range(precisions.shape[0]):
        p, r = precisions[i], recalls[i]
        out[i] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return out


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    prec = precision_per_class(cm)
    rec = recall_per_class(cm)
    f1 = f1_per_class(prec, rec)
    return {
        "name": name,
        "accuracy": accuracy(y_true, y_pred),
        "precisions": prec,
        "recalls": rec,
        "f1_scores": f1,
        "mean_precision": float(prec.mean()),
        "mean_recall": float(rec.mean()),
        "mean_f1": float(f1.mean()),
        "confusion_matrix": cm,
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def percent_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (rmse(y_true, y_pred) / float(np.mean(y_true))) * 100.0


def regression_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
) -> dict:
    return {
        "name": name,
        "rmse": rmse(y_true, y_pred),
        "percent_rmse": percent_rmse(y_true, y_pred),
    }