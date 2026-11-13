"""Training until the sheet's convergence criterion, evaluation and checkpoints."""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

import config
from models import SeqClassifier


@dataclass
class History:
    epoch_error: list = field(default_factory=list)  # average training error per epoch
    epoch_time: list = field(default_factory=list)   # seconds per epoch
    stop_reason: str = ""                            # "converged", "cap" or "diverged"


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer {name}")


def train_one_epoch(model, loader, optimizer, device, clip=None):
    """One pass over the training data. Returns the average error (mean cross-entropy per sample)
    of the batches seen during the epoch."""
    model.train()
    total_error = 0.0
    n_samples = 0
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_error += loss.item() * len(y)
        n_samples += len(y)
    return total_error / n_samples


def has_converged(previous_error, current_error, tol):
    """Sheet: difference between average errors of successive epochs falls below the threshold."""
    return abs(current_error - previous_error) < tol


def train_until_converged(model, loader, optimizer, device, tol, max_epochs,
                          clip=None, epoch_error=None, log_every=0):
    """Train until |E(k) - E(k-1)| < tol. max_epochs is only a safety cap.

    epoch_error = "running":   E(k) is the mean error of the batches during epoch k.
    epoch_error = "full_pass": E(k) is recomputed on the whole training set after epoch k.
    """
    epoch_error = epoch_error or config.EPOCH_ERROR
    history = History()
    previous = None
    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        error = train_one_epoch(model, loader, optimizer, device, clip)
        if epoch_error == "full_pass":
            error = evaluate(model, loader, device)["loss"]
        history.epoch_time.append(time.perf_counter() - start)
        history.epoch_error.append(error)

        if log_every and epoch % log_every == 0:
            print(f"    epoch {epoch:5d}  E = {error:.6f}")
        if not math.isfinite(error):
            history.stop_reason = "diverged"
            return history
        if previous is not None and has_converged(previous, error, tol):
            history.stop_reason = "converged"
            return history
        previous = error
    history.stop_reason = "cap"
    return history


# ---------------- Evaluation ----------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Predictions, true labels, sequence lengths and mean cross-entropy over a dataset."""
    model.eval()
    y_true, y_pred, lengths_all = [], [], []
    total_loss = 0.0
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(dim=1).tolist())
        lengths_all.extend(lengths.tolist())
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "lengths": lengths_all,
        "loss": total_loss / len(y_true),
    }


def confusion_matrix(y_true, y_pred, n_classes):
    """Rows = true class, columns = predicted class."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def compute_metrics(y_true, y_pred, n_classes):
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    per_class_total = matrix.sum(axis=1)
    recall = [matrix[k, k] / per_class_total[k] if per_class_total[k] else 0.0 for k in range(n_classes)]
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "per_class_recall": [float(r) for r in recall],
        "confusion_matrix": matrix.tolist(),
    }


# ---------------- Checkpoints ----------------

def save_checkpoint(path, model, info):
    """info: plain dict with everything needed to rebuild the model and its preprocessing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "info": info}, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    info = checkpoint["info"]
    dense_sizes = info.get("dense_sizes", [])  # checkpoints saved before per-time-step layers existed have none
    model = SeqClassifier(info["cell"], info["input_dim"], info["hidden_sizes"], len(info["classes"]), dense_sizes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, info
