"""
train.py - torch training: mini-batch SGD with momentum, early stopping on
validation loss with best-weight restore, and the learning-rate grid.

All data is preloaded as tensors and batched by index, so there is no
DataLoader and no worker processes (A4 hit Windows error 1455 with workers).
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from protocol import EarlyStopper, run_flags, run_grid


# ---------------------------------------------------------------- tensors

def mnist_tensor(X):
    """uint8 (N,28,28) -> float (N,1,28,28) in [0, 1]."""
    return torch.from_numpy(X).float().div_(255.0).unsqueeze(1)


def caltech_stats(X_train):
    """Per-channel mean/std of the TRAIN split, on [0, 1] pixels."""
    x = X_train.reshape(-1, 3).astype(np.float64) / 255.0
    return x.mean(0), x.std(0)


def caltech_tensor(X, mean, std):
    """uint8 (N,224,224,3) -> normalised float (N,3,224,224)."""
    t = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (t - m) / s


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def logits_of(model, X, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def loss_acc(model, X, y):
    z = logits_of(model, X)
    return F.cross_entropy(z, y).item(), (z.argmax(1) == y).float().mean().item(), z


# ---------------------------------------------------------------- training

def fit(model, Xtr, ytr, Xva, yva, lr, batch_size, max_epochs, seed,
        patience=config.ES_PATIENCE, tag="", log=print):
    """Train until early stopping; restore and return the best-val-loss weights.

    History epoch 0 is the untrained model. train_loss/train_acc are running
    averages over the epoch's mini-batches; val_* are measured after it.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM)
    gen = torch.Generator().manual_seed(seed)
    stopper = EarlyStopper(patience)
    vl, va, _ = loss_acc(model, Xva, yva)
    hist = {"epoch": [0], "train_loss": [None], "train_acc": [None],
            "val_loss": [vl], "val_acc": [va]}
    stopper.step(0, vl)
    best_state = copy.deepcopy(model.state_dict())
    spikes, min_train = 0, math.inf
    t0 = time.perf_counter()
    n = len(Xtr)
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            z = model(Xtr[idx])
            loss = F.cross_entropy(z, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (z.argmax(1) == ytr[idx]).sum().item()
        tl, ta = tot / n, correct / n
        vl, va, _ = loss_acc(model, Xva, yva)
        for k, v in (("epoch", epoch), ("train_loss", tl), ("train_acc", ta),
                     ("val_loss", vl), ("val_acc", va)):
            hist[k].append(v)
        if math.isfinite(tl):
            if epoch > 1 and tl > 1.5 * min_train:
                spikes += 1
            min_train = min(min_train, tl)
        improved, stop = stopper.step(epoch, vl if math.isfinite(tl) else math.nan)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0 or stop or epoch == 1:
            log("  %s ep %3d  train %.4f/%.3f  val %.4f/%.3f  best ep %s"
                % (tag, epoch, tl, ta, vl, va, stopper.best_epoch))
        if stop:
            break
    model.load_state_dict(best_state)
    info = run_flags(hist, stopper, max_epochs, epoch, patience)
    info.update({"lr": lr, "loss_spikes": spikes,
                 "seconds": round(time.perf_counter() - t0, 1)})
    return hist, info


def lr_grid(build, Xtr, ytr, Xva, yva, batch_size, max_epochs, seed, tag, log=print):
    """The shared grid protocol (protocol.run_grid) for a torch model.

    Each lr starts from an identically initialised model (`build()` reseeds).
    Returns (grid record, winning model, its history).
    """
    def run_one(lr):
        model = build()
        hist, info = fit(model, Xtr, ytr, Xva, yva, lr, batch_size, max_epochs,
                         seed, tag=tag, log=log)
        tl, ta, _ = loss_acc(model, Xtr, ytr)
        info.update({"train_loss_at_best": tl, "train_acc_at_best": ta,
                     "val_acc_at_best": hist["val_acc"][info["best_epoch"]]})
        return info, hist, model

    return run_grid(run_one, tag, log)
