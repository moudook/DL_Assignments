"""
protocol.py - the training protocol shared by the torch models and the Keras
VGG19 head, so both follow literally the same rules (README.md, Methodology).
Framework-free.

EarlyStopper: stop once val loss has gone `patience` epochs without a new
minimum (strictly lower than the best so far); the caller restores the best
epoch's weights. A non-finite val loss stops immediately.

run_grid: every model runs the same learning-rate grid; the winner is the
lowest best-val-loss; if it sits at a grid edge, one extra step that way is
tried, once.
"""

import math

import config

EXPLODE = 10.0     # val loss above 10x its untrained value counts as exploded
COLLAPSE_ACC = 1.25 / 5    # final train accuracy at or below 1.25 x chance (5 classes)


class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self.best = math.inf
        self.best_epoch = None
        self.wait = 0
        self.diverged = False

    def step(self, epoch, val_loss):
        """Returns (improved, stop)."""
        if not math.isfinite(val_loss):
            self.diverged = True
            return False, True
        if val_loss < self.best:
            self.best, self.best_epoch, self.wait = val_loss, epoch, 0
            return True, False
        self.wait += 1
        return False, self.wait >= self.patience


def run_flags(hist, stopper, max_epochs, last_epoch, patience):
    """Instability / outcome flags recorded for every run."""
    finite = [v for v in hist["val_loss"] if v is not None and math.isfinite(v)]
    init = hist["val_loss"][0]
    return {
        "best_epoch": stopper.best_epoch,
        "stop_epoch": last_epoch,
        "hit_epoch_cap": last_epoch == max_epochs and stopper.wait < patience,
        "best_val_loss": stopper.best,
        "diverged": stopper.diverged,
        "exploded": (max(finite) > EXPLODE * init) if finite else True,
        "no_learning": stopper.best > 0.95 * init,
        # collapsed: the network ends predicting (almost) one class, e.g. dead
        # ReLUs at a too-large lr (loss ln 5, accuracy 20%); the 0.95 x initial
        # loss test above misses this because the initial loss is 2.8-5.4
        "collapsed": (hist["train_acc"][-1] is not None
                      and hist["train_acc"][-1] <= COLLAPSE_ACC),
    }


def run_grid(run_one, tag, log=print):
    """run_one(lr) -> (info dict with "best_val_loss", history, model).

    Returns (record, winning model, winning history).
    """
    runs, best = [], {"loss": math.inf, "lr": None, "model": None, "hist": None}

    def run(lr):
        log("%s  lr %g" % (tag, lr))
        info, hist, model = run_one(lr)
        info["lr"] = lr
        runs.append({"info": info, "history": hist})
        if info["best_val_loss"] < best["loss"]:
            best.update(loss=info["best_val_loss"], lr=lr, model=model, hist=hist)

    for lr in config.LR_GRID:
        run(lr)
    if best["lr"] is None:
        raise RuntimeError("%s: every learning rate diverged" % tag)
    edge = None
    if best["lr"] == config.LR_GRID[0]:
        edge = config.LR_EDGE_STEP["low"]
    elif best["lr"] == config.LR_GRID[-1]:
        edge = config.LR_EDGE_STEP["high"]
    if edge is not None:
        log("%s  winner at grid edge -> one extra step lr %g" % (tag, edge))
        run(edge)
    record = {
        "grid": config.LR_GRID,
        "edge_extension": edge,
        "winner_lr": best["lr"],
        "selection_rule": "lowest early-stopped validation loss",
        "runs": [r["info"] for r in runs],
        "histories": {str(r["info"]["lr"]): r["history"] for r in runs},
    }
    return record, best["model"], best["hist"]
