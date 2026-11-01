"""
seeds_b1.py - additional B1 experiment: how stable is the architecture ranking?

    python seeds_b1.py [--outdir DIR] [--quick]

For every seed in config.SEEDS_B1 the full B1 protocol of part_b1.py is rerun
unchanged (same learning-rate grid, early stopping and selection rule); only
the initial weights and the order of the training images change. Seed 11
reproduces the main B1 results. The selection per seed uses validation only;
test accuracy of all three architectures is recorded afterwards, for this
analysis alone, and plays no part in any choice. Writes
<outdir>/B1_seeds/results.json.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import config
import layers
import train
from evaluate import summary
from part_b1 import load, log, select_best
from results import save_json


def run_seed(seed, T, classes, max_ep):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = T["train"], T["val"], T["test"]
    archs = {}
    for n in (1, 2, 3):
        def build(n=n):
            return layers.caltech_arch(n, seed)

        record, model, _ = train.lr_grid(build, Xtr, ytr, Xva, yva, config.BATCH_SIZE_B,
                                         max_ep, seed, tag="seed %d Arch-%d" % (seed, n),
                                         log=lambda *_: None)
        metrics = {}
        for split, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
            loss, _, z = train.loss_acc(model, X, y)
            metrics[split] = summary(y.numpy(), z.argmax(1).numpy(), classes, loss)
        archs[n] = {"name": layers.ARCH_NAMES[n], "params": layers.count_params(model),
                    "lr": record["winner_lr"], "metrics": metrics}
        log("seed %d %s: lr %g  train %.3f  val %.3f  test %.3f"
            % (seed, layers.ARCH_NAMES[n], record["winner_lr"], metrics["train"]["accuracy"],
               metrics["val"]["accuracy"], metrics["test"]["accuracy"]))
    best = select_best(archs)
    log("seed %d: selected %s" % (seed, archs[best]["name"]))
    return {"seed": seed, "selected": "arch%d" % best,
            "architectures": {"arch%d" % n: {"name": a["name"], "lr": a["lr"],
                                              **{s: a["metrics"][s]["accuracy"]
                                                 for s in config.SPLITS},
                                              "val_loss": a["metrics"]["val"]["loss"]}
                              for n, a in archs.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(config.OUT_DIR))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(config.TORCH_THREADS)
    out = Path(args.outdir) / "B1_seeds"
    out.mkdir(parents=True, exist_ok=True)

    c, T, _, _ = load(args.quick)
    max_ep = config.QUICK["max_epochs"] if args.quick else config.MAX_EPOCHS_B
    runs = [run_seed(s, T, c["classes"], max_ep) for s in config.SEEDS_B1]

    stats = {}
    for k in ("arch1", "arch2", "arch3"):
        stats[k] = {"name": runs[0]["architectures"][k]["name"]}
        for s in config.SPLITS:
            v = np.array([r["architectures"][k][s] for r in runs])
            # sample std (ddof=1): spread across seeds, not of the mean
            stats[k][s] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                           "min": float(v.min()), "max": float(v.max())}
    selected = [r["selected"] for r in runs]
    res = {"quick": args.quick, "seeds": config.SEEDS_B1,
           "protocol": "part_b1.py protocol per seed: lr grid + one edge step, early "
                       "stopping on val loss, selection by val acc -> val loss -> params",
           "test_use": "recorded for all architectures after selection; not used for any choice",
           "runs": runs, "summary": stats,
           "selected_count": {k: selected.count(k) for k in ("arch1", "arch2", "arch3")}}
    for k, v in stats.items():
        log("%s  train %.3f+-%.3f  val %.3f+-%.3f  test %.3f+-%.3f"
            % (v["name"], v["train"]["mean"], v["train"]["std"], v["val"]["mean"],
               v["val"]["std"], v["test"]["mean"], v["test"]["std"]))
    log("selected per seed: %s" % selected)
    p = save_json(res, out / "results.json")
    log("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
