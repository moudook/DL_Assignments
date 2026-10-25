"""
part_b1.py - Part B Q1 (Caltech-101, 5 classes): three CNNs from scratch.

    python part_b1.py [--outdir DIR] [--quick]

B1a train/val accuracy + confusion matrices per architecture (no test here),
B1b best architecture by validation accuracy -> test, B1c feature maps of one
training image, B1d most activated last-conv neuron per class image, traced to
its input patch. Writes <outdir>/B1/results.json.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import data
import layers
import plots
import receptive
import train
from evaluate import summary
from results import save_json


def log(msg):
    print(msg, flush=True)


def load(quick):
    c = data.load_caltech()
    if quick:
        c["train"] = data.subset_per_class(c["train"], config.QUICK["per_class_train"])
        for s in ("val", "test"):
            c[s] = data.subset_per_class(c[s], config.QUICK["per_class_eval"])
    mean, std = train.caltech_stats(c["train"]["X"])
    T = {s: (train.caltech_tensor(c[s]["X"], mean, std), torch.from_numpy(c[s]["y"]))
         for s in config.SPLITS}
    return c, T, mean, std


def select_best(archs):
    """Val accuracy, then val loss, then fewer parameters (fixed before training)."""
    return min(archs, key=lambda n: (-archs[n]["metrics"]["val"]["accuracy"],
                                     archs[n]["metrics"]["val"]["loss"],
                                     archs[n]["params"]))


def b1c(c, sel, model, arch_name, out):
    """All 8 conv1 maps and 8 maps of every later conv layer for one image."""
    i = sel["B_single"]
    img = c["train"]["X"][i]
    T = train.caltech_tensor(img[None], *c["_stats"])
    with torch.no_grad():
        feats = model.features(T)
    res = {"image": sel["B_single_file"], "layers": []}
    for li, f in enumerate(feats):
        a = f["conv"][0].numpy()                     # post-ReLU, pre-pool
        C, H, W = a.shape
        if li == 0:
            idx = list(range(C))[:config.N_MAPS_B1C]
            rule = "all %d maps" % C
        else:
            idx = np.argsort(-a.reshape(C, -1).mean(1), kind="stable")[:config.N_MAPS_B1C].tolist()
            rule = "top-%d of %d maps by mean activation" % (config.N_MAPS_B1C, C)
        vmax = float(a[idx].max()) or 1.0
        mosaic = [["img"] + ["m%d" % k for k in range(4)],
                  ["img"] + ["m%d" % k for k in range(4, 8)]]
        fig, ax = plt.subplot_mosaic(mosaic, figsize=(10, 4.6), layout="constrained",
                                     width_ratios=[1.6, 1, 1, 1, 1])
        plots.show_image(ax["img"], img, title="input (%s)" % c["classes"][c["train"]["y"][i]])
        for k, m in enumerate(idx):
            mean = float(a[m].mean())
            h = plots.show_positive(ax["m%d" % k], a[m], vmax=vmax,
                                    title="map %d (mean %.2f)" % (m + 1, mean))
        fig.colorbar(h, ax=[ax["m3"], ax["m7"]], shrink=0.8, label="activation")
        fig.suptitle("B1c %s conv%d (%d@%dx%d, after ReLU): %s, shared colour scale"
                     % (arch_name, li + 1, C, H, W, rule))
        plots.save_fig(fig, out / ("B1c_conv%d_maps" % (li + 1)))
        res["layers"].append({"layer": "conv%d" % (li + 1), "shape": [C, H, W],
                              "rule": rule, "maps_1based": [m + 1 for m in idx],
                              "mean_activation": [float(a[m].mean()) for m in idx],
                              "fraction_zero": [float((a[m] == 0).mean()) for m in idx]})
    return res


def b1d(c, sel, model, arch_name, out, tag="B1d"):
    """Most activated neuron of the last conv layer per image, traced back."""
    imgs = c["train"]["X"][sel["B_images"]]
    T = train.caltech_tensor(imgs, *c["_stats"])
    with torch.no_grad():
        a = model.features(T)[-1]["conv"].numpy()     # post-ReLU, pre-pool
    last = len(model.blocks) - 1
    specs = model.conv_specs(last)
    rf, jump = receptive.rf_size_and_jump(specs)
    fig, axes = plots.new_fig(len(imgs), 2, panel=2.4, extra_h=0.7)
    rows = []
    for r, idx in enumerate(sel["B_images"]):
        cls = c["classes"][c["train"]["y"][idx]]
        flat = int(a[r].argmax())
        m, i, j = np.unravel_index(flat, a[r].shape)
        box, clipped = receptive.neuron_box(specs, int(i), int(j), 224, 224)
        plots.show_image(axes[r, 0], imgs[r],
                         title="map %d (%d,%d), act %.2f" % (m + 1, i + 1, j + 1, a[r].max()))
        plots.draw_box(axes[r, 0], box, plots.BOX_COLOURS[0], lw=2)
        axes[r, 0].set_ylabel(cls)
        r0, r1, c0, c1 = box
        plots.show_image(axes[r, 1], imgs[r][r0:r1 + 1, c0:c1 + 1],
                         title="patch %dx%d" % (r1 - r0 + 1, c1 - c0 + 1))
        rows.append({"image": sel["B_files"][r], "class": cls,
                     "neuron": {"map_1based": int(m) + 1, "row_1based": int(i) + 1,
                                "col_1based": int(j) + 1, "activation": float(a[r].max())},
                     "input_box_rows_1based": [r0 + 1, r1 + 1],
                     "input_box_cols_1based": [c0 + 1, c1 + 1], "clipped": clipped})
    fig.suptitle("%s %s\nmost activated conv%d neuron (%dx%d maps)\n"
                 "receptive field %dx%d px, stride %d px"
                 % (tag, arch_name, last + 1, a.shape[2], a.shape[3], rf, rf, jump))
    plots.save_fig(fig, out / ("%s_patches" % tag))
    return {"layer": "conv%d after ReLU, before pooling" % (last + 1),
            "receptive_field": rf, "jump": jump, "images": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(config.OUT_DIR))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(config.TORCH_THREADS)
    out = Path(args.outdir) / "B1"
    out.mkdir(parents=True, exist_ok=True)

    c, T, mean, std = load(args.quick)
    c["_stats"] = (mean, std)
    sel = data.select_b(c)
    log("Part B images: %s | single: %s" % (", ".join(sel["B_files"]), sel["B_single_file"]))
    max_ep = config.QUICK["max_epochs"] if args.quick else config.MAX_EPOCHS_B
    (Xtr, ytr), (Xva, yva), (Xte, yte) = T["train"], T["val"], T["test"]

    archs, models = {}, {}
    for n in (1, 2, 3):
        name = layers.ARCH_NAMES[n]

        def build(n=n):
            return layers.caltech_arch(n, config.SEED)

        record, model, hist = train.lr_grid(build, Xtr, ytr, Xva, yva, config.BATCH_SIZE_B,
                                            max_ep, config.SEED, tag="Arch-%d" % n, log=log)
        torch.save(model.state_dict(), out / ("B1_arch%d_model.pt" % n))
        winner = next(r for r in record["runs"] if r["lr"] == record["winner_lr"])
        metrics = {}
        for split, X, y in (("train", Xtr, ytr), ("val", Xva, yva)):
            loss, _, z = train.loss_acc(model, X, y)
            metrics[split] = summary(y.numpy(), z.argmax(1).numpy(), c["classes"], loss)
            plots.confusion_matrix_plot(metrics[split]["confusion_matrix"], c["classes"],
                                        "%s - %s" % (name, split),
                                        out / ("B1a_arch%d_cm_%s" % (n, split)))
        plots.curves_plot(hist, winner["best_epoch"], "%s (lr %g, batch %d, momentum %.1f)"
                          % (name, record["winner_lr"], config.BATCH_SIZE_B, config.MOMENTUM),
                          out / ("B1a_arch%d_curves" % n))
        plots.grid_plot(record, "%s: learning-rate grid" % name, out / ("B1a_arch%d_lr_grid" % n))
        rf = receptive.rf_size_and_jump(model.conv_specs(len(model.blocks) - 1))
        archs[n] = {"name": name, "params": layers.count_params(model),
                    "size_table": model.size_table(),
                    "last_conv_receptive_field": rf[0], "last_conv_jump": rf[1],
                    "lr_grid": record, "metrics": metrics}
        models[n] = model
        log("%s: train %.3f  val %.3f (lr %g)" % (name, metrics["train"]["accuracy"],
                                                  metrics["val"]["accuracy"], record["winner_lr"]))

    best = select_best(archs)
    bname = archs[best]["name"]
    loss, _, z = train.loss_acc(models[best], Xte, yte)
    test = summary(yte.numpy(), z.argmax(1).numpy(), c["classes"], loss)
    plots.confusion_matrix_plot(test["confusion_matrix"], c["classes"],
                                "%s (best by val) - test" % bname, out / "B1b_best_cm_test")
    log("best: %s  test %.3f" % (bname, test["accuracy"]))

    res = {"quick": args.quick, "seed": config.SEED, "selection": sel,
           "split_sizes": {s: len(c[s]["y"]) for s in config.SPLITS},
           "normalisation": {"mean": mean, "std": std, "source": "train split"},
           "architectures": {"arch%d" % n: v for n, v in archs.items()},
           "B1b": {"best": "arch%d" % best, "name": bname,
                   "rule": "highest val accuracy -> lowest val loss -> fewer params",
                   "test": test},
           "B1c": b1c(c, sel, models[best], bname, out),
           "B1d": b1d(c, sel, models[best], bname, out)}
    p = save_json(res, out / "results.json")
    log("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
