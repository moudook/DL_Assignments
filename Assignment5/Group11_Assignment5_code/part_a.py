"""
part_a.py - Part A (MNIST, 5 classes): A1, A2, A3a-c.

    python part_a.py [--outdir DIR] [--quick]

Writes <outdir>/A/results.json, figures, and the trained model checkpoint.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import config
import data
import layers
import plots
import receptive
import train
from conv_manual import conv2d_manual, kaiming_filter, out_size
from evaluate import summary
from results import save_json


def log(msg):
    print(msg, flush=True)


def fmt(a):
    return np.array2string(np.asarray(a), precision=2, suppress_small=True,
                           max_line_width=120)


# ================================================================ A1

def a1(m, sel, out):
    """One Kaiming 5x5 filter traversed over each image by explicit loops."""
    rng = np.random.default_rng(config.SEED)
    k = kaiming_filter((5, 5), 25, rng)
    labels = ["digit %s" % m["classes"][m["train"]["y"][i]] for i in sel["A_images"]]
    imgs = [m["train"]["X"][i] / 255.0 for i in sel["A_images"]]
    maps = [conv2d_manual(im, k, stride=1, pad=0) for im in imgs]
    exp = out_size(28, 5, p=0, s=1)
    log("A1 filter (Kaiming normal, std sqrt(2/25) = %.4f):\n%s" % (np.sqrt(2 / 25), fmt(k)))
    dims = []
    for lab, fm in zip(labels, maps):
        dims.append({"image": lab, "expected": [exp, exp], "obtained": list(fm.shape)})
        log("A1 %s: expected (28 - 5 + 2*0)/1 + 1 = %d -> %dx%d, obtained %dx%d"
            % (lab, exp, exp, exp, *fm.shape))
        assert fm.shape == (exp, exp)

    plots.filter_row([k], ["5x5 filter"], "A1: Kaiming-initialised filter "
                     "(N(0, 2/25))", out / "A1_filter", panel=3.2)
    lim = max(float(np.abs(fm).max()) for fm in maps)
    fig, axes = plots.new_fig(len(imgs), 2, panel=2.0, extra_w=0.9, extra_h=0.6)
    for r, (im, fm, lab) in enumerate(zip(imgs, maps, labels)):
        plots.show_image(axes[r, 0], im)
        axes[r, 0].set_ylabel(lab)
        h = plots.show_signed(axes[r, 1], fm, lim=lim)
    axes[0, 0].set_title("input 28x28")
    axes[0, 1].set_title("feature map 24x24")
    fig.colorbar(h, ax=axes[:, 1].tolist(), shrink=0.5, label="filter response (no activation)")
    fig.suptitle("A1: stride 1, padding 0")
    plots.save_fig(fig, out / "A1_feature_maps")
    return {"images": sel["A_files"], "filter": k, "dims": dims,
            "feature_maps": {lab: fm for lab, fm in zip(labels, maps)}}


# ================================================================ A2

def a2(m, sel, out):
    """Two stacked ConvReLUPool layers, untrained (Kaiming init)."""
    blocks = layers.part_a_blocks(config.SEED)
    labels = ["digit %s" % m["classes"][m["train"]["y"][i]] for i in sel["A_images"]]
    x = train.mnist_tensor(m["train"]["X"][sel["A_images"]])
    with torch.no_grad():
        c1 = blocks[0].conv_out(x)
        p1 = blocks[0].pool_out(c1)
        c2 = blocks[1].conv_out(p1)
        p2 = blocks[1].pool_out(c2)

    dims = []
    h = 28
    for li, (b, conv, pool) in enumerate(((blocks[0], c1, p1), (blocks[1], c2, p2)), 1):
        ce, pe = b.out_hw(h)
        for stage, t, e in (("conv", conv, ce), ("pool", pool, pe)):
            got = list(t.shape[1:])
            exp = [b.out_ch, e, e]
            dims.append({"layer": "layer%d %s" % (li, stage), "expected": exp, "obtained": got})
            log("A2 layer %d %s: expected %s, obtained %s" % (li, stage, exp, got))
            assert got == exp
        h = pe
    for r, lab in enumerate(labels):
        log("A2 %s layer-1 output (6x12x12), filter %d:\n%s"
            % (lab, config.A2_FILTERS_L1[0] + 1, fmt(p1[r, config.A2_FILTERS_L1[0]])))
        log("A2 %s layer-2 output (16x4x4), filter %d:\n%s"
            % (lab, config.A2_FILTERS_L2[0] + 1, fmt(p2[r, config.A2_FILTERS_L2[0]])))

    W1 = blocks[0].weight.detach().numpy()
    W2 = blocks[1].weight.detach().numpy()
    f1, f2 = config.A2_FILTERS_L1, config.A2_FILTERS_L2
    plots.filter_row([W1[i, 0] for i in f1], ["filter %d" % (i + 1) for i in f1],
                     "A2: 3 of 6 layer-1 filters (5x5, untrained)", out / "A2_filters_l1")
    plots.filter_slices([W2[i] for i in f2], ["filter %d" % (i + 1) for i in f2],
                        "A2: 3 of 16 layer-2 filters (6x5x5, untrained), per input channel",
                        out / "A2_filters_l2", panel=1.1, annotate=True)
    imgs = m["train"]["X"][sel["A_images"]]
    cols = []
    for i in f1:
        cols += [("f%d conv+ReLU\n24x24" % (i + 1), c1[:, i].numpy()),
                 ("f%d pooled\n12x12" % (i + 1), p1[:, i].numpy())]
    plots.image_maps_grid(imgs, labels, cols, "A2 layer 1: feature maps of the 3 filters",
                          out / "A2_maps_l1", note="colour scale shared per column")
    cols = []
    for i in f2:
        cols += [("f%d conv+ReLU\n8x8" % (i + 1), c2[:, i].numpy()),
                 ("f%d pooled\n4x4" % (i + 1), p2[:, i].numpy())]
    plots.image_maps_grid(imgs, labels, cols, "A2 layer 2: feature maps of the 3 filters",
                          out / "A2_maps_l2", note="colour scale shared per column")
    res = {"dims": dims, "filters_l1_idx_1based": [i + 1 for i in f1],
           "filters_l2_idx_1based": [i + 1 for i in f2],
           "filters_l1": W1[f1, 0], "filters_l2": W2[f2],
           "layer1_output": {lab: p1[r].numpy() for r, lab in enumerate(labels)},
           "layer2_output": {lab: p2[r].numpy() for r, lab in enumerate(labels)}}
    return res, blocks


# ================================================================ A3

def a3(m, sel, blocks, out, quick):
    Xtr, ytr = train.mnist_tensor(m["train"]["X"]), torch.from_numpy(m["train"]["y"])
    Xva, yva = train.mnist_tensor(m["val"]["X"]), torch.from_numpy(m["val"]["y"])
    Xte, yte = train.mnist_tensor(m["test"]["X"]), torch.from_numpy(m["test"]["y"])
    max_ep = config.QUICK["max_epochs"] if quick else config.MAX_EPOCHS_A

    def build():
        return layers.part_a_cnn(blocks, config.SEED)

    record, model, hist = train.lr_grid(build, Xtr, ytr, Xva, yva, config.BATCH_SIZE_A,
                                        max_ep, config.SEED, tag="A3", log=log)
    torch.save(model.state_dict(), out / "A3_model.pt")
    winner = next(r for r in record["runs"] if r["lr"] == record["winner_lr"])
    classes = ["digit %s" % c for c in m["classes"]]

    metrics = {}
    for split, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        loss, _, z = train.loss_acc(model, X, y)
        metrics[split] = summary(y.numpy(), z.argmax(1).numpy(), classes, loss)
        plots.confusion_matrix_plot(metrics[split]["confusion_matrix"], classes,
                                    "A3 CNN - %s" % split, out / ("A3a_cm_%s" % split))
        log("A3 %s accuracy %.4f" % (split, metrics[split]["accuracy"]))
    plots.curves_plot(hist, winner["best_epoch"], "A3 CNN (lr %g, batch %d, momentum %.1f)"
                      % (record["winner_lr"], config.BATCH_SIZE_A, config.MOMENTUM),
                      out / "A3a_curves")
    plots.grid_plot(record, "A3: learning-rate grid", out / "A3a_lr_grid")

    res = {"lr_grid": record, "metrics": metrics, "params": layers.count_params(model),
           "size_table": model.size_table()}
    res["A3b"] = a3b(m, sel, blocks, model, out)
    res["A3c"] = a3c(m, sel, model, out)
    return res


def a3b(m, sel, blocks, model, out):
    """Trained filters at the A2 positions, before/after, and their maps."""
    W1 = model.blocks[0].weight.detach().numpy()
    W2 = model.blocks[1].weight.detach().numpy()
    B1 = blocks[0].weight.detach().numpy()
    B2 = blocks[1].weight.detach().numpy()
    f1, f2 = config.A3B_FILTERS_L1, config.A3B_FILTERS_L2
    plots.filter_row([W1[i, 0] for i in f1], ["filter %d" % (i + 1) for i in f1],
                     "A3b: all 6 layer-1 filters after training (sheet asks for 10; "
                     "the layer has 6)", out / "A3b_filters_l1", panel=1.7)
    plots.filter_slices([W2[i] for i in f2], ["filter %d" % (i + 1) for i in f2],
                        "A3b: 10 of 16 layer-2 filters after training, per input channel",
                        out / "A3b_filters_l2")

    a2l1 = config.A2_FILTERS_L1
    lim = max(np.abs(B1[a2l1]).max(), np.abs(W1[a2l1]).max())
    fig, axes = plots.new_fig(2, len(a2l1), panel=1.9, extra_h=0.8)
    for c, i in enumerate(a2l1):
        for r, (W, tag) in enumerate(((B1, "before"), (W1, "after"))):
            im = plots.show_signed(axes[r, c], W[i, 0], lim=lim,
                                   title="filter %d %s" % (i + 1, tag))
            plots.annotate_cells(axes[r, c], W[i, 0], lim=lim)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="weight")
    fig.suptitle("A3b: layer-1 filters at the A2 positions, untrained (A2) vs trained")
    plots.save_fig(fig, out / "A3b_before_after_l1")
    rows, labs = [], []
    for i in config.A2_FILTERS_L2:
        rows += [B2[i], W2[i]]
        labs += ["f%d before" % (i + 1), "f%d after" % (i + 1)]
    plots.filter_slices(rows, labs, "A3b: layer-2 filters at the A2 positions, "
                        "untrained (A2) vs trained", out / "A3b_before_after_l2")

    labels = ["digit %s" % m["classes"][m["train"]["y"][i]] for i in sel["A_images"]]
    imgs = m["train"]["X"][sel["A_images"]]
    with torch.no_grad():
        feats = model.features(train.mnist_tensor(imgs))
    c1, c2 = feats[0]["conv"].numpy(), feats[1]["conv"].numpy()
    plots.image_maps_grid(imgs, labels, [("f%d\n24x24" % (i + 1), c1[:, i]) for i in f1],
                          "A3b: trained layer-1 feature maps (conv+ReLU)",
                          out / "A3b_maps_l1", note="colour scale shared per column")
    plots.image_maps_grid(imgs, labels, [("f%d\n8x8" % (i + 1), c2[:, i]) for i in f2],
                          "A3b: trained layer-2 feature maps (conv+ReLU)",
                          out / "A3b_maps_l2", panel=1.2, note="colour scale shared per column")
    change = {"layer1": {str(i + 1): float(np.linalg.norm(W1[i] - B1[i]) / np.linalg.norm(B1[i]))
                         for i in f1},
              "layer2": {str(i + 1): float(np.linalg.norm(W2[i] - B2[i]) / np.linalg.norm(B2[i]))
                         for i in f2}}
    return {"filters_l1_idx_1based": [i + 1 for i in f1],
            "filters_l2_idx_1based": [i + 1 for i in f2],
            "note": "conv1 has 6 filters, so all 6 are shown instead of 10",
            "relative_weight_change": change,
            "filters_l1": W1[f1, 0], "filters_l2": W2[f2]}


def top_distinct_maps(fmap, n):
    """Top-n feature maps by peak activation, each at its argmax: (map, i, j, value)."""
    C = fmap.shape[0]
    flat = fmap.reshape(C, -1)
    peak = flat.max(1)
    out = []
    for c in np.argsort(-peak, kind="stable")[:n]:
        i, j = np.unravel_index(int(flat[c].argmax()), fmap.shape[1:])
        out.append((int(c), int(i), int(j), float(peak[c])))
    return out


def a3c(m, sel, model, out):
    """5 most activated neurons in conv2 (post-ReLU, pre-pool) per image,
    traced back through pool1 and conv1 to a 14x14 input box."""
    labels = ["digit %s" % m["classes"][m["train"]["y"][i]] for i in sel["A_images"]]
    imgs = m["train"]["X"][sel["A_images"]]
    with torch.no_grad():
        c2 = model.features(train.mnist_tensor(imgs))[1]["conv"].numpy()
    specs = model.conv_specs(1)
    rf, jump = receptive.rf_size_and_jump(specs)
    n = config.N_TRACE_NEURONS_A
    rows = []
    fig, axes = plots.new_fig(len(imgs), n, panel=1.6, extra_h=0.7)
    fig2, axes2 = plots.new_fig(1, len(imgs), panel=2.2, extra_h=1.5)
    for r, (img, lab) in enumerate(zip(imgs, labels)):
        neurons = []
        plots.show_image(axes2[0, r], img, title=lab)
        for k, (c, i, j, v) in enumerate(top_distinct_maps(c2[r], n)):
            box, clipped = receptive.neuron_box(specs, i, j, 28, 28)
            neurons.append({"map_1based": c + 1, "row_1based": i + 1, "col_1based": j + 1,
                            "activation": v, "input_box_rows_1based": [box[0] + 1, box[1] + 1],
                            "input_box_cols_1based": [box[2] + 1, box[3] + 1]})
            ax = axes[r, k]
            plots.show_image(ax, img, title="map %d (%d,%d)\nact %.2f" % (c + 1, i + 1, j + 1, v))
            plots.draw_box(ax, box, plots.BOX_COLOURS[k], lw=2)
            plots.draw_box(axes2[0, r], box, plots.BOX_COLOURS[k], lw=1.6,
                           label="map %d (%d,%d)" % (c + 1, i + 1, j + 1))
        axes[r, 0].set_ylabel(lab)
        axes2[0, r].legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=6.5,
                           frameon=False)
        rows.append({"image": lab, "neurons": neurons})
    fig.suptitle("A3c: 5 most activated conv2 neurons per image (distinct maps), "
                 "%dx%d input patch each" % (rf, rf))
    plots.save_fig(fig, out / "A3c_patches_grid")
    fig2.suptitle("A3c: all 5 patches per image")
    plots.save_fig(fig2, out / "A3c_patches_overlay")
    return {"layer": "conv2 output after ReLU, before pooling (16 x 8 x 8)",
            "selection": "top-5 distinct feature maps by peak activation, each at its argmax",
            "receptive_field": rf, "jump": jump,
            "trace": "conv2 5x5 on pool1 -> pool1 2x2/2 on conv1 -> conv1 5x5 on input; "
                     "neuron (i, j) (0-based) covers input rows/cols 2i..2i+13",
            "images": rows}


# ================================================================ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(config.OUT_DIR))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(config.TORCH_THREADS)
    out = Path(args.outdir) / "A"
    out.mkdir(parents=True, exist_ok=True)

    m = data.load_mnist()
    if args.quick:
        m["train"] = data.subset_per_class(m["train"], config.QUICK["per_class_train"])
        for s in ("val", "test"):
            m[s] = data.subset_per_class(m[s], config.QUICK["per_class_eval"])
    sel = data.select_a(m)
    log("Part A images: %s" % ", ".join(sel["A_files"]))

    res = {"quick": args.quick, "seed": config.SEED, "selection": sel,
           "split_sizes": {s: len(m[s]["y"]) for s in config.SPLITS}}
    res["A1"] = a1(m, sel, out)
    res["A2"], blocks = a2(m, sel, out)
    res["A3"] = a3(m, sel, blocks, out, args.quick)
    p = save_json(res, out / "results.json")
    log("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
