"""
part_b2_vgg.py - Part B Q2: VGG19 pretrained on ImageNet (TensorFlow Keras).
The ONLY module that imports TensorFlow; it never imports torch.

    python part_b2_vgg.py [--outdir DIR] [--quick]

Model: keras.applications.VGG19(include_top=True); only the final 1000-way
`predictions` layer is replaced (Dense(5, softmax)) and trained, everything
else frozen (the notes: "if the dataset is small, retrain only the softmax
layer").

Training the head on cached fc2 features is the same training as running the
frozen network every step: everything below the head is frozen and
deterministic, and VGG19's top has no dropout (asserted). The trained head is
then placed on the full Keras model, and predictions of the two are checked
equal.

Keras SGD momentum (v = m v - lr g; w += v) is the same update as torch's
(v = m v + g; w -= lr v) for a constant learning rate.

B2a train/val/test + comparison with B1b, B2b most activated block5_conv4
neuron per image traced back, B2c guided backpropagation, B2d Grad-CAM.
Writes <outdir>/B2/results.json.
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

tf.get_logger().setLevel("ERROR")        # silence predict() retracing notices

import config  # noqa: E402
import data  # noqa: E402
import plots  # noqa: E402
import receptive  # noqa: E402
from evaluate import summary  # noqa: E402
from protocol import EarlyStopper, run_flags, run_grid  # noqa: E402
from results import load_json, save_json  # noqa: E402

LAST_CONV = "block5_conv4"
HEAD = "predictions_5"
MIN_SEP = 3        # B2c: min Chebyshev distance (cells) between chosen neurons


def log(msg):
    print(msg, flush=True)


# ================================================================ model

def preprocess(X_uint8):
    """The same uint8 RGB pixels B1 uses -> VGG19's own preprocessing, once."""
    return keras.applications.vgg19.preprocess_input(np.array(X_uint8, dtype=np.float32))


def load_base():
    return keras.applications.VGG19(weights="imagenet", include_top=True)


def fc2_features(base, c, split, batch=16):
    """Cached 4096-d fc2 outputs for one split (keyed by the file list)."""
    files = c[split]["files"]
    key = hashlib.sha1(("vgg19_fc2|" + "|".join(files)).encode("utf-8")).hexdigest()[:16]
    f = config.CACHE_DIR / ("vgg19_fc2_%s_%s.npy" % (split, key))
    if f.exists():
        return np.load(f)
    feat = keras.Model(base.input, base.get_layer("fc2").output)
    t0 = time.perf_counter()
    F = feat.predict(preprocess(c[split]["X"]), batch_size=batch, verbose=0)
    log("  fc2 features %s: %d images in %.0fs" % (split, len(F), time.perf_counter() - t0))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, F)
    return F


def make_head(seed):
    keras.utils.set_random_seed(seed)
    inp = keras.Input((4096,))
    out = keras.layers.Dense(5, activation="softmax", name=HEAD,
                             kernel_initializer=keras.initializers.GlorotNormal(seed=seed),
                             bias_initializer="zeros")(inp)
    return keras.Model(inp, out)


def full_model(base, head):
    """VGG19 with its `predictions` layer replaced; everything but the head frozen."""
    out = keras.layers.Dense(5, activation="softmax", name=HEAD)(base.get_layer("fc2").output)
    model = keras.Model(base.input, out)
    for layer in model.layers:
        layer.trainable = layer.name == HEAD
    model.get_layer(HEAD).set_weights(head.get_layer(HEAD).get_weights())
    return model


def trainable_params(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def check_model(model):
    """Structural facts the plan relies on; raises if any fails."""
    names = [type(l).__name__ for l in model.layers]
    assert "Dropout" not in names, "VGG19 top unexpectedly has dropout"
    assert trainable_params(model) == 4096 * 5 + 5, trainable_params(model)
    assert model.output_shape == (None, 5)
    assert model.get_layer(LAST_CONV).output.shape[1:] == (14, 14, 512)


def vgg_specs(base, upto=LAST_CONV):
    """(k, s, p) of every conv/pool from the input to `upto`."""
    specs = []
    for layer in base.layers:
        if isinstance(layer, keras.layers.Conv2D):
            k, s = layer.kernel_size[0], layer.strides[0]
            specs.append((k, s, (k - 1) // 2 if layer.padding == "same" else 0))
        elif isinstance(layer, keras.layers.MaxPooling2D):
            specs.append((layer.pool_size[0], layer.strides[0], 0))
        if layer.name == upto:
            return specs
    raise ValueError(upto)


# ================================================================ training

def fit_head(Ftr, ytr, Fva, yva, lr, max_epochs, seed, patience=config.ES_PATIENCE):
    """Same protocol as train.fit: momentum SGD, early stopping on val loss,
    best weights restored."""
    head = make_head(seed)
    head.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=config.MOMENTUM),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    rng = np.random.default_rng(seed)
    stopper = EarlyStopper(patience)
    vl, va = head.evaluate(Fva, yva, batch_size=256, verbose=0)
    hist = {"epoch": [0], "train_loss": [None], "train_acc": [None],
            "val_loss": [float(vl)], "val_acc": [float(va)]}
    stopper.step(0, float(vl))
    best_w = head.get_weights()
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(len(Ftr))
        h = head.fit(Ftr[perm], ytr[perm], batch_size=config.BATCH_SIZE_B, epochs=1,
                     shuffle=False, verbose=0)
        tl, ta = float(h.history["loss"][0]), float(h.history["accuracy"][0])
        vl, va = (float(v) for v in head.evaluate(Fva, yva, batch_size=256, verbose=0))
        for k, v in (("epoch", epoch), ("train_loss", tl), ("train_acc", ta),
                     ("val_loss", vl), ("val_acc", va)):
            hist[k].append(v)
        improved, stop = stopper.step(epoch, vl if np.isfinite(tl) else float("nan"))
        if improved:
            best_w = head.get_weights()
        if stop:
            break
    head.set_weights(best_w)
    info = run_flags(hist, stopper, max_epochs, epoch, patience)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    return head, hist, info


def head_eval(head, F, y, classes):
    p = head.predict(F, batch_size=256, verbose=0)
    # + 0.0 turns -0.0 (all probabilities exactly 1) into 0.0
    loss = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))) + 0.0
    return summary(y, p.argmax(1), classes, loss)


# ================================================================ B2b

def b2b(c, sel, base, model, out):
    """Most activated block5_conv4 neuron per image, traced back. The
    theoretical receptive field (252 px) exceeds the image, so the box is the
    clipped field; the gradient magnitude shows where the neuron actually looks."""
    conv = keras.Model(base.input, base.get_layer(LAST_CONV).output)
    imgs = c["train"]["X"][sel["B_images"]]
    xs = preprocess(imgs)
    a = conv(xs).numpy()
    g = []
    for r in range(len(imgs)):                       # gradient of THE chosen neuron
        i, j, m = np.unravel_index(int(a[r].argmax()), a[r].shape)
        x = tf.constant(xs[r:r + 1])
        with tf.GradientTape() as tape:
            tape.watch(x)
            v = conv(x)[0, i, j, m]
        g.append(tape.gradient(v, x).numpy()[0])
    specs = vgg_specs(base)
    rf, jump = receptive.rf_size_and_jump(specs)
    fig, axes = plots.new_fig(len(imgs), 3, panel=2.4, extra_h=0.8)
    rows = []
    for r in range(len(imgs)):
        i, j, m = np.unravel_index(int(a[r].argmax()), a[r].shape)
        box, clipped = receptive.neuron_box(specs, int(i), int(j), 224, 224)
        cls = c["classes"][c["train"]["y"][sel["B_images"][r]]]
        plots.show_image(axes[r, 0], imgs[r], title="map %d (%d,%d), act %.0f"
                         % (m + 1, i + 1, j + 1, a[r].max()))
        plots.draw_box(axes[r, 0], box, plots.BOX_COLOURS[0], lw=2)
        axes[r, 0].set_ylabel(cls)
        r0, r1, c0, c1 = box
        plots.show_image(axes[r, 1], imgs[r][r0:r1 + 1, c0:c1 + 1],
                         title="patch %dx%d (clipped)" % (r1 - r0 + 1, c1 - c0 + 1))
        gm = np.abs(g[r]).sum(-1)
        plots.show_positive(axes[r, 2], gm / (gm.max() or 1.0), vmax=1.0,
                            title="|d neuron / d input|")
        rows.append({"image": sel["B_files"][r], "class": cls,
                     "neuron": {"map_1based": int(m) + 1, "row_1based": int(i) + 1,
                                "col_1based": int(j) + 1, "activation": float(a[r].max())},
                     "theoretical_box_rows_1based_unclipped":
                         [receptive.trace_box(specs, int(i), int(j))[0] + 1,
                          receptive.trace_box(specs, int(i), int(j))[1] + 1],
                     "input_box_rows_1based": [r0 + 1, r1 + 1],
                     "input_box_cols_1based": [c0 + 1, c1 + 1], "clipped": clipped,
                     "gradient_support_fraction": float((gm > 1e-3 * gm.max()).mean())})
    fig.suptitle("B2b VGG19 %s: most activated neuron (14x14x512)\n"
                 "receptive field %d px > 224 px image, box clipped to the image"
                 % (LAST_CONV, rf))
    plots.save_fig(fig, out / "B2b_patches")
    return {"layer": "%s after ReLU, before pooling (14 x 14 x 512)" % LAST_CONV,
            "receptive_field": rf, "jump": jump, "images": rows}


# ================================================================ B2c

@tf.custom_gradient
def guided_relu(x):
    """Forward ReLU; backward passes gradient only where the forward input was
    positive AND the incoming gradient is positive (Springenberg et al.)."""
    def grad(dy):
        return dy * tf.cast(dy > 0, dy.dtype) * tf.cast(x > 0, dy.dtype)
    return tf.nn.relu(x), grad


def guided_model(base):
    """Copy of VGG19 up to block5_conv4 with every ReLU guided; the trained
    model is not touched."""
    conv = keras.Model(base.input, base.get_layer(LAST_CONV).output)
    clone = keras.models.clone_model(conv)
    clone.set_weights(conv.get_weights())
    for layer in clone.layers:
        if isinstance(layer, keras.layers.Conv2D):
            layer.activation = guided_relu
    return conv, clone


def to_display(g):
    """Standard gradient visualisation; BGR (VGG input order) -> RGB."""
    g = g[..., ::-1]
    g = (g - g.mean()) / (g.std() + 1e-8)
    return np.clip(g * 0.1 + 0.5, 0, 1)


def b2c(c, sel, base, out):
    """Guided backprop for 5 block5_conv4 neurons of the B1c image."""
    conv, guided = guided_model(base)
    img = c["train"]["X"][sel["B_single"]]
    x = tf.constant(preprocess(img[None]))
    a = conv(x).numpy()[0]                                  # (14, 14, 512)
    flat = a.reshape(-1, 512)
    peak = flat.max(0)
    # Distinct maps by peak activation, each at its argmax, skipping any whose
    # position is within MIN_SEP cells of one already chosen: the strongest
    # maps all peak around the same object part, and five near-identical
    # gradient images would show one thing five times (the notes' example
    # uses neurons at varied positions).
    chosen = []
    for m in np.argsort(-peak, kind="stable"):
        i, j = np.unravel_index(int(flat[:, m].argmax()), (14, 14))
        if all(max(abs(i - pi), abs(j - pj)) >= MIN_SEP for _, pi, pj in chosen):
            chosen.append((int(m), int(i), int(j)))
        if len(chosen) == config.N_GUIDED_NEURONS:
            break
    neurons, guided_imgs, plain_imgs = [], [], []
    for m, i, j in chosen:
        assert a[i, j, m] > 0, "chosen neuron must have non-zero activation"
        for mdl, store in ((guided, guided_imgs), (conv, plain_imgs)):
            with tf.GradientTape() as tape:
                tape.watch(x)
                v = mdl(x)[0, i, j, m]             # only this neuron is kept
            store.append(tape.gradient(v, x).numpy()[0])
        neurons.append({"map_1based": m + 1, "row_1based": int(i) + 1,
                        "col_1based": int(j) + 1, "activation": float(a[i, j, m])})
    n = len(chosen)
    fig, axes = plots.new_fig(2, n + 1, panel=2.0, extra_h=0.9)
    plots.show_image(axes[0, 0], img, title="input (%s)"
                     % c["classes"][c["train"]["y"][sel["B_single"]]])
    plots.clean_ax(axes[1, 0])
    axes[1, 0].axis("off")
    for k, (nr, gg, pg) in enumerate(zip(neurons, guided_imgs, plain_imgs), 1):
        t = "(%d, %d) in map %d" % (nr["row_1based"], nr["col_1based"], nr["map_1based"])
        axes[0, k].imshow(to_display(gg))
        plots.clean_ax(axes[0, k])
        axes[0, k].set_title(t)
        axes[1, k].imshow(to_display(pg))
        plots.clean_ax(axes[1, k])
    axes[0, 1].set_ylabel("guided backprop")
    axes[1, 1].set_ylabel("plain gradient")
    fig.suptitle("B2c guided backpropagation: influence of input pixels on 5 %s neurons "
                 "(ImageNet VGG19)" % LAST_CONV)
    plots.save_fig(fig, out / "B2c_guided_backprop")
    return {"image": sel["B_single_file"], "layer": LAST_CONV,
            "selection": "distinct feature maps in order of peak activation, each at its "
                         "argmax, positions at least %d cells apart (all non-zero)" % MIN_SEP,
            "neurons": neurons,
            "positive_fraction_guided": [float((g > 0).mean()) for g in guided_imgs]}


# ================================================================ B2d

def gradcam(base, head_layer, x, n_classes):
    """keras.io Grad-CAM on block5_conv4, class score = pre-softmax logit
    (computed from fc2 with the head's weights, so no model is mutated)."""
    grad_model = keras.Model(base.input, [base.get_layer(LAST_CONV).output,
                                          base.get_layer("fc2").output])
    W, b = (tf.constant(w) for w in head_layer.get_weights())
    with tf.GradientTape(persistent=True) as tape:
        conv_out, f = grad_model(x)
        logits = tf.matmul(f, W) + b
        scores = [logits[0, k] for k in range(n_classes)]   # recorded on the tape
    maps, raw = [], []
    for k in range(n_classes):
        g = tape.gradient(scores[k], conv_out)
        pooled = tf.reduce_mean(g, axis=(0, 1, 2))
        h = tf.nn.relu(tf.squeeze(conv_out[0] @ pooled[..., None], -1)).numpy()
        raw.append(float(h.max()))
        maps.append(h / h.max() if h.max() > 0 else h)
    del tape
    probs = tf.nn.softmax(logits).numpy()[0]
    return maps, raw, probs


def overlay(img, heat):
    """Heat map (14x14 in [0,1]) bilinearly resized and blended at alpha 0.4."""
    import matplotlib
    h = tf.image.resize(heat[..., None], (224, 224), method="bilinear").numpy()[..., 0]
    jet = matplotlib.colormaps["jet"](np.clip(h, 0, 1))[..., :3]
    return np.clip(0.6 * img / 255.0 + 0.4 * jet, 0, 1)


def b2d(c, sel, base, model, out):
    classes = c["classes"]
    head_layer = model.get_layer(HEAD)
    imgs = c["train"]["X"][sel["B_images"]]
    n = len(classes)
    fig, axes = plots.new_fig(len(imgs), n + 1, panel=1.75, extra_h=0.9)
    rows = []
    for r, img in enumerate(imgs):
        true = int(c["train"]["y"][sel["B_images"][r]])
        maps, raw, probs = gradcam(base, head_layer, preprocess(img[None]), n)
        pred = int(probs.argmax())
        plots.show_image(axes[r, 0], img, title="true %s\npred %s (%.2f)"
                         % (classes[true], classes[pred], probs[pred]))
        for k in range(n):
            ax = axes[r, k + 1]
            ax.imshow(overlay(img, maps[k]))
            plots.clean_ax(ax)
            ax.set_title("%s%s\np %.2f, max %.2g" % ("target: " if r == 0 else "",
                                                     classes[k], probs[k], raw[k]),
                         fontsize=7.5)
            if raw[k] == 0:
                ax.text(0.5, 0.5, "no positive\nevidence", transform=ax.transAxes,
                        ha="center", va="center", color="white", fontsize=7)
            if k == true:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_edgecolor("#e6194b")
                    s.set_linewidth(2.5)
        rows.append({"image": sel["B_files"][r], "true": classes[true],
                     "pred": classes[pred], "probs": dict(zip(classes, probs.round(4))),
                     "cam_raw_max": dict(zip(classes, raw))})
    fig.suptitle("B2d Grad-CAM (%s, VGG19 with retrained 5-class head): each image x each class\n"
                 "red frame = true class; each map scaled to its own max (raw max shown)"
                 % LAST_CONV)
    plots.save_fig(fig, out / "B2d_gradcam")
    return {"layer": LAST_CONV, "score": "pre-softmax logit (keras.io: last activation None)",
            "normalisation": "per map: relu, divide by its max (raw max reported)",
            "images": rows}


# ================================================================ B2a comparison

def comparison(vgg_metrics, outdir, out):
    p = Path(outdir) / "B1" / "results.json"
    if not p.exists():
        log("B1 results not found at %s; comparison skipped" % p)
        return {"note": "B1 results missing; run part_b1.py first"}
    b1 = load_json(p)
    best = b1["B1b"]["best"]
    arch = b1["architectures"][best]
    row_b1 = {"model": b1["B1b"]["name"] + " (from scratch)",
              "train": arch["metrics"]["train"]["accuracy"],
              "val": arch["metrics"]["val"]["accuracy"],
              "test": b1["B1b"]["test"]["accuracy"],
              "test_per_class_recall": b1["B1b"]["test"]["per_class_recall"],
              "trainable_params": arch["params"], "lr": arch["lr_grid"]["winner_lr"]}
    row_vgg = {"model": "VGG19 ImageNet, head retrained",
               "train": vgg_metrics["train"]["accuracy"], "val": vgg_metrics["val"]["accuracy"],
               "test": vgg_metrics["test"]["accuracy"],
               "test_per_class_recall": vgg_metrics["test"]["per_class_recall"],
               "trainable_params": 4096 * 5 + 5}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    xs = np.arange(3)
    for k, (row, col) in enumerate(((row_b1, "#4363d8"), (row_vgg, "#e6194b"))):
        vals = [row[s] for s in ("train", "val", "test")]
        bars = ax.bar(xs + (k - 0.5) * 0.38, vals, 0.36, color=col, label=row["model"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, "%.0f%%" % (100 * v),
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs, ["train", "validation", "test"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.set_title("B2a: features learned from scratch vs fixed ImageNet features")
    plots.save_fig(fig, out / "B2a_comparison")
    return {"best_B1": row_b1, "vgg19": row_vgg,
            "framing": "fixed ImageNet-pretrained features vs features learned from "
                       "scratch on 250 images; not a pure architecture comparison"}


# ================================================================ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(config.OUT_DIR))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.outdir) / "B2"
    out.mkdir(parents=True, exist_ok=True)

    c = data.load_caltech()
    if args.quick:
        c["train"] = data.subset_per_class(c["train"], config.QUICK["per_class_train"])
        for s in ("val", "test"):
            c[s] = data.subset_per_class(c[s], config.QUICK["per_class_eval"])
    sel = data.select_b(c)
    max_ep = config.QUICK["max_epochs"] if args.quick else config.MAX_EPOCHS_B

    base = load_base()
    ref_conv = base.get_layer("block1_conv1").get_weights()[0].copy()
    F = {s: fc2_features(base, c, s) for s in config.SPLITS}
    y = {s: c[s]["y"] for s in config.SPLITS}

    def run_one(lr):
        head, hist, info = fit_head(F["train"], y["train"], F["val"], y["val"], lr,
                                    max_ep, config.SEED)
        tr = head_eval(head, F["train"], y["train"], c["classes"])
        info.update({"train_loss_at_best": tr["loss"], "train_acc_at_best": tr["accuracy"],
                     "val_acc_at_best": hist["val_acc"][info["best_epoch"]]})
        log("  VGG head lr %g: best ep %s val loss %.4f val acc %.3f"
            % (lr, info["best_epoch"], info["best_val_loss"], info["val_acc_at_best"]))
        return info, hist, head

    record, head, hist = run_grid(run_one, "VGG19 head", log)
    model = full_model(base, head)
    check_model(model)
    assert np.array_equal(base.get_layer("block1_conv1").get_weights()[0], ref_conv)

    # the full model and the cached-feature head must agree
    Xs = preprocess(c["train"]["X"][sel["B_images"]])
    p_full = model.predict(Xs, verbose=0)
    p_head = head.predict(F["train"][sel["B_images"]], verbose=0)
    assert np.allclose(p_full, p_head, atol=1e-5), np.abs(p_full - p_head).max()
    # the base is stock ImageNet VGG19, so the head's weights are the whole model
    W, b = model.get_layer(HEAD).get_weights()
    np.savez(out / "B2_head_weights.npz", kernel=W, bias=b)

    metrics = {s: head_eval(head, F[s], y[s], c["classes"]) for s in config.SPLITS}
    for s in config.SPLITS:
        plots.confusion_matrix_plot(metrics[s]["confusion_matrix"], c["classes"],
                                    "VGG19 (head retrained) - %s" % s, out / ("B2a_cm_%s" % s))
        log("VGG19 %s accuracy %.3f" % (s, metrics[s]["accuracy"]))
    winner = next(r for r in record["runs"] if r["lr"] == record["winner_lr"])
    plots.curves_plot(hist, winner["best_epoch"], "VGG19 head (lr %g, batch %d, momentum %.1f)"
                      % (record["winner_lr"], config.BATCH_SIZE_B, config.MOMENTUM),
                      out / "B2a_curves")
    plots.grid_plot(record, "VGG19 head: learning-rate grid", out / "B2a_lr_grid")

    res = {"quick": args.quick, "seed": config.SEED, "selection": sel,
           "model": {"base": "keras.applications.VGG19(weights='imagenet', include_top=True)",
                     "replaced": "predictions (1000) -> Dense(5, softmax)",
                     "trainable_params": trainable_params(model),
                     "total_params": int(model.count_params()),
                     "preprocessing": "keras.applications.vgg19.preprocess_input on the "
                                      "same resized 224x224 RGB pixels as B1",
                     "training": "head trained on cached fc2 features (frozen, no dropout); "
                                 "full-model predictions verified equal"},
           "lr_grid": record, "B2a": {"metrics": metrics,
                                      "comparison": comparison(metrics, args.outdir, out)}}
    log("B2b ...")
    res["B2b"] = b2b(c, sel, base, model, out)
    log("B2c ...")
    res["B2c"] = b2c(c, sel, base, out)
    log("B2d ...")
    res["B2d"] = b2d(c, sel, base, model, out)
    p = save_json(res, out / "results.json")
    log("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
