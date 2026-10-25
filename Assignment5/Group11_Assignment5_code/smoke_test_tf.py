"""
smoke_test_tf.py - checks for the VGG19 part. Runs in its own process
(launched by smoke_test.py's "vgg" group) because TensorFlow and torch never
share a process. Exit code non-zero on any failure.
"""

import os
import sys
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import config  # noqa: E402
import data  # noqa: E402
import part_b2_vgg as V  # noqa: E402
import receptive  # noqa: E402

TESTS = []
_s = {}


def test(fn):
    TESTS.append(fn)
    return fn


def base():
    if "base" not in _s:
        _s["base"] = V.load_base()
    return _s["base"]


def imgs():
    if "imgs" not in _s:
        c = data.load_caltech()
        _s["imgs"] = c["train"]["X"][data.select_b(c)["B_images"]]
    return _s["imgs"]


@test
def preprocessing_applied_once():
    x = imgs()[:2]
    ours = V.preprocess(x)
    manual = x[..., ::-1].astype(np.float32) - np.array([103.939, 116.779, 123.68], np.float32)
    assert np.allclose(ours, manual, atol=1e-4), np.abs(ours - manual).max()
    assert x.dtype == np.uint8 and ours.dtype == np.float32
    return "RGB->BGR minus ImageNet means, exactly once"


@test
def head_swap_structure():
    b = base()
    ref = b.get_layer("block5_conv4").get_weights()[0].copy()
    head = V.make_head(0)
    model = V.full_model(b, head)
    V.check_model(model)
    assert np.array_equal(b.get_layer("block5_conv4").get_weights()[0], ref)
    assert [l.name for l in model.layers][-2:] == ["fc2", V.HEAD]
    return "5 outputs, trainable 20,485, no dropout, block5_conv4 14x14x512"


@test
def cached_features_equal_full_model():
    b = base()
    head = V.make_head(1)
    model = V.full_model(b, head)
    x = V.preprocess(imgs()[:4])
    feat = tf.keras.Model(b.input, b.get_layer("fc2").output).predict(x, verbose=0)
    p_head = head.predict(feat, verbose=0)
    p_full = model.predict(x, verbose=0)
    assert np.allclose(p_head, p_full, atol=1e-5), np.abs(p_head - p_full).max()
    return "head(fc2 features) == full model, max diff %.1e" % np.abs(p_head - p_full).max()


@test
def head_early_stopping_restores():
    rng = np.random.default_rng(0)
    Ftr = rng.normal(size=(40, 4096)).astype(np.float32)
    ytr = rng.integers(0, 5, 40)
    Fva = rng.normal(size=(20, 4096)).astype(np.float32)
    yva = rng.integers(0, 5, 20)
    head, hist, info = V.fit_head(Ftr, ytr, Fva, yva, lr=0.01, max_epochs=60, seed=0, patience=4)
    vl, _ = head.evaluate(Fva, yva, verbose=0)
    assert abs(vl - info["best_val_loss"]) < 1e-5, (vl, info["best_val_loss"])
    return "restored head val loss == best epoch's (%d of %d)" % (info["best_epoch"],
                                                                   info["stop_epoch"])


@test
def vgg_receptive_field():
    specs = V.vgg_specs(base())
    rf, jump = receptive.rf_size_and_jump(specs)
    assert (rf, jump) == (252, 16), (rf, jump)
    box, clipped = receptive.neuron_box(specs, 7, 7, 224, 224)
    assert clipped
    return "block5_conv4 RF 252 px, jump 16 (> 224 image -> clipped)"


@test
def guided_relu_rule():
    x = tf.constant([-1.0, 2.0, 3.0, -4.0])
    up = tf.constant([1.0, -1.0, 2.0, 5.0])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = V.guided_relu(x)
        s = tf.reduce_sum(y * up)
    g = tape.gradient(s, x).numpy()
    assert np.array_equal(g, [0.0, 0.0, 2.0, 0.0]), g
    return "gradient kept only where input > 0 AND upstream > 0"


@test
def guided_backprop_output():
    conv, guided = V.guided_model(base())
    x = tf.constant(V.preprocess(imgs()[:1]))
    a = conv(x).numpy()[0]
    i, j, m = np.unravel_index(int(a.argmax()), a.shape)
    gs = []
    for mdl in (guided, conv):
        with tf.GradientTape() as tape:
            tape.watch(x)
            v = mdl(x)[0, i, j, m]
        gs.append(tape.gradient(v, x).numpy()[0])
    assert gs[0].shape == (224, 224, 3) and np.abs(gs[0]).sum() > 0
    assert not np.allclose(gs[0], gs[1])
    assert np.allclose(guided(x).numpy(), a, atol=1e-4), "guided forward must equal ReLU forward"
    return "224x224x3, non-zero, differs from plain gradient, same forward"


@test
def gradcam_maps():
    b = base()
    model = V.full_model(b, V.make_head(2))
    x = V.preprocess(imgs()[:1])
    before = model.predict(x, verbose=0)
    maps, raw, probs = V.gradcam(b, model.get_layer(V.HEAD), x, 5)
    after = model.predict(x, verbose=0)
    assert all(mp.shape == (14, 14) and (mp >= 0).all() for mp in maps)
    assert np.allclose(before, after) and np.allclose(probs, before[0], atol=1e-5)
    assert all(r == 0 or abs(mp.max() - 1) < 1e-6 for mp, r in zip(maps, raw))
    return "5 maps 14x14, >= 0, model unchanged, logit-based probs match"


def main():
    failed = 0
    for fn in TESTS:
        t0 = time.perf_counter()
        try:
            msg, status = fn() or "", "PASS"
        except Exception as e:
            failed += 1
            status, msg = "FAIL", "%s: %s" % (type(e).__name__, e)
            traceback.print_exc()
        print("[%s] vgg    %-34s %5.1fs  %s" % (status, fn.__name__,
                                                time.perf_counter() - t0, msg), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
