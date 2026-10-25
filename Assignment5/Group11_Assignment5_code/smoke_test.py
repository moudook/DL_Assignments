"""
smoke_test.py - correctness checks to pass on every machine before any full
run. Plain script, no pytest:

    python smoke_test.py                 # all groups
    python smoke_test.py data conv rf    # selected groups

Exit code is non-zero if any check fails.
"""

import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import config
import data
import layers
import plots
import receptive
from conv_manual import conv2d_manual, kaiming_filter, out_size

TESTS = []
OUT = config.OUT_DIR / "smoke"
_cache = {}


def test(group):
    def wrap(fn):
        TESTS.append((group, fn))
        return fn
    return wrap


def mnist():
    if "mnist" not in _cache:
        _cache["mnist"] = data.load_mnist()
    return _cache["mnist"]


def caltech():
    if "caltech" not in _cache:
        _cache["caltech"] = data.load_caltech()
    return _cache["caltech"]


def all_models():
    blocks = layers.part_a_blocks(config.SEED)
    out = {"PartA": layers.part_a_cnn(blocks, config.SEED)}
    for n in (1, 2, 3):
        out["Arch%d" % n] = layers.caltech_arch(n, config.SEED)
    return out


# ================================================================ data

@test("data")
def counts_and_class_order():
    for ds, per in ((mnist(), config.MNIST_PER_CLASS), (caltech(), config.CALTECH_PER_CLASS)):
        for split in config.SPLITS:
            y = ds[split]["y"]
            got = np.bincount(y, minlength=len(ds["classes"])).tolist()
            assert got == [per[split]] * len(ds["classes"]), (split, got)
            # label c <-> folder ds["classes"][c], for every file
            for f, c in zip(ds[split]["files"], y):
                assert f.split("/")[1] == ds["classes"][c], f
    assert mnist()["classes"] == config.MNIST_CLASSES
    assert caltech()["classes"] == config.CALTECH_CLASSES
    return "MNIST %s/class, Caltech %s/class" % (
        "/".join(str(config.MNIST_PER_CLASS[s]) for s in config.SPLITS),
        "/".join(str(config.CALTECH_PER_CLASS[s]) for s in config.SPLITS))


@test("data")
def array_formats():
    m, c = mnist(), caltech()
    for split in config.SPLITS:
        assert m[split]["X"].dtype == np.uint8 and m[split]["X"].shape[1:] == (28, 28)
        X = c[split]["X"]
        assert X.dtype == np.uint8 and X.shape[1:] == (224, 224, 3), X.shape
        g = c[split]["grey"]
        if g.any():
            Xg = X[g].astype(np.int16)
            assert (Xg[..., 0] == Xg[..., 1]).all() and (Xg[..., 1] == Xg[..., 2]).all()
    grey = {}
    for split in config.SPLITS:
        for gi, yi in zip(c[split]["grey"], c[split]["y"]):
            grey[c["classes"][yi]] = grey.get(c["classes"][yi], 0) + int(gi)
    return "greyscale per class: %s" % grey


@test("data")
def no_split_leakage():
    msgs = []
    for name, ds in (("MNIST", mnist()), ("Caltech", caltech())):
        seen = {}
        for split in config.SPLITS:
            for f, h in data.file_hashes(ds["root"], ds[split]["files"]).items():
                seen.setdefault(h, []).append(split)
        cross = [s for s in seen.values() if len(set(s)) > 1]
        assert not cross, "%s: %d identical files across splits" % (name, len(cross))
        within = sum(1 for s in seen.values() if len(s) > 1)
        msgs.append("%s 0 cross-split, %d within-split duplicates" % (name, within))
    return "; ".join(msgs)


@test("data")
def selections_fixed():
    m, c = mnist(), caltech()
    sel = data.selections(m, c)
    assert sorted(m["train"]["y"][sel["A_images"]].tolist()) == list(range(5))
    assert sorted(c["train"]["y"][sel["B_images"]].tolist()) == list(range(5))
    assert c["classes"][c["train"]["y"][sel["B_single"]]] == config.SINGLE_IMAGE_CLASS_B
    assert not c["train"]["grey"][sel["B_single"]], "B1c image must be colour"
    data.write_selection(sel, OUT / "selection.json")
    return "A: %s | B single: %s" % (", ".join(sel["A_files"]), sel["B_single_file"])


# ================================================================ conv & sizes

@test("conv")
def manual_conv_matches_framework():
    rng = np.random.default_rng(0)
    cases = [((28, 28), 5, 1, 0), ((13, 17), 3, 2, 1), ((224, 224), 11, 4, 0)]
    for (h, w), k, s, p in cases:
        img, ker = rng.normal(size=(h, w)), rng.normal(size=(k, k))
        ours = conv2d_manual(img, ker, stride=s, pad=p)
        ref = F.conv2d(torch.from_numpy(img)[None, None], torch.from_numpy(ker)[None, None],
                       stride=s, padding=p)[0, 0].numpy()
        assert ours.shape == (out_size(h, k, p, s), out_size(w, k, p, s)) == ref.shape
        assert np.allclose(ours, ref, atol=1e-9), np.abs(ours - ref).max()
    m = mnist()
    sel = data.selections(m, caltech())
    ker = kaiming_filter((5, 5), 25, np.random.default_rng(config.SEED))
    for i in sel["A_images"]:
        img = m["train"]["X"][i] / 255.0
        ours = conv2d_manual(img, ker)
        ref = F.conv2d(torch.from_numpy(img)[None, None], torch.from_numpy(ker)[None, None])
        assert ours.shape == (24, 24)
        assert np.allclose(ours, ref[0, 0].numpy(), atol=1e-9)
    return "loops == F.conv2d on 3 random cases (incl. 11x11 s4) and the 5 A1 images"


EXPECTED_SHAPES = {
    "PartA": [(6, 24), (6, 12), (16, 8), (16, 4)],
    "Arch1": [(8, 54), (8, 26), (16, 22), (16, 10)],
    "Arch2": [(8, 54), (8, 26), (16, 22), (16, 10), (32, 8), (32, 3)],
    "Arch3": [(8, 54), (8, 26), (16, 22), (16, 10), (32, 8), (64, 6), (64, 2)],
}
EXPECTED_PARAMS = {"PartA": 19345, "Arch1": 211701, "Arch2": 48405, "Arch3": 62805}
EXPECTED_RF = {"PartA": (14, 2), "Arch1": (51, 8), "Arch2": (99, 16), "Arch3": (131, 16)}


@test("conv")
def layer_sizes_formula_and_actual():
    for name, model in all_models().items():
        x = torch.zeros(1, model.in_ch, model.in_hw, model.in_hw)
        actual = []
        for f, b in zip(model.features(x), model.blocks):
            actual.append(tuple(f["conv"].shape[1:]))
            if b.pool:
                actual.append(tuple(f["out"].shape[1:]))
        formula = [(c, h, w) for n, c, h, w in model.size_table()
                   if n.startswith(("conv", "pool"))]
        hand = [(c, s, s) for c, s in EXPECTED_SHAPES[name]]
        assert actual == formula == hand, (name, actual, formula, hand)
        assert model(x).shape == (1, 5)
    return "every conv/pool stage of PartA, Arch1-3 matches formula and hand values"


@test("conv")
def parameter_counts():
    got = {k: layers.count_params(m) for k, m in all_models().items()}
    assert got == EXPECTED_PARAMS, got
    return ", ".join("%s %d" % kv for kv in got.items())


# ================================================================ init

def _std_ok(w, expected):
    n = w.numel()
    tol = 5.0 / np.sqrt(2 * n)          # 5 sigma of the sample-std estimator
    return abs(w.std().item() / expected - 1) < tol


@test("init")
def kaiming_and_xavier():
    torch.manual_seed(0)
    big = layers.ConvReLUPool(64, 64, 5)
    assert _std_ok(big.weight.detach(), np.sqrt(2 / (64 * 25)))
    for name, model in all_models().items():
        for b in model.blocks:
            fan_in = b.in_ch * b.k * b.k
            assert _std_ok(b.weight.detach(), np.sqrt(2 / fan_in)), (name, b)
            assert (b.bias == 0).all()
        for fc in (model.fc1, model.fc2):
            exp = np.sqrt(2 / (fc.in_features + fc.out_features))
            assert _std_ok(fc.weight.detach(), exp), (name, fc)
            assert (fc.bias == 0).all()
    return "conv std = sqrt(2/fan_in), FC std = sqrt(2/(fan_in+fan_out)), biases 0"


@test("init")
def shared_initial_weights():
    blocks = layers.part_a_blocks(config.SEED)
    a3 = layers.part_a_cnn(blocks, config.SEED)
    for b2, b3 in zip(blocks, a3.blocks):
        assert torch.equal(b2.weight, b3.weight) and b2.weight is not b3.weight
    archs = [layers.caltech_arch(n, config.SEED) for n in (1, 2, 3)]
    for i in (0, 1):
        for other in archs[1:]:
            assert torch.equal(archs[0].blocks[i].weight, other.blocks[i].weight)
    again = layers.part_a_blocks(config.SEED)
    assert torch.equal(again[0].weight, blocks[0].weight), "seeding not deterministic"
    return "A3 starts from A2's weights (copied); Arch1-3 share conv1/conv2 init"


# ================================================================ receptive field

@test("rf")
def notes_toy_example():
    # Slide 124: 4x4 input, 2x2 filter, stride 2; h4 sees pixels 11, 12, 15, 16.
    r0, r1, c0, c1 = receptive.trace_box([(2, 2, 0)], 1, 1)
    pixels = sorted(r * 4 + c + 1 for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    assert pixels == [11, 12, 15, 16], pixels
    return "h4 -> pixels %s" % pixels


@test("rf")
def box_equals_gradient_support():
    rng = np.random.default_rng(0)
    for name, model in all_models().items():
        model = model.double()
        model.set_probe(True)
        last = len(model.blocks) - 1
        specs = model.conv_specs(last)
        assert receptive.rf_size_and_jump(specs) == EXPECTED_RF[name], \
            (name, receptive.rf_size_and_jump(specs))
        for bi in range(len(model.blocks)):
            sp = model.conv_specs(bi)
            x = torch.randn(1, model.in_ch, model.in_hw, model.in_hw,
                            dtype=torch.float64, requires_grad=True)
            fmap = model.features(x)[bi]["conv"]
            _, C, H, W = fmap.shape
            for _ in range(3):
                c, i, j = rng.integers(C), rng.integers(H), rng.integers(W)
                g, = torch.autograd.grad(fmap[0, c, i, j], x, retain_graph=True)
                nz = (g[0].abs().sum(0) > 0).nonzero()
                support = (nz[:, 0].min().item(), nz[:, 0].max().item(),
                           nz[:, 1].min().item(), nz[:, 1].max().item())
                box, _ = receptive.neuron_box(sp, int(i), int(j), model.in_hw, model.in_hw)
                assert support == box, (name, bi, (c, i, j), support, box)
                h, w = box[1] - box[0] + 1, box[3] - box[2] + 1
                assert len(nz) == h * w, "support is not a full rectangle"
    return "traced box == exact gradient support; RF %s" % EXPECTED_RF


# ================================================================ plots

@test("plots")
def lint_catches_problems():
    import matplotlib.pyplot as plt
    fig, axes = plots.new_fig(2, 2)
    for ax in axes.flat:
        plots.show_signed(ax, np.random.default_rng(0).normal(size=(5, 5)), title="ok")
    assert not plots.lint(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_title("a title far too long to possibly fit inside a two inch figure")
    ax.text(0.5, 0.5, "A", transform=ax.transAxes)
    ax.text(0.5, 0.5, "B", transform=ax.transAxes)
    probs = plots.lint(fig)
    plt.close(fig)
    assert any("outside" in p for p in probs) and any("overlapping" in p for p in probs), probs
    return "clean figure passes; clipped title and overlapping text are caught"


@test("plots")
def standard_figures_pass_lint():
    rng = np.random.default_rng(1)
    cm = rng.integers(0, 20, size=(5, 5))
    p1 = plots.confusion_matrix_plot(cm, config.CALTECH_CLASSES, "Smoke test", OUT / "cm_caltech")
    p2 = plots.confusion_matrix_plot(cm, config.MNIST_CLASSES, "Smoke test", OUT / "cm_mnist")

    fig, axes = plots.new_fig(1, 3, panel=2.2)
    k = rng.normal(size=(5, 5))
    for ax in axes.flat:
        plots.show_signed(ax, k, title="Filter 1 (5x5)")
        plots.annotate_cells(ax, k)
    p3 = plots.save_fig(fig, OUT / "annotated_filter")

    m = mnist()
    fig, axes = plots.new_fig(1, 2, panel=2.2)
    plots.show_image(axes[0, 0], m["train"]["X"][0], title="digit")
    plots.show_image(axes[0, 1], caltech()["train"]["X"][0], title="bonsai")
    plots.draw_box(axes[0, 0], (2, 15, 4, 17), plots.BOX_COLOURS[0])
    p4 = plots.save_fig(fig, OUT / "image_box")
    return "4 figures written to %s" % OUT.relative_to(config.CODE_DIR).as_posix()


# ================================================================ training

@test("train")
def softmax_ce_wiring():
    import math
    for name, model in all_models().items():
        torch.nn.init.zeros_(model.fc2.weight)
        torch.nn.init.zeros_(model.fc2.bias)
        x = torch.randn(4, model.in_ch, model.in_hw, model.in_hw)
        y = torch.arange(4) % 5
        loss = F.cross_entropy(model(x), y).item()
        assert abs(loss - math.log(5)) < 1e-6, (name, loss)
    z = torch.randn(6, 5)
    y = torch.arange(6) % 5
    manual = -torch.log_softmax(z, 1)[torch.arange(6), y].mean()
    assert torch.allclose(F.cross_entropy(z, y), manual)
    return "zero output layer -> loss = ln 5 exactly; CE == -log softmax"


@test("train")
def every_parameter_gets_gradient():
    for name, model in all_models().items():
        x = torch.randn(2, model.in_ch, model.in_hw, model.in_hw)
        F.cross_entropy(model(x), torch.tensor([0, 1])).backward()
        for pn, p in model.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, (name, pn)
    return "non-zero gradient reaches every weight and bias"


@test("train")
def overfit_tiny_subset():
    import train
    m, c = mnist(), caltech()
    msgs = []
    sm = data.subset_per_class(m["train"], 4)
    X, y = train.mnist_tensor(sm["X"]), torch.from_numpy(sm["y"])
    sc = data.subset_per_class(c["train"], 4)
    mean, std = train.caltech_stats(c["train"]["X"])
    Xc, yc = train.caltech_tensor(sc["X"], mean, std), torch.from_numpy(sc["y"])
    cases = [("PartA", lambda: layers.part_a_cnn(layers.part_a_blocks(0), 0), X, y)]
    cases += [("Arch%d" % n, (lambda n=n: layers.caltech_arch(n, 0)), Xc, yc) for n in (1, 2, 3)]
    for name, build, Xs, ys in cases:
        model = build()
        # lr 0.003: at 0.01+ (batch 4) the Caltech nets collapse to chance
        # (loss ln 5, dead ReLUs) - a step-size effect, not a wiring fault.
        hist, info = train.fit(model, Xs, ys, Xs, ys, lr=0.003, batch_size=4,
                               max_epochs=150, seed=0, patience=150, log=lambda *_: None)
        _, acc, _ = train.loss_acc(model, Xs, ys)
        assert acc == 1.0, (name, acc)
        msgs.append("%s %d ep" % (name, info["best_epoch"]))
    return "20 images fitted to 100%%: %s" % ", ".join(msgs)


@test("train")
def early_stopping_rule():
    from protocol import EarlyStopper
    s = EarlyStopper(patience=3)
    # minimum 2.9 at epoch 5, three non-improving epochs (6, 7, 8) -> stop at 8;
    # the later 1.0 is never reached
    losses = [5, 4, 3, 3.5, 3.2, 2.9, 3.0, 3.0, 3.0, 1.0]
    stops = []
    for e, v in enumerate(losses):
        imp, stop = s.step(e, v)
        if stop:
            stops.append(e)
            break
    assert stops == [8] and s.best_epoch == 5, (stops, s.best_epoch)
    s = EarlyStopper(patience=3)
    assert not any(s.step(e, 10 - e)[1] for e in range(10)), "stopped while improving"
    s = EarlyStopper(patience=3)
    s.step(0, 1.0)
    assert s.step(1, float("nan")) == (False, True) and s.diverged
    return "stops patience epochs after the minimum; never while improving; NaN stops"


@test("train")
def fit_restores_best_weights():
    import train
    sm = data.subset_per_class(mnist()["train"], 6)
    sv = data.subset_per_class(mnist()["val"], 6)
    X, y = train.mnist_tensor(sm["X"]), torch.from_numpy(sm["y"])
    Xv, yv = train.mnist_tensor(sv["X"]), torch.from_numpy(sv["y"])
    model = layers.part_a_cnn(layers.part_a_blocks(0), 0)
    hist, info = train.fit(model, X, y, Xv, yv, lr=0.1, batch_size=4, max_epochs=40,
                           seed=0, patience=4, log=lambda *_: None)
    vl, _, _ = train.loss_acc(model, Xv, yv)
    assert abs(vl - info["best_val_loss"]) < 1e-6, (vl, info["best_val_loss"])
    assert hist["val_loss"][info["best_epoch"]] == info["best_val_loss"]
    assert info["stop_epoch"] - info["best_epoch"] == 4 or info["hit_epoch_cap"]
    return "after stop, model val loss == best epoch's (%d of %d)" % (
        info["best_epoch"], info["stop_epoch"])


# ================================================================ TensorFlow (own process)

@test("vgg")
def tensorflow_checks():
    import subprocess
    r = subprocess.run([sys.executable, str(config.CODE_DIR / "smoke_test_tf.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [l for l in r.stdout.splitlines() if l.startswith("[")]
    print("\n".join("        " + l for l in lines))
    if r.returncode != 0:
        raise AssertionError("TF checks failed:\n" + r.stdout[-3000:] + r.stderr[-2000:])
    return "%d TF checks passed (separate process)" % len(lines)


# ================================================================ runner

def main(groups):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.TORCH_THREADS)
    selected = [(g, fn) for g, fn in TESTS if not groups or g in groups]
    failed = 0
    for g, fn in selected:
        t0 = time.perf_counter()
        try:
            msg = fn() or ""
            status = "PASS"
        except Exception as e:  # report every failure, keep going
            failed += 1
            status = "FAIL"
            msg = "%s: %s" % (type(e).__name__, e)
            traceback.print_exc()
        print("[%s] %-6s %-34s %5.1fs  %s" % (status, g, fn.__name__,
                                              time.perf_counter() - t0, msg), flush=True)
    print("\n%d/%d passed" % (len(selected) - failed, len(selected)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
