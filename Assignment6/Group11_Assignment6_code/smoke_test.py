"""Correctness checks. Run with:  python run.py smoke   (or python smoke_test.py)

Each check prints PASS or FAIL. Exit code 0 only if every check passes.
"""
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

import config
import plots
from data import (Sample, SeqDataset, Standardiser, collate, load_split, make_loader, minmax_joint,
                  minmax_per_axis, read_cv_file, read_hw_file, stratified_split)
from models import SeqClassifier
from train import (compute_metrics, evaluate, has_converged, load_checkpoint, make_optimizer, save_checkpoint,
                   train_until_converged)

CPU = torch.device("cpu")


def check(condition, message="check failed"):
    """Like assert, but not removed when Python runs with -O."""
    if not condition:
        raise AssertionError(message)


def check_mapping_parser():
    hw_text = "Group 1: ai, bA, chA, dA, tA\nGroup 11: a, ai, chA, dA, tA\nGroup 12: ai, bA\n"
    cv_text = "Each group has to take 5 classes as given below.\nGroup-1 ba, ni\nGroup-11 ba, pa, bhA, paa, hai \n"
    check(config.parse_group_classes(hw_text) == ["a", "ai", "chA", "dA", "tA"])
    check(config.parse_group_classes(cv_text) == ["ba", "bhA", "hai", "pa", "paa"])


def check_readers(tmp):
    good = tmp / "good.txt"
    good.write_text("3 0.1 0.2 0.3 0.4 0.5 0.6", encoding="utf-8")
    points = read_hw_file(good)
    check(points.shape == (3, 2) and points[1, 0] == 0.3 and points[1, 1] == 0.4)
    bad = tmp / "bad.txt"
    bad.write_text("3 0.1 0.2 0.3", encoding="utf-8")
    try:
        read_hw_file(bad)
        raise AssertionError("header/count mismatch was not detected")
    except ValueError:
        pass
    mfcc = tmp / "x.mfcc"
    np.savetxt(mfcc, np.random.randn(5, 39))
    check(read_cv_file(mfcc).shape == (5, 39))
    np.savetxt(mfcc, np.random.randn(5, 38))
    try:
        read_cv_file(mfcc)
        raise AssertionError("wrong column count was not detected")
    except ValueError:
        pass


def check_normalisation():
    points = np.random.default_rng(0).uniform(-3, 7, size=(50, 2))
    out = minmax_per_axis(points)
    check(np.allclose(out.min(axis=0), 0) and np.allclose(out.max(axis=0), 1))
    vertical = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 3.0]])
    out = minmax_per_axis(vertical)
    check(np.all(out[:, 0] == 0.5) and np.isfinite(out).all())
    joint = minmax_joint(points)
    check(np.isclose(joint.max(), 1) and np.isclose(joint.min(axis=0), 0).all())


def check_padding_is_ignored():
    """A sequence must get the same output alone and inside a batch with a much longer sequence."""
    torch.manual_seed(0)
    short = torch.randn(5, 3)
    long = torch.randn(15, 3)
    for cell in ("rnn", "lstm"):
        for dense, sizes in (([], [8]), ([], [8, 6, 4]), ([7], [8]), ([7, 5], [8, 6])):
            model = SeqClassifier(cell, 3, sizes, 5, dense).eval()
            with torch.no_grad():
                alone = model(short.unsqueeze(0), torch.tensor([5]))
                x, lengths, _ = collate([(short, 0), (long, 1)])
                x[0, 5:] = 99.0  # garbage in the padding must not matter
                batched = model(x, lengths)
            check(torch.allclose(alone[0], batched[0], atol=1e-5), f"{cell} {dense} {sizes}")


def manual_forward(model, x):
    """The slides' equations written out step by step, using the model's own weights."""
    a = []
    for x_t in x:  # hidden layers at each time step: a_t = tanh(D x_t + e), same D, e at every step
        for dense in model.dense_layers:
            x_t = torch.tanh(dense.weight @ x_t + dense.bias)
        a.append(x_t)
    sequence = torch.stack(a)
    for layer in model.layers:
        U = layer.weight_ih_l0
        W = layer.weight_hh_l0
        b = layer.bias_ih_l0 + layer.bias_hh_l0
        hidden = W.shape[1]
        s = torch.zeros(hidden)  # s_0 = 0
        h = torch.zeros(hidden)
        outputs = []
        for x_t in sequence:
            if model.cell == "rnn":
                s = torch.tanh(U @ x_t + W @ s + b)  # s_t = tanh(U x_t + W s_{t-1} + b)
                outputs.append(s)
            else:
                # PyTorch stores the four LSTM blocks in the order: input, forget, candidate, output
                a = U @ x_t + W @ h + b
                i_t = torch.sigmoid(a[0:hidden])
                f_t = torch.sigmoid(a[hidden:2 * hidden])
                s_tilde = torch.tanh(a[2 * hidden:3 * hidden])
                o_t = torch.sigmoid(a[3 * hidden:4 * hidden])
                s = f_t * s + i_t * s_tilde          # selective forget + selective read
                h = o_t * torch.tanh(s)              # selective write
                outputs.append(h)
        sequence = torch.stack(outputs)  # states of this layer are the inputs of the next
    return model.output(sequence[-1])    # y = V s_T + c (softmax applied in the loss)


def check_matches_slide_equations():
    torch.manual_seed(1)
    x = torch.randn(9, 4)
    for cell in ("rnn", "lstm"):
        for dense, sizes in (([], [6]), ([], [6, 5]), ([7], [6]), ([7, 3], [6, 5])):
            model = SeqClassifier(cell, 4, sizes, 5, dense).eval()
            with torch.no_grad():
                expected = manual_forward(model, x)
                got = model(x.unsqueeze(0), torch.tensor([9]))[0]
            check(torch.allclose(expected, got, atol=1e-5), f"{cell} {dense} {sizes}")


def check_standardiser():
    rng = np.random.default_rng(0)
    samples = [Sample(rng.normal(5, 3, size=(rng.integers(3, 10), 4)), 0, Path("x")) for _ in range(30)]
    standardiser = Standardiser("zscore_train").fit(samples)
    frames = np.concatenate([standardiser.transform(s.x) for s in samples])
    check(np.allclose(frames.mean(axis=0), 0, atol=1e-9) and np.allclose(frames.std(axis=0), 1))
    copy = Standardiser.from_state_dict(standardiser.state_dict())
    check(np.allclose(copy.transform(samples[0].x), standardiser.transform(samples[0].x)))
    check(np.array_equal(Standardiser("none").fit(samples).transform(samples[0].x), samples[0].x))


def check_split():
    samples = [Sample(np.zeros((2, 2)), label, Path(f"{label}_{i}")) for label in range(3) for i in range(10)]
    train, val = stratified_split(samples, 0.2, seed=11)
    check(len(val) == 6 and len(train) == 24)
    check(all(sum(s.label == k for s in val) == 2 for k in range(3)))
    check({s.path for s in train}.isdisjoint({s.path for s in val}))
    train2, val2 = stratified_split(samples, 0.2, seed=11)
    check([s.path for s in val] == [s.path for s in val2])


def toy_dataset(n=20, seed=0):
    """Class 0: values around +1, class 1: values around -1; lengths 5..20."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        label = i % 2
        length = int(rng.integers(5, 21))
        x = rng.normal(1.0 if label == 0 else -1.0, 0.5, size=(length, 2))
        samples.append(Sample(x, label, Path(f"toy_{i}")))
    return SeqDataset(samples, Standardiser("none"))


def check_convergence_logic():
    check(has_converged(1.0, 1.00005, 1e-4))
    check(not has_converged(1.0, 1.0002, 1e-4))
    loader = make_loader(toy_dataset(), 8, shuffle=True)
    model = SeqClassifier("rnn", 2, [4], 2)
    optimizer = make_optimizer("adam", model.parameters(), 1e-3)
    history = train_until_converged(model, loader, optimizer, CPU, tol=0.0, max_epochs=3)
    check(history.stop_reason == "cap" and len(history.epoch_error) == 3)
    history = train_until_converged(model, loader, optimizer, CPU, tol=1e9, max_epochs=50)
    check(history.stop_reason == "converged" and len(history.epoch_error) == 2)


def check_learning():
    """Both cell types must be able to fit a small, easy training set."""
    dataset = toy_dataset()
    for cell in ("rnn", "lstm"):
        torch.manual_seed(0)
        model = SeqClassifier(cell, 2, [16], 2)
        optimizer = make_optimizer("adam", model.parameters(), 1e-2)
        train_until_converged(model, make_loader(dataset, 8, shuffle=True), optimizer, CPU,
                              tol=1e-5, max_epochs=300)
        result = evaluate(model, make_loader(dataset, 64, shuffle=False), CPU)
        accuracy = compute_metrics(result["y_true"], result["y_pred"], 2)["accuracy"]
        check(accuracy == 1.0, f"{cell} train accuracy {accuracy}")


def check_checkpoint(tmp):
    dataset = toy_dataset()
    model = SeqClassifier("lstm", 2, [8, 4], 2, [6]).eval()
    info = {"cell": "lstm", "input_dim": 2, "dense_sizes": [6], "hidden_sizes": [8, 4], "classes": ["p", "q"],
            "standardiser": Standardiser("none").state_dict(), "hw_norm": "minmax_axis"}
    save_checkpoint(tmp / "m" / "model.pt", model, info)
    loaded, loaded_info = load_checkpoint(tmp / "m" / "model.pt", CPU)
    loader = make_loader(dataset, 64, shuffle=False)
    check(evaluate(model, loader, CPU)["y_pred"] == evaluate(loaded, loader, CPU)["y_pred"])
    check(loaded_info["classes"] == ["p", "q"])


def check_metrics():
    metrics = compute_metrics([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
    check(metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]])
    check(np.isclose(metrics["accuracy"], 0.6))
    check(np.allclose(metrics["per_class_recall"], [0.5, 1.0, 0.0]))


def check_plots(tmp):
    rng = np.random.default_rng(0)
    errors = list(np.exp(-np.arange(30) / 8) + 0.1)
    plots.plot_error_curve(errors, "test", tmp / "curve")
    plots.plot_curve_overlay([("a", errors), ("b", errors[:20])], "test", tmp / "overlay")
    matrix = [[5, 1], [2, 7]]
    plots.plot_confusion_pair(matrix, matrix, ["c0", "c1"], "test", tmp / "cm")
    plots.plot_length_hist({"c0": [3, 4, 5], "c1": [6, 7, 9]}, "test", "length", tmp / "hist")
    strokes = {"c0": [rng.uniform(size=(20, 2))] * 2, "c1": [rng.uniform(size=(15, 2))]}
    plots.plot_hw_gallery(strokes, "test", tmp / "gallery")
    plots.plot_acc_vs_length([("m", [5, 9, 12, 20], [1, 0, 1, 1])], [5, 10, 15, 20], "test", tmp / "acc")
    for name in ("curve", "overlay", "cm", "hist", "gallery", "acc"):
        check((tmp / f"{name}.png").exists() and (tmp / f"{name}.pdf").exists(), name)


def check_real_data():
    """Only if the data folder is present: both datasets load, 5 classes, expected dimensions."""
    for name, spec in config.DATASETS.items():
        if not (config.DATA_DIR / spec.folder / spec.mapping_file).exists():
            print(f"      (skipped {name}: no data at {config.DATA_DIR / spec.folder})")
            continue
        check(len(config.get_classes(spec)) == 5)
        for split in ("train", "test"):
            samples = load_split(spec, split)
            check(len({s.label for s in samples}) == 5)
            check(all(s.x.shape[1] == spec.input_dim for s in samples))
            if name == "hw":
                check(all(s.x.min() >= 0 and s.x.max() <= 1 for s in samples))


def main():
    checks = [
        ("mapping file parser", check_mapping_parser, False),
        ("file readers", check_readers, True),
        ("handwriting normalisation", check_normalisation, False),
        ("padding is ignored", check_padding_is_ignored, False),
        ("model matches slide equations", check_matches_slide_equations, False),
        ("CV standardiser", check_standardiser, False),
        ("stratified split", check_split, False),
        ("convergence criterion", check_convergence_logic, False),
        ("RNN and LSTM can learn", check_learning, False),
        ("checkpoint save/load", check_checkpoint, True),
        ("metrics", check_metrics, False),
        ("plots", check_plots, True),
        ("real data", check_real_data, False),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for name, check, needs_tmp in checks:
            try:
                if needs_tmp:
                    check(tmp)
                else:
                    check()
                print(f"PASS  {name}")
            except Exception:
                failures += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
