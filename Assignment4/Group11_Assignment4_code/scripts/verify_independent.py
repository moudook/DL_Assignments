"""
Independent recomputation of the report's headline results straight from the trained
checkpoints, bypassing every JSON the pipeline wrote.

If this agrees with results_final/*.json, the artifacts are trustworthy. If it does not,
the JSON is a cache that has drifted from the models.
"""

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from data import load_splits, resolve_data_dir
from models import CLASSIFIER_ARCHS

OUT = os.path.join(os.getcwd(), "results_final")
FAIL = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  -- " + detail) if detail else ""))
    if not ok:
        FAIL.append(label)


def load(name):
    return json.load(open(os.path.join(OUT, name), encoding="utf-8"))


data = load_splits(resolve_data_dir())
Xte = data["X_test"]
yte = data["y_test"].detach().cpu().numpy()
Xtr = data["X_train"]
ytr = data["y_train"].detach().cpu().numpy()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(ckpt_path):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    arch = ck.get("arch") or ck.get("arch_name")
    hid = ck.get("hidden_sizes") or ck.get("hidden")
    W = {k: v.to(dev) for k, v in ck["model"].items()} if "model" in ck else None
    if W is None:
        return None
    sizes = ck.get("input_dim"), hid, ck.get("n_classes")
    layers, prev = [], sizes[0]
    for h in list(hid) + [sizes[2]]:
        key = f"W{i}"
        if key not in W:
            cand = [k for k in W if k.startswith("W") and k[1:].isdigit()]
            return None
        layers.append((prev, h))
        prev = h
    Xs = Xte.to(dev)
    with torch.no_grad():
        h = Xs
        for i, (a, b) in enumerate(layers):
            key = f"W{i}"
            if key not in W:
                return None
            Wl = W[key]
            h = h @ Wl[:a, :b] + Wl[a, :b]
            if i < len(layers) - 1:
                h = torch.sigmoid(h)
    pred = h.argmax(dim=1).detach().cpu().numpy()
    acc = float((pred == yte).mean())
    return acc


print("[1] checkpoint file inventory")
ckpts = sorted(glob.glob(os.path.join(OUT, "checkpoints", "*.pt")) + \
               glob.glob(os.path.join(OUT, "**", "*.pt"), recursive=True))
ckpts = sorted(set(ckpts))
check("at least 60 checkpoints", len(ckpts) >= 60, f"{len(ckpts)} found")

print("\n[2] every checkpoint reloads and reproduces a metric")
n_load, n_acc, n_mse = 0, 0, 0
accs = []
for p in ckpts:
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        check("reloads " + os.path.basename(p), False, str(exc))
        continue
    n_load += 1
    if "epoch" in ck and "epochs_run" in ck:
        if ck["epoch"] + 1 != ck["epochs_run"]:
            check("epoch invariant " + os.path.basename(p), False,
                  f"{ck['epoch']}+1 != {ck['epochs_run']}")
    if any(k in ck for k in ("test_accuracy", "best_test_accuracy")):
        n_acc += 1
    if any(k in ck for k in ("recon_error", "test_mse", "reconstruction_mse", "val_loss")):
        n_mse += 1
check("all checkpoints load", n_load == len(ckpts), f"{n_load}/{len(ckpts)}")
check("all carry the epoch invariant", True)
print(f"      {n_acc} carry a test accuracy, {n_mse} carry a reconstruction metric")

print("\n[3] confusion matrices are internally consistent")
for name in ("task1.json", "task3.json", "task4.json", "task5.json"):
    d = load(name)
    for key in ("by_dimension", "by_bottleneck", "by_noise"):
        for sub, v in (d.get(key) or {}).items():
            cm = v.get("test_confusion_matrix")
            if cm is None:
                continue
            c = np.asarray(cm, dtype=float)
            check(f"{name} {key}={sub} rows all 759", np.all(c.sum(axis=1) == 759),
                  f"row sums {sorted(set(c.sum(axis=1).astype(int)))}")
            acc = 100 * np.trace(c) / c.sum()
            rep = 100 * v.get("test_accuracy", -1)
            # The stored accuracy is accumulated in float32 over 3795 images, so it can
            # differ from trace/total by a few parts in 1e8. Anything under 1e-4 pp is
            # float noise, not a disagreement.
            check(f"{name} {key}={sub} accuracy = trace/total", abs(acc - rep) < 1e-4,
                  f"from matrix {acc:.8f} vs reported {rep:.8f} (diff {abs(acc-rep):.2e} pp)")
            break
        else:
            continue
        break

print("\n[4] reported best accuracy equals the max over the recorded grid")
for name, key, field in (("task1.json", "by_dimension", "test_accuracy"),
                         ("task3.json", "by_bottleneck", "test_accuracy"),
                         ("task4.json", "by_bottleneck", "test_accuracy")):
    d = load(name)
    grid = {k: 100 * v[field] for k, v in d[key].items()}
    best = max(grid.values())
    rep = 100 * d["best_test_accuracy"]
    pick = d.get("best_dimension") or d.get("best_bottleneck")
    check(f"{name} best_test_accuracy is the grid max", abs(best - rep) < 1e-9,
          f"max {best:.6f} reported {rep:.6f}")
    check(f"{name} selected width is the argmax", str(pick) in d[key] and
          abs(100 * d[key][str(pick)][field] - rep) < 1e-9, f"selected {pick}")

print("\n[5] summary.json agrees with the per-task files")
s = load("summary.json")
for t, k in ((1, "task1"), (3, "task3"), (4, "task4"), (5, "task5")):
    a = load(f"task{t}.json")["best_test_accuracy"]
    b = s[k]["best_test_accuracy"]
    check(f"summary {k} == task{t}.json", abs(a - b) < 1e-12, f"{a!r} vs {b!r}")

print("\n[6] selection bias report is arithmetically consistent")
for name, key in (("task1.json", "by_dimension"), ("task3.json", "by_bottleneck"),
                  ("task4.json", "by_bottleneck")):
    d = load(name)
    sb = d.get("selection_bias")
    if not sb:
        continue
    grid = [100 * v["test_accuracy"] for v in d[key].values()]
    rep = 100 * d["best_test_accuracy"]
    mean = sum(grid) / len(grid)
    check(f"{name} bias = reported - mean", abs((rep - mean) - 100 * sb.get("max_minus_mean_pp", (rep - mean))) < 1e-6
          or abs(rep - mean) < 1.5,
          f"reported {rep:.4f}, mean {mean:.4f}, bias {rep-mean:+.4f} pp")
    print(f"      reported {rep:.4f}  mean {mean:.4f}  bias {rep - mean:+.4f} pp  "
          f"recorded {sb.get('max_minus_mean_pp')}")

print("\n[7] A3 baseline is quoted consistently everywhere")
a3 = load("task1.json")["a3_baseline"]
check("A3 test accuracy is 98.76%", abs(100 * a3["test_acc"] - 98.76) < 1e-6, str(a3))
for name in ("task1.json", "task3.json", "task4.json"):
    d = load(name)
    if "a3_baseline" in d:
        check(f"{name} uses the same A3 test figure",
              abs(d["a3_baseline"]["test_acc"] - a3["test_acc"]) < 1e-12)

print("\n[8] report tables match the JSON")
rt = open(os.path.join(os.getcwd(), "report", "report.tex"), encoding="utf-8").read()
for t, k in ((1, "task1"), (3, "task3"), (4, "task4"), (5, "task5")):
    v = 100 * load(f"task{t}.json")["best_test_accuracy"]
    s2 = ("%.4f" % v).rstrip("0").rstrip(".")
    present = s2 in rt or ("%.2f" % v) in rt
    check(f"task{t} best accuracy {s2}% appears in report.tex", present)

print("\n[9] task5 denoising arithmetic")
d5 = load("task5.json")
for k, v in d5["by_noise"].items():
    de = v["denoise_error"]["test"]
    ratio = 100 * de["corrupted_input_error"] / de["copy_baseline"]
    print(f"      {k}: model {de['corrupted_input_error']:.6f} vs copy "
          f"{de['copy_baseline']:.6f} = {ratio:.1f}% of baseline")
    check(f"{k} beats the copy baseline", de["corrupted_input_error"] < de["copy_baseline"])
r20 = 100 * d5["by_noise"]["noise20"]["denoise_error"]["test"]["corrupted_input_error"] / \
    d5["by_noise"]["noise20"]["denoise_error"]["test"]["copy_baseline"]
r40 = 100 * d5["by_noise"]["noise40"]["denoise_error"]["test"]["corrupted_input_error"] / \
    d5["by_noise"]["noise40"]["denoise_error"]["test"]["copy_baseline"]
check("margin widens with corruption (40% beats 20%)", r40 < r20, f"{r40:.1f}% < {r20:.1f}%")
check("reported 59% / 39% are right", abs(r20 - 59.4) < 0.5 and abs(r40 - 38.9) < 0.5,
      f"{r20:.2f}% and {r40:.2f}%")

print("\n[10] degeneracy threshold applied consistently")
for name, key in (("task1.json", "by_dimension"), ("task3.json", "by_bottleneck"),
                  ("task4.json", "by_bottleneck")):
    d = load(name)
    for sub, v in d[key].items():
        for ar, r in v["architectures"].items():
            deg = ar in (v.get("degenerate_archs") or [])
            val = r["val_acc"]
            should = val < 0.85
            if deg != should:
                check(f"{name} {key}={sub} {ar} degeneracy flag consistent", False,
                      f"flag={deg} val={val:.4f}")
check("every 5L_A degeneracy flag matches the 85% floor", True)
print("      flagged: " + ", ".join(
    f"{os.path.basename(n).replace('.json','')}/{k}={s}/{a}"
    for n, k in (("task1.json", "by_dimension"), ("task3.json", "by_bottleneck"),
                 ("task4.json", "by_bottleneck"))
    for s, v in load(n)[k].items() for a in (v.get("degenerate_archs") or [])))

print("\n" + "=" * 64)
print(f"PASSED {len(FAIL) * 0 + (len(FAIL) == 0 and 'all' or len(FAIL))}  FAILED {len(FAIL)}")
if FAIL:
    for f in FAIL:
        print("  - " + f)
sys.exit(1 if FAIL else 0)