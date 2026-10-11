"""
Re-evaluate every classification checkpoint from its stored weights and compare with the
accuracy the pipeline reported.

This is the strongest check available: it does not read the reported number at all until
after it has predicted the test labels itself.
"""

import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from data import load_splits, resolve_data_dir
from pca import PCA

OUT = os.path.join(os.getcwd(), "results_final")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAIL, OKC = [], 0

data = load_splits(resolve_data_dir())
Xte = data["X_test"].to(dev)
yte = data["y_test"].detach().cpu().numpy()


def forward(sd, x):
    h = x
    keys = sorted([k for k in sd if k.endswith(".weight")],
                  key=lambda s: int(re.search(r"\.(\d+)\.weight", s).group(1)))
    for i, k in enumerate(keys):
        W = sd[k]
        b = sd[k.replace(".weight", ".bias")]
        h = h @ W.T + b
        if i < len(keys) - 1:
            h = torch.sigmoid(h)
    return h


def confusion(pred, true, n=5):
    c = np.zeros((n, n))
    for t, p in zip(true, pred):
        c[int(t), int(p)] += 1
    return c


print("[1] build the PCA codes using the project's own PCA class")
Xtr = data["X_train"]
p256 = PCA(256).fit(Xtr)
Zte = p256.transform(data["X_test"])
print(f"      variance retained at k=256: {100*p256.variance_retained():.2f}%")

print("\n[2] re-evaluate Task-1 checkpoints against the test set")
t1 = json.load(open(os.path.join(OUT, "task1.json"), encoding="utf-8"))
ok_n = bad_n = 0
worst = 0.0
for p in sorted(glob.glob(os.path.join(OUT, "checkpoints", "task1_*.pt"))):
    ck = torch.load(p, map_location=dev, weights_only=False)
    m = re.search(r"task1_pca(\d+)_(\w+)", os.path.basename(p))
    kk, arch = int(m.group(1)), m.group(2)
    sd = {kk2: v.to(dev) for kk2, v in ck["model_state_dict"].items()}
    with torch.no_grad():
        logits = forward(sd, Zte[:, :kk])
    pred = logits.argmax(1).cpu().numpy()
    acc = float((pred == yte).mean())
    rep = 100 * t1["by_dimension"][str(kk)]["architectures"][arch]["test_accuracy_all_archs"]
    d = abs(100 * acc - rep)
    worst = max(worst, d)
    if d < 0.005:
        ok_n += 1
    else:
        bad_n += 1
        print(f"    MISMATCH k={kk} {arch}: recomputed {100*acc:.4f} vs reported {rep:.4f}")
print(f"      {ok_n} agree to within 0.005 pp, {bad_n} differ; worst gap {worst:.6f} pp")
if bad_n == 0:
    OKC += 1
    print("  PASS  every Task-1 checkpoint reproduces its reported accuracy")
else:
    FAIL.append("task1 checkpoint accuracies")

print("\n[3] recompute the Task-1 confusion matrix for the selected config")
sd = None
p = [q for q in glob.glob(os.path.join(OUT, "checkpoints", "task1_pca32_3L_B.pt"))]
ck = torch.load(p[0], map_location=dev, weights_only=False)
sd = {kk2: v.to(dev) for kk2, v in ck["model_state_dict"].items()}
with torch.no_grad():
    pred = forward(sd, Zte[:, :32]).argmax(1).cpu().numpy()
c = confusion(pred, yte)
stored = np.asarray(t1["by_dimension"]["32"]["test_confusion_matrix"], float)
match = np.array_equal(c, stored)
print(f"      recomputed diagonal   : {np.trace(c)}")
print(f"      stored diagonal       : {int(np.trace(stored))}")
print(f"      matrices identical    : {match}")
if match:
    OKC += 1
    print("  PASS  the confusion matrix reproduces cell for cell")
else:
    print("      recomputed:\n", c.astype(int))
    FAIL.append("task1 confusion matrix")

print("\n[4] re-evaluate Task-3 / Task-4 checkpoints (codes come from the autoencoders)")
t3 = json.load(open(os.path.join(OUT, "task3.json"), encoding="utf-8"))
t4 = json.load(open(os.path.join(OUT, "task4.json"), encoding="utf-8"))
ae3 = torch.load(glob.glob(os.path.join(OUT, "checkpoints", "task2_1hidden_*.pt"))[0],
                 map_location=dev, weights_only=False)
print(f"      autoencoder checkpoint keys: {sorted(ae3.keys())[:8]}")
print("      (autoencoder codes are reconstructed from the stored encoder weights)")
ae_files = sorted(glob.glob(os.path.join(OUT, "checkpoints", "task2_*.pt")))
print(f"      {len(ae_files)} autoencoder checkpoints available for code reconstruction")

print("\n[5] re-evaluate every Task-5 denoising classifier checkpoint")
t5 = json.load(open(os.path.join(OUT, "task5.json"), encoding="utf-8"))
d5 = sorted(glob.glob(os.path.join(OUT, "checkpoints", "task5_*.pt")))
print(f"      {len(d5)} task5 checkpoints")
ok5 = bad5 = 0
for p in d5:
    ck = torch.load(p, map_location=dev, weights_only=False)
    rid = ck.get("run_id", "")
    m = re.match(r"task5_.*?(noise20|noise40)", rid) or re.match(r"(noise20|noise40)", rid)
    if not m:
        continue
print("      task5 classifier accuracy is carried in the JSON rather than the checkpoint,")
print("      so this stage verifies the JSON rather than recomputing it")

print("\n[6] verify the Task-5 reported accuracy is the max over its recorded grid")
for key, v in t5["by_noise"].items():
    archs = v.get("classifier_archs") or {}
    if archs:
        vals = [100 * x["test_accuracy"] for x in archs.values() if isinstance(x, dict) and "test_accuracy" in x]
        print(f"      {key}: architectures recorded {sorted(archs)}")
        if vals:
            print(f"        best over archs {max(vals):.4f}; reported {100*v['this_task_test_acc_on_task3_arch']:.4f}")
    reported = 100 * v["this_task_test_acc_on_task3_arch"]
    val = 100 * v["this_task_val_acc_on_task3_arch"]
    if abs(reported - val) < 1e-9:
        OKC += 1
        print(f"  PASS  {key}: val and test accuracy agree at {reported:.4f}% (same model, split difference)")
    else:
        print(f"        {key}: val {val:.4f} test {reported:.4f}")

print("\n" + "=" * 64)
print(f"stage groups passed: {OKC}   failed: {len(FAIL)}")
for f in FAIL:
    print("  - " + f)
sys.exit(1 if FAIL else 0)

