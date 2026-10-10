"""Rebuild the Task-3/4/5 classifier predictions from the stored autoencoder + classifier weights."""

import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from data import load_splits, resolve_data_dir

OUT = os.path.join(os.getcwd(), "results_final")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_splits(resolve_data_dir())
Xte = data["X_test"].to(dev)
yte = data["y_test"].detach().cpu().numpy()
FAIL = []


def enc(sd, x):
    """Forward through the encoder sub-network only, returning the bottleneck code."""
    keys = sorted([k for k in sd if k.startswith("encoder.") and k.endswith(".weight")],
                  key=lambda s: int(re.search(r"\.(\d+)\.weight", s).group(1)))
    h = x
    for i, k in enumerate(keys):
        h = h @ sd[k].T + sd[k.replace(".weight", ".bias")]
        if i < len(keys) - 1:
            h = torch.sigmoid(h)
    return h


def dec(sd, z):
    keys = sorted([k for k in sd if k.startswith("decoder.") and k.endswith(".weight")],
                  key=lambda s: int(re.search(r"\.(\d+)\.weight", s).group(1)))
    h = z
    for i, k in enumerate(keys):
        h = h @ sd[k].T + sd[k.replace(".weight", ".bias")]
        if i < len(keys) - 1:
            h = torch.sigmoid(h)
    return h


def net_forward(sd, x, hidden_only=False):
    h = x
    keys = sorted([k for k in sd if k.endswith(".weight")],
                  key=lambda s: int(re.search(r"\.(\d+)\.weight", s).group(1)))
    for i, k in enumerate(keys):
        h = h @ sd[k].T + sd[k.replace(".weight", ".bias")]
        if i < len(keys) - 1:
            h = torch.sigmoid(h)
    return h


"""Rebuild the Task-2/3/4 predictions from the stored autoencoder + classifier weights."""

import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from data import load_splits, resolve_data_dir
from models import Autoencoder, FCNN, CLASSIFIER_ARCHS

OUT = os.path.join(os.getcwd(), "results_final")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_splits(resolve_data_dir())
Xte = data["X_test"].to(dev)
yte = data["y_test"].detach().cpu().numpy()
FAIL = []


def load_ae(kind, k):
    f = glob.glob(os.path.join(OUT, "checkpoints", f"task2_{kind}_k{k}.pt"))[0]
    ck = torch.load(f, map_location=dev, weights_only=False)
    sd = ck["model_state_dict"]
    b = ck.get("bottleneck", k)
    ae = Autoencoder(bottleneck=b, kind="1hidden" if kind == "1hidden" else "3hidden").to(dev)
    ae.load_state_dict(sd)
    ae.eval()
    return ae, ck


def load_clf(path, arch):
    ck = torch.load(path, map_location=dev, weights_only=False)
    ind = ck["model_state_dict"]["net.0.weight"].shape[1]
    net = FCNN(input_dim=ind, hidden_sizes=CLASSIFIER_ARCHS[arch], num_classes=5).to(dev)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net, ck

print("[B] Task-3: encode with the one-hidden AE, then score the stored classifier")
t3 = json.load(open(os.path.join(OUT, "task3.json"), encoding="utf-8"))
ok, bad, worst = 0, 0, 0.0
for p in sorted(glob.glob(os.path.join(OUT, "checkpoints", "task3_*.pt"))):
    m = re.search(r"task3_.*?k?(\d+)_(\w+)\.pt", os.path.basename(p))
    if not m:
        continue
    kk, arch = int(m.group(1)), m.group(2)
    ae, _ = load_ae("1hidden", kk)
    net, _ = load_clf(p, arch)
    with torch.no_grad():
        pred = net(ae.encode(Xte)).argmax(1).cpu().numpy()
    acc = 100 * float((pred == yte).mean())
    rep = 100 * t3["by_bottleneck"][str(kk)]["architectures"][arch]["test_accuracy_all_archs"]
    d = abs(acc - rep)
    worst = max(worst, d)
    if d < 0.02:
        ok += 1
    else:
        bad += 1
        print(f"    MISMATCH k={kk} {arch}: recomputed {acc:.4f} vs reported {rep:.4f}")
print(f"   {ok} agree within 0.02 pp, {bad} differ; worst {worst:.6f} pp")
if bad == 0 and ok:
    print("  PASS  every Task-3 checkpoint reproduces its reported accuracy")
else:
    FAIL.append("task3 recomputation")

print("\n[C] Task-4: same with the three-hidden autoencoders")
t4 = json.load(open(os.path.join(OUT, "task4.json"), encoding="utf-8"))
ok4, bad4, worst4 = 0, 0, 0.0
for p in sorted(glob.glob(os.path.join(OUT, "checkpoints", "task4_*.pt"))):
    m = re.search(r"task4_.*?k?(\d+)_(\w+)\.pt", os.path.basename(p))
    if not m:
        continue
    kk, arch = int(m.group(1)), m.group(2)
    ae, _ = load_ae("3hidden", kk)
    net, _ = load_clf(p, arch)
    with torch.no_grad():
        pred = net(ae.encode(Xte)).argmax(1).cpu().numpy()
    acc = 100 * float((pred == yte).mean())
    rep = 100 * t4["by_bottleneck"][str(kk)]["architectures"][arch]["test_accuracy_all_archs"]
    d = abs(acc - rep)
    worst4 = max(worst4, d)
    if d < 0.02:
        ok4 += 1
    else:
        bad4 += 1
        print(f"    MISMATCH k={kk} {arch}: recomputed {acc:.4f} vs reported {rep:.4f}")
print(f"   {ok4} agree within 0.02 pp, {bad4} differ; worst {worst4:.6f} pp")
if bad4 == 0 and ok4:
    print("  PASS  every Task-4 checkpoint reproduces its reported accuracy")
else:
    FAIL.append("task4 recomputation")

print("\n[D] Task-2: reconstruction error from the stored autoencoder weights")
t2 = json.load(open(os.path.join(OUT, "task2.json"), encoding="utf-8"))
worst_mse = 0.0
for kind in ("1hidden", "3hidden"):
    for kk in (32, 64, 128, 256):
        ae, _ = load_ae(kind, kk)
        with torch.no_grad():
            rec = ae(Xte)
            mse = float(((rec - Xte) ** 2).mean())
        rep = t2["by_model"][f"{kind}_{kk}"]["recon_error"]["test"]
        rel = abs(mse - rep) / rep
        worst_mse = max(worst_mse, rel)
        print(f"    {kind}_k{kk}: recomputed {mse:.8f} vs reported {rep:.8f}  (rel {rel:.2e})")
if worst_mse < 1e-4:
    print("  PASS  every reconstruction error reproduces from the stored weights")
else:
    FAIL.append("task2 reconstruction errors")
    print(f"  FAIL  worst relative error {worst_mse:.3e}")

print("\n[E] confusion matrices for the selected configurations, recomputed cell-for-cell")
for name, key, sub, clffile, kind, kk in (("task3.json", "by_bottleneck", "32", "task3_k32_3L_A.pt", "1hidden", 32),
                                          ("task4.json", "by_bottleneck", "64", "task4_k64_4L_A.pt", "3hidden", 64)):
    d = json.load(open(os.path.join(OUT, name), encoding="utf-8"))
    stored = np.asarray(d[key][sub]["test_confusion_matrix"], float)
    arch = "_".join(clffile[:-3].split("_")[-2:])
    cp = os.path.join(OUT, "checkpoints", clffile)
    if not os.path.exists(cp):
        cands = glob.glob(os.path.join(OUT, "checkpoints", f"*{sub}_{arch}.pt"))
        cp = cands[0] if cands else None
    if cp is None:
        print(f"    {name}: no checkpoint matching {clffile}")
        continue
    ae, _ = load_ae(kind, kk)
    net, _ = load_clf(cp, arch)
    with torch.no_grad():
        pred = net(ae.encode(Xte)).argmax(1).cpu().numpy()
    c = np.zeros((5, 5))
    for t_, p_ in zip(yte, pred):
        c[int(t_), int(p_)] += 1
    same = np.array_equal(c, stored)
    print(f"    {name} {key}={sub} {arch}: identical = {same} (trace {int(np.trace(c))} vs {int(np.trace(stored))})")
    if same:
        print(f"      PASS  matrix reproduces cell for cell")
    else:
        FAIL.append(f"{name} confusion matrix")
        print("      recomputed:\n", c.astype(int))
        print("      stored:\n", stored.astype(int))

print("\n" + "=" * 64)
print(f"FAILED: {len(FAIL)}")
for f in FAIL:
    print("  - " + f)
sys.exit(1 if FAIL else 0)


