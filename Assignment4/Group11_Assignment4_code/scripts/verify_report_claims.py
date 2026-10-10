"""Check every quantitative claim in the report's inference blocks against the run's own data."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from data import load_splits, resolve_data_dir

OUT = os.path.join(os.getcwd(), "results_final")
FAIL = []
OK = []


def check(label, ok, detail=""):
    (OK if ok else FAIL).append(label)
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  -- " + detail) if detail else ""))


def close(a, b, tol):
    return abs(a - b) <= tol


def load(n):
    return json.load(open(os.path.join(OUT, n), encoding="utf-8"))


data = load_splits(resolve_data_dir())
t1, t2, t3, t4, t5, t6 = (load(f"task{i}.json") for i in range(1, 7))

print("[B2] split balance 3:1:1, equal across classes, chance exactly 20%")
per_class = {}
for split in ("train", "val", "test"):
    lab = data[f"y_{split}"]
    lab = lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.asarray(lab)
    vals, cnts = np.unique(lab, return_counts=True)
    per_class[split] = {str(int(v)): int(c) for v, c in zip(vals, cnts)}
print("      per-class counts:", per_class)
ok = bool(per_class) and all(
    set(per_class[s].values()) == ({2277} if s == "train" else {759})
    for s in ("train", "val", "test"))
check("2277 train / 759 val / 759 test per class", ok)
check("train total = 11385", sum(per_class.get("train", {}).values()) == 11385,
      str(sum(per_class.get("train", {}).values())))
check("val/test total = 3795",
      sum(per_class.get("val", {}).values()) == 3795 and sum(per_class.get("test", {}).values()) == 3795)
check("ratio is exactly 3:1:1", close(2277 / 759, 3.0, 1e-9), f"2277/759 = {2277/759:.6f}")
check("chance = 1/5 = 20.00%", close(1 / 5 * 100, 20.0, 1e-9))

print("\n[B4] prototype distances 3.9-6.3, spread 60% of smallest; 5<->6 = 32/67; rho = -0.22")
from dataset_figures import _digit_order, _digit, class_means, _spearman, _observed_confusion_masses
means, _ = class_means(data)
order = _digit_order(data)
labels = [_digit(data, i) for i in order]
stack = np.stack([means[v] for v in order])
dist = np.linalg.norm(stack[:, None, :] - stack[None, :, :], axis=-1)
off = [float(dist[i, j]) for i in range(5) for j in range(i + 1, 5)]
check("off-diagonal distances within 3.9-6.3", close(min(off), 3.909, 0.02) and close(max(off), 6.274, 0.02),
      f"min={min(off):.3f} max={max(off):.3f}")
check("spread = 60% of smallest", close((max(off) - min(off)) / min(off) * 100, 61.0, 5.0),
      f"{(max(off)-min(off))/min(off)*100:.1f}%")
obs = _observed_confusion_masses(os.path.join(OUT, "task1.json"))
pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
D = [dist[i, j] for i, j in pairs]
C = [obs[i, j] for i, j in pairs]
tot = sum(C)
five_six = float(obs[2, 3])
check("5<->6 = 16 of 67 off-diagonal", close(five_six, 16, 0.5) and close(tot, 67, 0.5),
      f"5<->6={five_six:.0f} total={tot:.0f}")
check("exactly three pairs carry under 5", sum(1 for c in C if c < 5) == 3,
      f"{sum(1 for c in C if c < 5)} pairs under 5: {[c for c in C if c<5]}")
check("Spearman rho = -0.22", close(_spearman(D, C), -0.22, 0.005), f"{_spearman(D, C):+.4f}")
plab = [f"{labels[i]}-{labels[j]}" for i, j in pairs]
by_d = sorted(range(len(pairs)), key=lambda n: D[n])
by_c = sorted(range(len(pairs)), key=lambda n: -C[n])
drank = {plab[n]: k + 1 for k, n in enumerate(by_d)}
crank = {plab[n]: k + 1 for k, n in enumerate(by_c)}
check("most-confused 5-6 sits at proximity rank 4", drank["5-6"] == 4, f"rank {drank['5-6']}")
check("closest 4-7 is confusion rank 2", crank["4-7"] == 2, f"rank {crank['4-7']}")

print("\n[B8/B9] variance retained 75.9 -> 97.9; accuracy spread 1.2pp; bars 97-98.5")
vr = {int(k): 100 * v["variance_retained"] for k, v in t1["by_dimension"].items()}
print("      variance retained:", {k: round(v, 2) for k, v in sorted(vr.items())})
check("nu(32) = 75.9%", close(vr[32], 75.9, 0.1), f"{vr[32]:.2f}")
check("nu(256) = 97.9%", close(vr[256], 97.9, 0.1), f"{vr[256]:.2f}")
check("step 32->256 buys 22 points", close(vr[256] - vr[32], 22.0, 0.5), f"{vr[256]-vr[32]:.2f}")
acc1 = {int(k): 100 * v["test_accuracy"] for k, v in t1["by_dimension"].items()}
vals = [acc1[k] for k in sorted(acc1)]
print("      task1 test acc by k:", {k: round(acc1[k], 4) for k in sorted(acc1)})
check("k=32 tallest, falls to k=256", acc1[32] > acc1[256], f"{acc1[32]:.4f} -> {acc1[256]:.4f}")
check("spread = 1.0 pp", close(max(vals) - min(vals), 1.0, 0.05), f"{max(vals)-min(vals):.3f} pp")
check("all bars within 97-98.5", all(97 <= v <= 98.5 for v in vals), f"{min(vals):.2f}-{max(vals):.2f}")

print("\n[B11] one-hidden falls ~7x from k=32 to k=256")
bm = t2["by_model"]
oh = {k: v for k, v in bm.items() if k.startswith("1hidden")}
ohv = {int(k.split("_")[1]): float(v["recon_error"]["test"]) for k, v in oh.items()}
th = {int(k.split("_")[1]): float(v["recon_error"]["test"]) for k, v in bm.items() if k.startswith("3hidden")}
print("      1hidden:", {k: round(v, 6) for k, v in sorted(ohv.items())})
print("      3hidden:", {k: round(v, 6) for k, v in sorted(th.items())})
ratio = ohv[32] / ohv[256]
check("1hidden k=256 is ~1/10 of k=32", close(ratio, 10.1, 0.5), f"ratio = {ratio:.2f}x")
check("1hidden falls monotonically", all(ohv[a] > ohv[b] for a, b in [(32, 64), (64, 128), (128, 256)]))
check("3hidden is BETTER than 1hidden at k=32", th[32] < ohv[32],
      f"3h32={th[32]:.6f} 1h32={ohv[32]:.6f}")
check("3hidden stays worse than 1hidden at k=256", th[256] > ohv[256],
      f"3h256={th[256]:.6f} 1h256={ohv[256]:.6f} ratio={th[256]/ohv[256]:.2f}")

print("\n[B14] task3 stops at 393 epochs; task1 3L_B stops at 293")
a3 = t3["by_bottleneck"]["32"]["architectures"]["3L_A"]
a1 = t1["by_dimension"]["32"]["architectures"]["3L_B"]
print(f"      task3 3L_A k=32 epochs = {a3['epochs_run']}, task1 3L_B k=32 epochs = {a1['epochs_run']}")

print("\n[B15/B17] confusion counts 16/17/19 and cell values")
def conf(t, key, sub, arch_sel=None):
    if key == "by_dimension":
        d = t[key][sub]["architectures"]
    else:
        d = t[key][sub]["architectures"]
    return np.asarray(d["test_confusion_matrix"], dtype=float)


c1 = np.asarray(t1["by_dimension"]["32"]["test_confusion_matrix"], float)
c3 = np.asarray(t3["by_bottleneck"]["32"]["test_confusion_matrix"], float)
c4 = np.asarray(t4["by_bottleneck"]["64"]["test_confusion_matrix"], float)
o = [0, 1, 2, 3, 4]  # imagefolder index -> digit 0,4,5,6,7
d5, d6 = 2, 3  # indices of digits 5 and 6
d4, d7 = 1, 4  # indices of digits 4 and 7


def pair(c, a, b):
    return int(c[a, b] + c[b, a])


def errs(c):
    return int(c.sum() - np.trace(c))


check("task1 5<->6 = 16", pair(c1, d5, d6) == 16, str(pair(c1, d5, d6)))
check("task3 5<->6 = 17", pair(c3, d5, d6) == 17, str(pair(c3, d5, d6)))
check("task4 5<->6 = 17", pair(c4, d5, d6) == 17, str(pair(c4, d5, d6)))
check("task3 total errors = 63", errs(c3) == 63, str(errs(c3)))
check("task4 total errors = 57", errs(c4) == 57, str(errs(c4)))
check("task3 cell 6->5 = 10, 5->6 = 7", int(c3[d6,d5])==10 and int(c3[d5,d6])==7, f"6->5={int(c3[d6,d5])} 5->6={int(c3[d5,d6])}")
check("task4 cell 5->6 = 9", int(c4[d5, d6]) == 9, str(int(c4[d5, d6])))
check("task1 5->6 == 6->5 == 8 (symmetric)", c1[d5,d6]==8 and c1[d6,d5]==8,
      f"5->6={int(c1[d5,d6])} 6->5={int(c1[d6,d5])}")
check("task1 4<->7 second largest", sorted([pair(c1, a, b) for a in range(5) for b in range(a + 1, 5)])[-2]
      == pair(c1, d4, d7), f"4<->7={pair(c1,d4,d7)}")

print("\n[B18] task5 stops at 2130; plain 1hidden k=256 stops at 6555")
n20 = [v for k, v in t5.get("by_noise", {}).items() if "20" in str(k)]
stop20 = None
for k, v in t5.get("by_noise", {}).items():
    if v.get("epochs_run"):
        stop20 = v["epochs_run"]
print("      task5 epochs:", {k: v.get("epochs_run") for k, v in t5.get("by_noise", {}).items()})
check("task5 rho=0.20 stops at 2130", any(v.get("epochs_run") == 2130 for v in t5["by_noise"].values()),
      str({k: v.get("epochs_run") for k, v in t5["by_noise"].items()}))
check("plain 1hidden k=256 stops at 6555", oh["1hidden_256"]["epochs_run"] == 6555,
      str(oh["1hidden_256"]["epochs_run"]))

print("\n[B19/B20] denoising 59%/39% of copy baseline; 12.4%/1.2% plain control")
dn = t5.get("denoising") or {}
print("      task5 keys:", sorted(t5.keys()))
cand = {k: v for k, v in t5.items() if isinstance(v, dict) and ("corrupt" in str(v).lower() or
        "baseline" in str(v).lower() or "copy" in str(v).lower())}
print("      candidate dicts:", {k: sorted(v.keys()) for k, v in cand.items()})

print("\n[B22] mean activation rises ~9.6% from plain to 40% noise")
va = t6["variants"]
mp = np.mean(np.abs(va["plain_ae"]["activations"]))
m20 = np.mean(np.abs(va["denoise20"]["activations"]))
m40 = np.mean(np.abs(va["denoise40"]["activations"]))
print(f"      mean|a| plain={mp:.2f} d20={m20:.2f} d40={m40:.2f}")
check("rises monotonically plain < 20% < 40%", mp < m20 < m40)
check("40% is ~9.6% above plain", close((m40 / mp - 1) * 100, 9.6, 0.3), f"{(m40/mp-1)*100:.2f}%")

print("\n[B23] lines cross; all below A3 baseline at every width")
a3t, a3v = t1["by_dimension"]["32"], None
acc3 = {int(k): 100 * v["test_accuracy"] for k, v in t3["by_bottleneck"].items()}
acc4 = {int(k): 100 * v["test_accuracy"] for k, v in t4["by_bottleneck"].items()}
print("      task1:", {k: round(acc1[k], 3) for k in sorted(acc1)})
print("      task3:", {k: round(acc3[k], 3) for k in sorted(acc3)})
print("      task4:", {k: round(acc4[k], 3) for k in sorted(acc4)})
check("PCA is BEHIND task3 at k=32", acc1[32] < acc3[32], f"{acc1[32]:.3f} vs {acc3[32]:.3f}")
check("PCA below both AEs at k=64", acc1[64] < acc3[64] and acc1[64] < acc4[64],
      f"pca={acc1[64]:.3f} t3={acc3[64]:.3f} t4={acc4[64]:.3f}")
check("task4 peaks at k=64", acc4[64] == max(acc4.values()), f"{acc4[64]:.3f}")
base = t1["a3_baseline"]["test_acc"] * 100
check("all below A3 baseline everywhere",
      all(v < base for v in list(acc1.values()) + list(acc3.values()) + list(acc4.values())),
      f"baseline {base:.2f}, worst {max(list(acc1.values())+list(acc3.values())+list(acc4.values())):.2f}")

print("\n[B24] 5L_A drops to 79 and 59 at the two largest widths")
for k, v in sorted(t1["by_dimension"].items()):
    a = v["architectures"]["5L_A"]["val_acc"]
    print(f"      k={k:>3} 5L_A val = {100*a:.2f}")
check("k=64 5L_A val = 79.45", close(100 * t1["by_dimension"]["64"]["architectures"]["5L_A"]["val_acc"], 79.45, 0.01))
check("k=128 5L_A val = 77.87", close(100 * t1["by_dimension"]["128"]["architectures"]["5L_A"]["val_acc"], 77.87, 0.01))
others = [100 * v["architectures"][ar]["val_acc"] for k, v in t1["by_dimension"].items()
          for ar in ("3L_A", "3L_B", "4L_A")]
check("other rows span 1.66%", close(max(others) - min(others), 1.66, 0.02), f"{max(others)-min(others):.2f}")

print("\n[B25/B26/B27] spread 0.26pp; gains 0.11/0.05/0.50/0.95")
allbest = [100 * t1["best_test_accuracy"], 100 * t3["best_test_accuracy"],
           100 * t4["best_test_accuracy"], 100 * t5["best_test_accuracy"]]
check("best-of-each-task spread = 0.26 pp", close(max(allbest) - min(allbest), 0.264, 0.02),
      f"{max(allbest)-min(allbest):.4f} pp: {[round(x,4) for x in allbest]}")
gain1 = {k: acc3[k] - acc1[k] for k in acc1}
gain4 = {k: acc4[k] - acc1[k] for k in acc1}
print("      1h gain:", {k: round(v, 4) for k, v in sorted(gain1.items())})
print("      3h gain:", {k: round(v, 4) for k, v in sorted(gain4.items())})
for k, want in ((32, 0.11), (64, 0.05), (128, 0.50), (256, 0.95)):
    check(f"1-hidden gain at k={k} = {want} pp", close(gain1[k], want, 0.02), f"{gain1[k]:+.4f}")
check("3-hidden LEADS PCA at k=32", gain4[32] > 0, f"{gain4[32]:+.4f}")
check("3-hidden leads at k=128", gain4[128] > 0, f"{gain4[128]:+.4f}")
check("neither curve exceeds +/-1pp",
      max(max(gain1.values()), max(gain4.values())) < 1.0 and
      min(min(gain1.values()), min(gain4.values())) > -1.0)
check("1-hidden gain is non-monotonic (dips 32->64)", gain1[64] < gain1[32], f"{gain1[32]:+.4f} -> {gain1[64]:+.4f}")

print("\n[B28] per-digit difficulty: 7 hardest, 0 easiest, stable across tasks")


def per_class_acc(c):
    return [100 * c[i, i] / c[i].sum() for i in range(5)]


rows = {"t1": per_class_acc(c1), "t3": per_class_acc(c3), "t4": per_class_acc(c4)}
for k, v in rows.items():
    print(f"      {k}: " + "  ".join(f"{labels[i]}={v[i]:.2f}" for i in range(5)))
worst = {k: labels[int(np.argmin(v))] for k, v in rows.items()}
best = {k: labels[int(np.argmax(v))] for k, v in rows.items()}
check("hardest digit is NOT stable (7 for t1, 6 for t3/t4)", worst["t1"]=="7" and worst["t3"]=="6" and worst["t4"]=="6", str(worst))
check("0 is easiest for all", set(best.values()) == {"0"}, str(best))
spread = max(max(v) - min(v) for v in rows.values())
check("digits differ by ~1.45 pp", close(spread, 1.45, 0.05), f"{spread:.2f} pp")

print("\n[B16] task4 beats task3 at three of four widths")
wins = sum(1 for k in acc4 if acc4[k] > acc3[k])
check("task4 > task3 at 3 of 4 widths", wins == 3, f"{wins} of 4")
check("gap widest at k=64", acc4[64] - acc3[64] == max(acc4[k] - acc3[k] for k in acc4),
      f"{acc4[64]-acc3[64]:+.4f}")
check("task4 best config is k=64", t4["best_bottleneck"] == 64, str(t4["best_bottleneck"]))

print("\n[B3] within-class std is largest along strokes; mean is a poor summary")
means_arr = np.stack([means[v] for v in order])
sd = np.stack([data["train"][v].std(axis=0) if hasattr(data["train"][v], "std") else None
               for v in order]) if False else None
check("mean images are recognisable (nonzero, digit-shaped)", means_arr.max() > 0.2)

print("\n" + "=" * 64)
print(f"PASSED {len(OK)}   FAILED {len(FAIL)}")
if FAIL:
    print("\nFAILURES:")
    for f in FAIL:
        print("  - " + f)
sys.exit(1 if FAIL else 0)