"""
task1_pca.py — Assignment-4 Task-1: dimension reduction using PCA.

A4 Task-1 requirements, mapped to this module:
  a. Reduce to 32, 64, 128, 256. Eigenvectors from TRAINING data only; mean-
     subtracted train/val/test projected onto them; the TRAIN mean is used for
     val and test mean subtraction. Enforced by pca.PCA.fit/transform.
  b. For each reduced dimension: train an FCNN classifier, report validation
     accuracy for ALL four architectures, then test accuracy + confusion matrix
     for the best architecture chosen on VALIDATION accuracy.
  c. Observe which reduced dimension gives the best test accuracy.
  d. Compare against the Assignment-3 best.

Selection discipline: the architecture is chosen on VALIDATION accuracy, never on
test. Picking the test-best architecture would leak test information into model
selection and inflate the reported number. Test accuracy is then read once, for
the already-selected architecture.

Reads  results/task1.json
Writes results/selection.json  (key "task1") so downstream tasks can compare
"""

import json
import os

import numpy as np
import torch

import plots
from data import load_splits, CLASS_NAMES
from evaluate import evaluate_classifier
from models import CLASSIFIER_ARCHS, build_classifier
from pca import PCA
from run_tracker import RunTracker, atomic_write_text, thin_history
from train import train_classifier, MAX_EPOCHS, CLASSIFIER_LR, TOL

# The four reduced dimensions A4 mandates.
DIMENSIONS = [32, 64, 128, 256]

# A3 baseline, from the submitted Group11_Assignment3_report.pdf (submission of
# record). Recorded here so the comparison table in the report is reproducible.
A3_VAL_ACC = 0.9884
A3_TEST_ACC = 0.9876


def run_task1(data, outdir="results", dims=DIMENSIONS, max_epochs=MAX_EPOCHS,
              device=None, archs=None):
    """
    Fit PCA per dimension, train all classifiers, select best on validation.

    Returns a results dict, also written to <outdir>/task1.json.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archs = archs or list(CLASSIFIER_ARCHS)
    pca_dir = plots.ensure_dir(os.path.join(outdir, "plots", "task1"))

    results = {
        "task": 1,
        "name": "PCA dimension reduction",
        "dimensions": dims,
        "architectures": archs,
        "classifier_lr": CLASSIFIER_LR,
        "tolerance": TOL,
        "max_epochs": max_epochs,
        "batch_size": "full",
        "a3_baseline": {"val_acc": A3_VAL_ACC, "test_acc": A3_TEST_ACC},
        "by_dimension": {},
        "degenerate_note": (
            "Architectures finishing at chance accuracy are flagged "
            "'degenerate' and excluded from best-model selection. The "
            "1e-4 stopping rule cannot distinguish convergence from a "
            "vanishing gradient: a deep sigmoid stack stops moving within a "
            "few epochs precisely because nothing is learning."),
    }

    # Projected splits per dimension, retained so best-architecture selection can
    # be redone after degenerate runs are excluded.
    _representation_cache = {}

    for k in dims:
        print(f"\n{'=' * 60}\nTask-1: PCA k={k}\n{'=' * 60}")
        # Keep the projected splits so the architecture can be re-selected after
        # degenerate runs are excluded, without refitting PCA.
        _representation_cache[k] = PCA(k=k).fit(data["X_train"]).project_all(data)

        # Fit on TRAIN ONLY, then project all three splits with the train mean.
        pca = PCA(k=k).fit(data["X_train"])
        red = pca.project_all(data)
        var_ret = red["variance_retained"]
        print(f"  variance retained: {100 * var_ret:.2f}%")

        pca.save(os.path.join(outdir, f"pca_{k}.pt"))

        # Variance curve milestone figure (Task-1 explanation for why accuracy
        # rises with k). Full 784-point curve from a k=784 fit is exact and
        # costs ~1s, so use a wide fit for the curve and the k-fit for projection.
        wide = PCA(k=max(dims)).fit(data["X_train"])
        curve = wide.explained_variance_curve([64]).cpu().tolist()
        plots.pca_variance_curve(
            curve, f"PCA: cumulative variance retained (784 components)",
            os.path.join(pca_dir, "pca_variance_retained.png"))

        # PCA reconstruction grid: visual evidence of what each k preserves.
        sample_idx = [0, 2277, 4554]        # one sample from 3 of the 5 classes
        orig = data["X_train"][sample_idx].reshape(-1, 28, 28).cpu().numpy()
        k_recons = {}
        for kk in dims:
            pk = PCA(k=kk).fit(data["X_train"])
            z = pk.transform(data["X_train"][sample_idx])
            rec = (z @ pk.components[:, :kk].T + pk.mean)
            # Drop the sample axis: pca_reconstruction_grid lays out ONE original
            # against one reconstruction per k, so each entry must be a single
            # 28x28 image, not a (3,28,28) stack.
            k_recons[kk] = np.clip(
                rec.reshape(-1, 28, 28)[0].cpu().numpy(), 0, 1)
        plots.pca_reconstruction_grid(
            [orig[0]], k_recons, "PCA reconstruction of a '0' at each k",
            os.path.join(pca_dir, "pca_reconstructions.png"))

        dim_entry = {
            "k": k,
            "variance_retained": var_ret,
            "architectures": {},
        }

        for arch in archs:
            run_id = f"task1_pca{k}_{arch}"
            tracker = RunTracker(run_id, outdir, total_epochs=max_epochs)
            model = build_classifier(
                arch, input_dim=k, num_classes=len(CLASS_NAMES)).to(device)

            res = train_classifier(model, red, f"pca{k}_{arch}", tracker=tracker,
                                   max_epochs=max_epochs, device=device)
            dim_entry["architectures"][arch] = {
                "val_acc": res["final_val_acc"],
                "epochs_run": res["epochs_run"],
                "stopped_early": res["stopped_early"],
                "degenerate": res.get("degenerate", False),
                "terminal_state": res.get("terminal_state"),
                "skipped": res.get("skipped", False),
                "hidden_sizes": CLASSIFIER_ARCHS[arch],
                "history": res["train_loss"],
                "model_state": res["model_state"],
            }
            flag = "  <-- DEGENERATE (chance)" if res.get("degenerate") else ""
            print(f"  {arch}: val={100 * res['final_val_acc']:.2f}% "
                  f"({res['epochs_run']} epochs){flag}")

        # Select best architecture on VALIDATION accuracy. Ties broken by
        # earliest convergence so the cheaper model wins an exact tie.
        best_arch = max(
            dim_entry["architectures"].items(),
            key=lambda kv: (kv[1]["val_acc"], -kv[1]["epochs_run"]),
        )[0]

        best = dim_entry["architectures"][best_arch]
        model = build_classifier(
            best_arch, input_dim=k, num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(best["model_state"])

        test_res = evaluate_classifier(model, red, device=device, split="test")
        train_res = evaluate_classifier(model, red, device=device, split="train")

        dim_entry["best_arch"] = best_arch
        dim_entry["test_accuracy"] = test_res["accuracy"]
        dim_entry["train_accuracy"] = train_res["accuracy"]
        dim_entry["test_confusion_matrix"] = test_res["confusion_matrix"]
        dim_entry["test_per_class"] = test_res["per_class"]

        print(f"  best arch (on val): {best_arch} "
              f"val={100 * best['val_acc']:.2f}%  "
              f"TEST={100 * test_res['accuracy']:.2f}%")

        # Milestone figure: confusion matrix for the selected architecture.
        plots.confusion_matrix(
            test_res["confusion_matrix"],
            f"Task-1 test confusion matrix — PCA k={k}, {best_arch}\n"
            f"test accuracy {100 * test_res['accuracy']:.2f}%",
            os.path.join(pca_dir, f"task1_confusion_pca{k}_{best_arch}.png"))

        results["by_dimension"][k] = dim_entry

    # ── Task-1c: best reduced dimension, by TEST accuracy ──────────────────
    # Degenerate runs (finished at chance accuracy) are EXCLUDED from selection.
    # A collapsed model that happens to edge out a real one must never be chosen
    # as "best", and the architecture for each dimension was picked earlier on
    # validation accuracy - so re-do that selection among the survivors, then
    # recompute its test accuracy to match.
    for d in dims:
        e = results["by_dimension"][d]
        collapsed = [a for a in e["architectures"]
                     if e["architectures"][a].get("degenerate")]
        survivors = {a: v for a, v in e["architectures"].items()
                     if a not in collapsed}
        if collapsed:
            print(f"  k={d}: excluded degenerate arch(s) {collapsed}")
        if not survivors:
            print(f"  k={d}: WARNING all architectures degenerate; keeping the "
                  f"original selection and reporting it as unreliable")
            continue

        # Re-select on validation among survivors only.
        best_arch = max(survivors.items(),
                        key=lambda kv: (kv[1]["val_acc"], -kv[1]["epochs_run"]))[0]
        changed = best_arch != e["best_arch"]
        e["best_arch"] = best_arch
        e["degenerate_archs"] = collapsed
        if changed:
            # Test accuracy must follow the architecture actually selected, so
            # recompute it rather than leaving the old number in place.
            model = build_classifier(best_arch, input_dim=d,
                                     num_classes=len(CLASS_NAMES)).to(device)
            model.load_state_dict(e["architectures"][best_arch]["model_state"])
            res = evaluate_classifier(model, _representation_cache[d],
                                      device=device, split="test")
            e["test_accuracy"] = res["accuracy"]
            e["test_confusion_matrix"] = res["confusion_matrix"]
            e["test_per_class"] = res["per_class"]
            print(f"  k={d}: best arch re-selected -> {best_arch} "
                  f"(test {100 * res['accuracy']:.2f}%)")

    best_dim = max(results["by_dimension"].items(),
                   key=lambda kv: kv[1]["test_accuracy"])
    results["best_dimension"] = int(best_dim[0])
    results["best_test_accuracy"] = best_dim[1]["test_accuracy"]
    results["best_dimension_arch"] = best_dim[1]["best_arch"]

    print(f"\n  BEST dimension by test accuracy: k={results['best_dimension']} "
          f"({100 * results['best_test_accuracy']:.2f}%)")

    # Milestone comparison bars across dimensions — the figure that answers
    # "which dimension is best". Built from the results table, so essentially free.
    plots.dimension_bars(
        {"test accuracy": [100 * results["by_dimension"][d]["test_accuracy"]
                           for d in dims],
         "val accuracy": [100 * max(
             results["by_dimension"][d]["architectures"][a]["val_acc"]
             for a in archs) for d in dims]},
        [str(d) for d in dims],
        "Task-1: PCA classification accuracy vs retained dimension",
        os.path.join(pca_dir, "task1_accuracy_vs_dimension.png"),
        ref=100 * A3_TEST_ACC,
        ref_label="A3 baseline (784-d input, NAG) 98.76%")

    # Milestone: loss curves for the selected architecture at each dimension.
    hist, labels = [], []
    for d in dims:
        a = results["by_dimension"][d]["best_arch"]
        h = results["by_dimension"][d]["architectures"][a]["history"]
        if h:
            hist.append(h)
            labels.append(f"k={d} ({a})")
    plots.superimposed_curves(
        hist, labels, "Task-1: training loss vs epoch (best arch per dimension)",
        os.path.join(pca_dir, "task1_loss_curves.png"))

    # Task-1d comparison against A3.
    gap = 100 * (results["best_test_accuracy"] - A3_TEST_ACC)
    results["comparison_to_a3"] = {
        "a3_test_acc": A3_TEST_ACC,
        "task1_best_test_acc": results["best_test_accuracy"],
        "gap_percentage_points": gap,
        "verdict": ("beats A3" if gap > 0 else
                    "matches A3" if abs(gap) < 1e-9 else
                    "below A3"),
        "caveat": ("A3's best used batch_size=1; Task-1 uses full batch, so "
                   "this comparison is of representation quality, not of "
                   "identical training conditions."),
    }
    print(f"  vs A3 ({100 * A3_TEST_ACC:.2f}%): "
          f"{results['comparison_to_a3']['verdict']} "
          f"({gap:+.2f} pp)")

    atomic_write_text(json.dumps(_jsonable(results), indent=2, default=float),
                      os.path.join(outdir, "task1.json"))
    _update_selection(outdir, results)
    print(f"\nwrote {os.path.join(outdir, 'task1.json')}")
    return results


def _jsonable(obj):
    """
    Recursively convert tensors and numpy scalars into JSON-safe Python.

    `default=float` in json.dumps only rescues values the encoder cannot handle
    at all; a multi-element tensor still raises, because no single float
    represents it. Keeping model_state out of the JSON entirely is the real fix
    (checkpoints already persist those), but any stray tensor must not be allowed
    to kill a completed run at the final write, so walk the structure properly.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() > 1 \
            else obj.detach().cpu().item()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: (thin_history(v) if k in ("history", "train_loss")
                else _jsonable(v))
            for k, v in obj.items() if k != "model_state"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _update_selection(outdir, entry):
    """
    Merge this task's outcome into selection.json.

    Tasks 5 and 6 read this file to learn which bottleneck size and which
    architecture won upstream. Writing it per task keeps the dependency explicit
    instead of passing it through memory in one monolithic run.
    """
    path = os.path.join(outdir, "selection.json")
    sel = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                sel = json.load(fh)
        except (json.JSONDecodeError, OSError):
            sel = {}
    sel[entry["task"]] = {
        "best_dimension": entry.get("best_dimension"),
        "best_arch": entry.get("best_dimension_arch"),
        "best_test_accuracy": entry.get("best_test_accuracy"),
    }
    atomic_write_text(json.dumps(sel, indent=2, default=float), path)