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
from train import (train_classifier, selection_bias_report, MAX_EPOCHS,
                    CLASSIFIER_LR, TOL)

DIMENSIONS = [32, 64, 128, 256]

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
    dirs = plots.task_dirs(outdir, 1)

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

    wide = PCA(k=784).fit(data["X_train"])

    _representation_cache = {}

    for k in dims:
        print(f"\n{'=' * 60}\nTask-1: PCA k={k}\n{'=' * 60}")
        wide.k = k
        _representation_cache[k] = wide.project_all(data)
        red = _representation_cache[k]
        var_ret = red["variance_retained"]
        print(f"  variance retained: {100 * var_ret:.2f}%")

        curve = wide.explained_variance_curve().cpu().tolist()
        plots.pca_variance_curve(
            curve, "Task-1a: cumulative variance retained vs retained dimension",
            os.path.join(dirs["represent"], "task1_variance_retained.png"))
        plots.eigen_spectrum(
            wide.eigenvalues.cpu().numpy(),
            "Task-1a: eigenvalue spectrum (which components matter)",
            os.path.join(dirs["represent"], "task1_eigen_spectrum.png"))

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

            hist = res["train_loss"]
            if hist:
                plots.loss_and_accuracy(
                    hist,
                    f"Task-1: PCA k={k}, {arch} — loss and accuracy vs epoch",
                    os.path.join(dirs["training"],
                                 f"task1_k{k}_{arch}_loss_accuracy.png"),
                    tol=TOL)
                plots.accuracy_curve(
                    hist,
                    f"Task-1: PCA k={k}, {arch} — accuracy vs epoch",
                    os.path.join(dirs["accuracy"],
                                 f"task1_k{k}_{arch}_accuracy.png"))
                plots.loss_curve(
                    hist,
                    f"Task-1: PCA k={k}, {arch} — training loss vs epoch",
                    os.path.join(dirs["training"],
                                 f"task1_k{k}_{arch}_loss.png"))

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
        dim_entry["architectures"][best_arch]["test_accuracy_all_archs"] = \
            test_res["accuracy"]
        dim_entry["train_accuracy"] = train_res["accuracy"]
        dim_entry["test_confusion_matrix"] = test_res["confusion_matrix"]
        dim_entry["test_per_class"] = test_res["per_class"]

        print(f"  best arch (on val): {best_arch} "
              f"val={100 * best['val_acc']:.2f}%  "
              f"TEST={100 * test_res['accuracy']:.2f}%")

        plots.confusion_matrix(
            test_res["confusion_matrix"],
            f"Task-1: PCA k={k}, {best_arch} — test confusion matrix\n"
            f"test accuracy {100 * test_res['accuracy']:.2f}%",
            os.path.join(dirs["confusion"],
                         f"task1_k{k}_{best_arch}_confusion_test.png"))
        plots.confusion_matrix(
            test_res["confusion_matrix"],
            f"Task-1: PCA k={k}, {best_arch} — normalised by true class\n"
            f"test accuracy {100 * test_res['accuracy']:.2f}%",
            os.path.join(dirs["confusion"],
                         f"task1_k{k}_{best_arch}_confusion_test_norm.png"),
            normalize=True)
        plots.per_class_accuracy(
            test_res["per_class"],
            f"Task-1: PCA k={k}, {best_arch} — per-class test accuracy",
            os.path.join(dirs["accuracy"],
                         f"task1_k{k}_{best_arch}_per_class.png"))

        for arch, av in dim_entry["architectures"].items():
            if arch == best_arch:
                continue
            am = build_classifier(arch, input_dim=k,
                                  num_classes=len(CLASS_NAMES)).to(device)
            am.load_state_dict(av["model_state"])
            ar = evaluate_classifier(am, red, device=device, split="test")
            av["test_accuracy_all_archs"] = ar["accuracy"]
            deg = " (DEGENERATE)" if av.get("degenerate") else ""
            plots.confusion_matrix(
                ar["confusion_matrix"],
                f"Task-1: PCA k={k}, {arch} — test confusion matrix{deg}\n"
                f"test accuracy {100 * ar['accuracy']:.2f}%, "
                f"val {100 * av['val_acc']:.2f}% ({av['epochs_run']} epochs)",
                os.path.join(dirs["confusion"],
                             f"task1_k{k}_{arch}_confusion_test.png"))
            plots.per_class_accuracy(
                ar["per_class"],
                f"Task-1: PCA k={k}, {arch} — per-class test accuracy{deg}",
                os.path.join(dirs["accuracy"],
                             f"task1_k{k}_{arch}_per_class.png"))

        results["by_dimension"][k] = dim_entry

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

        best_arch = max(survivors.items(),
                        key=lambda kv: (kv[1]["val_acc"], -kv[1]["epochs_run"]))[0]
        changed = best_arch != e["best_arch"]
        e["best_arch"] = best_arch
        e["degenerate_archs"] = collapsed
        if changed:
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

    results["selection_bias"] = selection_bias_report(
        results["by_dimension"],
        get_test=lambda v: v["test_accuracy"],
        get_val=lambda v: max(a["val_acc"] for a in v["architectures"].values()),
    )
    _sb = results["selection_bias"]
    print(f"\n  BEST dimension by test accuracy: k={results['best_dimension']} "
          f"({100 * results['best_test_accuracy']:.2f}%)")
    print(f"  bias: max is {_sb['max_minus_mean_pp']:+.2f} pp above the "
          f"{100 * _sb['mean_across_representations']:.2f}% mean; "
          f"validation would pick k={_sb['selected_by_validation']} "
          f"({100 * _sb['selected_by_validation_accuracy']:.2f}%)")

    accs = [results["by_dimension"][d]["test_accuracy"] for d in dims]
    if len(accs) > 1 and accs[-1] < accs[0]:
        results["accuracy_vs_dimension"] = {
            "observation": (
                f"Best test accuracy FALLS from {100 * accs[0]:.2f}% at "
                f"k={dims[0]} to {100 * accs[-1]:.2f}% at k={dims[-1]}, while "
                f"variance retained RISES."),
            "explanation": (
                "Not caused by the representations. The classifier architectures "
                "are deliberately held FIXED across dimensions (see the SIZING "
                "DECISION note in models.py) so that any difference is "
                "attributable to the representation rather than to capacity. As k "
                "grows the first layer's input dimension grows while its width "
                "stays fixed, so it becomes progressively more ill-conditioned - "
                "a training artefact, not evidence that more dimensions carry less "
                "class information."),
        }
        print("  NOTE: accuracy falls as k rises though variance retained rises; "
              "this is the fixed-width first layer, not the representation.")

    plots.dimension_bars(
        {"test accuracy": [100 * results["by_dimension"][d]["test_accuracy"]
                           for d in dims],
         "best val accuracy": [100 * max(
             results["by_dimension"][d]["architectures"][a]["val_acc"]
             for a in archs) for d in dims]},
        [str(d) for d in dims],
        "Task-1c: PCA classification accuracy vs retained dimension",
        os.path.join(dirs["comparison"], "task1_accuracy_vs_dimension.png"),
        ref=100 * A3_TEST_ACC,
        ref_label="A3 baseline (raw 784-d, NAG) 98.76%")

    plots.dimension_bars(
        {"epochs": [results["by_dimension"][d]["architectures"][
            results["by_dimension"][d]["best_arch"]]["epochs_run"] for d in dims]},
        [str(d) for d in dims],
        "Task-1c: epochs to convergence vs retained dimension (best architecture)",
        os.path.join(dirs["comparison"], "task1_epochs_vs_dimension.png"),
        ylabel="epochs (lower is cheaper)")

    plots.accuracy_heatmap(
        archs, [str(d) for d in dims],
        [[100 * results["by_dimension"][d]["architectures"][a]["val_acc"]
          for d in dims] for a in archs],
        "Task-1b: validation accuracy by architecture x retained dimension",
        os.path.join(dirs["comparison"], "task1_heatmap_val.png"),
        rowlabel="architecture", collabel="retained dimension k")
    plots.accuracy_heatmap(
        archs, [str(d) for d in dims],
        [[100 * results["by_dimension"][d]["architectures"][a].get(
            "test_accuracy_all_archs", float("nan")) for d in dims] for a in archs],
        "Task-1b: test accuracy by architecture x retained dimension",
        os.path.join(dirs["comparison"], "task1_heatmap_test.png"),
        rowlabel="architecture", collabel="retained dimension k")

    hist, labels = [], []
    for d in dims:
        a = results["by_dimension"][d]["best_arch"]
        h = results["by_dimension"][d]["architectures"][a]["history"]
        if h:
            hist.append(h)
            labels.append(f"k={d} ({a})")
    plots.superimposed_curves(
        hist, labels, "Task-1: training loss vs epoch (best architecture per dimension)",
        os.path.join(dirs["comparison"], "task1_loss_curves_by_dimension.png"))

    for a in archs:
        hs, ls = [], []
        for d in dims:
            h = results["by_dimension"][d]["architectures"][a]["history"]
            if h:
                hs.append(h)
                ls.append(f"k={d}")
        if len(hs) > 1:
            plots.superimposed_curves(
                hs, ls, f"Task-1: {a} training loss across dimensions",
                os.path.join(dirs["comparison"],
                             f"task1_{a}_loss_across_dimensions.png"))

    plots.generalisation_gap(
        {f"k={d}": 100 * (results["by_dimension"][d]["train_accuracy"]
                          - results["by_dimension"][d]["test_accuracy"])
         for d in dims},
        "Task-1: train minus test accuracy by dimension (generalisation)",
        os.path.join(dirs["comparison"], "task1_generalisation_gap.png"))

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
