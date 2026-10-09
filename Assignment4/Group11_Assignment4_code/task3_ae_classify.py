"""
task3_ae_classify.py / task4_ae_classify.py — classification from autoencoder
bottleneck representations.

Task-3 uses the 1-hidden AE compressed representations; Task-4 uses the 3-hidden
AE's (which A4 Task-4 calls the "2-hidden layer autoencoder" — same model, counted
from the other end). The two tasks are structurally identical, so they share one
implementation parameterised by `kind`.

A4 requirements for both:
  a. Save the output of the middle layer for train, val and test, for EVERY
     bottleneck. Different autoencoders give different representations, so the
     experiment is repeated per dimension. -> encode_all() from train.py.
  b. For each representation: FCNN classifier, validation accuracy for all four
     architectures, then test accuracy + confusion matrix for the best
     architecture SELECTED ON VALIDATION accuracy. Same architectures as Task-1.
  c. Observe the best reduced representation by test accuracy.
  d. Compare against A3, Task-1, and (for Task-4) Task-3.

Reuses the SAME four classifier architectures as Task-1, per A4's "consider the
same architecture you have considered in Task-1". That also keeps capacity
identical across PCA and AE representations, so a difference in accuracy is
attributable to the representation rather than to classifier size.

Writes results/task3.json / task4.json and selection.json.
"""

import json
import os

import torch

import plots
from data import CLASS_NAMES
from evaluate import evaluate_classifier
from models import BOTTLENECKS, CLASSIFIER_ARCHS, build_classifier, \
    build_autoencoder
from run_tracker import RunTracker, atomic_write_text, thin_history
from train import (train_classifier, train_autoencoder, encode_all,
                    selection_bias_report, MAX_EPOCHS, AUTOENCODER_LR,
                    CLASSIFIER_LR, TOL)

A3_TEST_ACC = 0.9876
A3_VAL_ACC = 0.9884


def _jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() > 1 \
            else obj.detach().cpu().item()
    if isinstance(obj, dict):
        return {k: (thin_history(v) if k in ("history", "train_loss")
                else _jsonable(v))
            for k, v in obj.items() if k != "model_state"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_ae_classify(data, kind, task_id, outdir="results",
                    bottlenecks=BOTTLENECKS, max_epochs=MAX_EPOCHS,
                    device=None, archs=None, compare_to=None,
                    ae_epochs=None, ae_state=None):
    """
    Shared Task-3 / Task-4 implementation.

    kind:       "1hidden" for Task-3, "3hidden" for Task-4
    task_id:    3 or 4, used in filenames and result keys
    ae_epochs:  autoencoder training budget. None reuses max_epochs. Task-3/4
                TRAIN their own autoencoders (A4 requires the experiment per
                bottleneck), but ae_state lets an orchestrator pass in weights
                already trained by Task-2 so they are not trained twice.
    ae_state:   optional dict mapping bottleneck -> state_dict, to reuse Task-2's
                trained autoencoders instead of retraining.

    compare_to: optional dict of prior results to compare against, recorded into
                the output for the report's comparison tables.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archs = archs or list(CLASSIFIER_ARCHS)
    ae_epochs = ae_epochs or max_epochs
    tag = "task3" if task_id == 3 else "task4"
    out_dir = plots.task_dirs(outdir, task_id)["root"]
    dirs = plots.task_dirs(outdir, task_id)

    label = "1-hidden AE" if kind == "1hidden" else "3-hidden AE"
    results = {
        "task": task_id,
        "name": f"Classification from {label} compressed representation",
        "kind": kind,
        "bottlenecks": bottlenecks,
        "architectures": archs,
        "classifier_lr": CLASSIFIER_LR,
        "tolerance": TOL,
        "max_epochs": max_epochs,
        "batch_size": "full",
        "by_bottleneck": {},
        "compare_to": compare_to or {},
        "a3_baseline": {"val_acc": A3_VAL_ACC, "test_acc": A3_TEST_ACC},
    }

    for b in bottlenecks:
        print(f"\n{'=' * 60}\nTask-{task_id}: {label} bottleneck k={b}\n{'=' * 60}")

        state_key = f"{kind}_{b}"
        if ae_state and state_key in ae_state and ae_state[state_key] is not None:
            ae = build_autoencoder(kind, b).to(device)
            ae.load_state_dict(ae_state[state_key])
            ae_epochs_run = 0
            recon = None
            print(f"  reusing supplied autoencoder state ({state_key})")
        else:
            ae_run = f"{tag}_ae_{kind}_k{b}"
            tracker = RunTracker(ae_run, outdir, total_epochs=ae_epochs)
            ae = build_autoencoder(kind, b).to(device)
            r = train_autoencoder(ae, data, f"{kind}_k{b}", tracker=tracker,
                                  max_epochs=ae_epochs, device=device)
            ae.load_state_dict(r["model_state"])
            ae_epochs_run = r["epochs_run"]
            recon = r["recon_error"]
            print(f"  autoencoder trained {ae_epochs_run} epochs, "
                  f"test recon={recon['test']:.6f}")

        red = encode_all(ae, data)
        print(f"  representation: {tuple(red['X_train'].shape)}")

        entry = {
            "bottleneck": b,
            "ae_epochs_run": ae_epochs_run,
            "ae_recon_error": recon,
            "architectures": {},
        }

        for arch in archs:
            run_id = f"{tag}_k{b}_{arch}"
            tracker = RunTracker(run_id, outdir, total_epochs=max_epochs)
            model = build_classifier(arch, input_dim=b,
                                     num_classes=len(CLASS_NAMES)).to(device)
            res = train_classifier(model, red, f"{tag}_k{b}_{arch}",
                                   tracker=tracker, max_epochs=max_epochs,
                                   device=device)
            entry["architectures"][arch] = {
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
                    f"Task-{task_id}: {label} k={b}, {arch} — loss and accuracy",
                    os.path.join(dirs["training"],
                                 f"{tag}_k{b}_{arch}_loss_accuracy.png"),
                    tol=TOL)
                plots.accuracy_curve(
                    hist,
                    f"Task-{task_id}: {label} k={b}, {arch} — accuracy vs epoch",
                    os.path.join(dirs["accuracy"],
                                 f"{tag}_k{b}_{arch}_accuracy.png"))
                plots.loss_curve(
                    hist,
                    f"Task-{task_id}: {label} k={b}, {arch} — training loss vs epoch",
                    os.path.join(dirs["training"],
                                 f"{tag}_k{b}_{arch}_loss.png"))

        collapsed = [a for a in entry["architectures"]
                     if entry["architectures"][a].get("degenerate")]
        survivors = {a: v for a, v in entry["architectures"].items()
                     if a not in collapsed}
        entry["degenerate_archs"] = collapsed
        if collapsed:
            print(f"  excluding degenerate arch(s): {collapsed}")
        if not survivors:
            print("  WARNING every architecture degenerate; selection unreliable")
            survivors = entry["architectures"]

        best_arch = max(
            survivors.items(),
            key=lambda kv: (kv[1]["val_acc"], -kv[1]["epochs_run"]),
        )[0]
        best = entry["architectures"][best_arch]

        model = build_classifier(best_arch, input_dim=b,
                                 num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(best["model_state"])

        test_res = evaluate_classifier(model, red, device=device, split="test")
        train_res = evaluate_classifier(model, red, device=device, split="train")

        entry["best_arch"] = best_arch
        entry["test_accuracy"] = test_res["accuracy"]
        entry["architectures"][best_arch]["test_accuracy_all_archs"] = \
            test_res["accuracy"]
        entry["train_accuracy"] = train_res["accuracy"]
        entry["test_confusion_matrix"] = test_res["confusion_matrix"]
        entry["test_per_class"] = test_res["per_class"]

        print(f"  best (on val): {best_arch} val={100 * best['val_acc']:.2f}%  "
              f"TEST={100 * test_res['accuracy']:.2f}%")

        plots.confusion_matrix(
            test_res["confusion_matrix"],
            f"Task-{task_id}: {label} k={b}, {best_arch} — test confusion matrix\n"
            f"test accuracy {100 * test_res['accuracy']:.2f}%",
            os.path.join(dirs["confusion"],
                         f"{tag}_k{b}_{best_arch}_confusion_test.png"))
        plots.confusion_matrix(
            test_res["confusion_matrix"],
            f"Task-{task_id}: {label} k={b}, {best_arch} — normalised by true class\n"
            f"test accuracy {100 * test_res['accuracy']:.2f}%",
            os.path.join(dirs["confusion"],
                         f"{tag}_k{b}_{best_arch}_confusion_test_norm.png"),
            normalize=True)
        plots.per_class_accuracy(
            test_res["per_class"],
            f"Task-{task_id}: {label} k={b}, {best_arch} — per-class test accuracy",
            os.path.join(dirs["accuracy"], f"{tag}_k{b}_{best_arch}_per_class.png"))

        for arch, av in entry["architectures"].items():
            if arch == best_arch:
                continue
            am = build_classifier(arch, input_dim=b,
                                  num_classes=len(CLASS_NAMES)).to(device)
            am.load_state_dict(av["model_state"])
            ar = evaluate_classifier(am, red, device=device, split="test")
            av["test_accuracy_all_archs"] = ar["accuracy"]
            deg = " (DEGENERATE)" if av.get("degenerate") else ""
            plots.confusion_matrix(
                ar["confusion_matrix"],
                f"Task-{task_id}: {label} k={b}, {arch} — test confusion matrix{deg}\n"
                f"test accuracy {100 * ar['accuracy']:.2f}%, "
                f"val {100 * av['val_acc']:.2f}% ({av['epochs_run']} epochs)",
                os.path.join(dirs["confusion"],
                             f"{tag}_k{b}_{arch}_confusion_test.png"))
            plots.per_class_accuracy(
                ar["per_class"],
                f"Task-{task_id}: {label} k={b}, {arch} — per-class accuracy{deg}",
                os.path.join(dirs["accuracy"],
                             f"{tag}_k{b}_{arch}_per_class.png"))

        results["by_bottleneck"][b] = entry

    best_b = max(results["by_bottleneck"].items(),
                 key=lambda kv: kv[1]["test_accuracy"])
    results["best_bottleneck"] = int(best_b[0])
    results["best_arch"] = best_b[1]["best_arch"]
    results["best_test_accuracy"] = best_b[1]["test_accuracy"]

    results["selection_bias"] = selection_bias_report(
        results["by_bottleneck"],
        get_test=lambda v: v["test_accuracy"],
        get_val=lambda v: v["architectures"][v["best_arch"]]["val_acc"],
    )
    _sb = results["selection_bias"]
    print(f"\n  BEST representation by test accuracy: k={results['best_bottleneck']} "
          f"({100 * results['best_test_accuracy']:.2f}%)")
    print(f"  bias: max is {_sb['max_minus_mean_pp']:+.2f} pp above the "
          f"{100 * _sb['mean_across_representations']:.2f}% mean; "
          f"validation would pick k={_sb['selected_by_validation']} "
          f"({100 * _sb['selected_by_validation_accuracy']:.2f}%)")

    plots.dimension_bars(
        {"test accuracy": [100 * results["by_bottleneck"][b]["test_accuracy"]
                           for b in bottlenecks],
         "best val accuracy": [100 * max(
             results["by_bottleneck"][b]["architectures"][a]["val_acc"]
             for a in archs) for b in bottlenecks]},
        [str(b) for b in bottlenecks],
        f"Task-{task_id}: {label} representation — accuracy vs bottleneck",
        os.path.join(dirs["comparison"], f"{tag}_accuracy_vs_bottleneck.png"),
        ref=100 * A3_TEST_ACC,
        ref_label="A3 baseline 98.76%")

    plots.dimension_bars(
        {"epochs": [results["by_bottleneck"][b]["architectures"][
            results["by_bottleneck"][b]["best_arch"]]["epochs_run"]
            for b in bottlenecks]},
        [str(b) for b in bottlenecks],
        f"Task-{task_id}: epochs to convergence vs bottleneck (best architecture)",
        os.path.join(dirs["comparison"], f"{tag}_epochs_vs_bottleneck.png"),
        ylabel="epochs (lower is cheaper)")

    plots.accuracy_heatmap(
        archs, [str(b) for b in bottlenecks],
        [[100 * results["by_bottleneck"][b]["architectures"][a]["val_acc"]
          for b in bottlenecks] for a in archs],
        f"Task-{task_id}b: validation accuracy by architecture x bottleneck",
        os.path.join(dirs["comparison"], f"{tag}_heatmap_val.png"),
        rowlabel="architecture", collabel="bottleneck k")
    plots.accuracy_heatmap(
        archs, [str(b) for b in bottlenecks],
        [[100 * results["by_bottleneck"][b]["architectures"][a].get(
            "test_accuracy_all_archs", float("nan")) for b in bottlenecks]
         for a in archs],
        f"Task-{task_id}b: test accuracy by architecture x bottleneck",
        os.path.join(dirs["comparison"], f"{tag}_heatmap_test.png"),
        rowlabel="architecture", collabel="bottleneck k")

    hist, labels = [], []
    for b in bottlenecks:
        a = results["by_bottleneck"][b]["best_arch"]
        h = results["by_bottleneck"][b]["architectures"][a]["history"]
        if h:
            hist.append(h)
            labels.append(f"k={b} ({a})")
    plots.superimposed_curves(
        hist, labels,
        f"Task-{task_id}: classifier training loss, best arch per bottleneck",
        os.path.join(dirs["comparison"], f"{tag}_loss_by_bottleneck.png"))

    for a in archs:
        hs, ls = [], []
        for b in bottlenecks:
            h = results["by_bottleneck"][b]["architectures"][a]["history"]
            if h:
                hs.append(h)
                ls.append(f"k={b}")
        if len(hs) > 1:
            plots.superimposed_curves(
                hs, ls, f"Task-{task_id}: {a} training loss across bottlenecks",
                os.path.join(dirs["comparison"],
                             f"{tag}_{a}_loss_across_bottlenecks.png"))

    plots.generalisation_gap(
        {f"k={b}": 100 * (results["by_bottleneck"][b]["train_accuracy"]
                          - results["by_bottleneck"][b]["test_accuracy"])
         for b in bottlenecks},
        f"Task-{task_id}: train minus test accuracy by bottleneck (generalisation)",
        os.path.join(dirs["comparison"], f"{tag}_generalisation_gap.png"))

    cmp_ = {"a3_test_acc": A3_TEST_ACC,
            "task_test_acc": results["best_test_accuracy"],
            "gap_vs_a3_pp": 100 * (results["best_test_accuracy"] - A3_TEST_ACC)}
    for name, prior in (compare_to or {}).items():
        if prior is not None:
            cmp_[f"gap_vs_{name}_pp"] = 100 * (
                results["best_test_accuracy"] - prior)
    cmp_["caveat"] = ("A3's best used batch_size=1; A4 uses full batch, so "
                      "comparisons are of representation quality rather than "
                      "identical training conditions.")
    results["comparison"] = cmp_
    print(f"  vs A3: {cmp_['gap_vs_a3_pp']:+.2f} pp")

    atomic_write_text(json.dumps(_jsonable(results), indent=2, default=float),
                      os.path.join(outdir, f"{tag}.json"))
    _update_selection(outdir, results)
    print(f"\nwrote {os.path.join(outdir, tag + '.json')}")
    return results


def _update_selection(outdir, results):
    path = os.path.join(outdir, "selection.json")
    sel = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                sel = json.load(fh)
        except (json.JSONDecodeError, OSError):
            sel = {}
    sel[str(results["task"])] = {
        "best_bottleneck": results.get("best_bottleneck"),
        "best_arch": results.get("best_arch"),
        "best_test_accuracy": results.get("best_test_accuracy"),
    }
    atomic_write_text(json.dumps(_jsonable(sel), indent=2, default=float), path)


def run_task3(data, **kw):
    return run_ae_classify(data, kind="1hidden", task_id=3, **kw)


def run_task4(data, **kw):
    return run_ae_classify(data, kind="3hidden", task_id=4, **kw)
