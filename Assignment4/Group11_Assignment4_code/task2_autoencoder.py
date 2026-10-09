"""
task2_autoencoder.py — Assignment-4 Task-2: autoencoder image reconstruction.

A4 Task-2 requirements, mapped here:
  a. Autoencoders with 1-hidden and 3-hidden architectures. 400 neurons in the
     first and third layers of the 3-hidden one. Bottleneck 32/64/128/256.
     -> models.build_autoencoder handles the mandated geometry.
  b. Train with Adam (MANDATED). Bottleneck always linear. Sigmoid on remaining
     hidden layers. -> models.Autoencoder.
  c. Average reconstruction errors for train, val and test, computed AFTER
     training. -> train.train_autoencoder returns these post-training.
  d. One image per class from train, val and test, reconstructed, alongside the
     originals. -> reconstruction_grid per architecture per split.

Note on architecture naming: A4 says "1-hidden layer" and "3-hidden layer"
architectures in Task-2a, but Task-4 calls the same 3-hidden model a "2-hidden
layer autoencoder". These are the SAME model counted from opposite ends - one
hidden layer before the bottleneck, one after. Both labels appear in the report
so the reader is not confused; internally they are kind="3hidden".

Writes results/task2.json and selection.json["task2"].
"""

import json
import os

import torch

import plots
from data import CLASS_NAMES, one_per_class
from evaluate import reconstruction_grid
from models import BOTTLENECKS, build_autoencoder, count_params
from run_tracker import RunTracker, atomic_write_text, thin_history
from train import train_autoencoder, MAX_EPOCHS, AUTOENCODER_LR, TOL

AE_KINDS = ["1hidden", "3hidden"]

KIND_LABEL = {"1hidden": "1-hidden AE", "3hidden": "3-hidden AE (A4 Task-4: 2-hidden)"}


def run_task2(data, outdir="results", bottlenecks=BOTTLENECKS, max_epochs=MAX_EPOCHS,
              device=None, kinds=AE_KINDS):
    """
    Train every (depth, bottleneck) autoencoder and report reconstruction error.

    Returns a results dict, also written to <outdir>/task2.json.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = plots.task_dirs(outdir, 2)

    results = {
        "task": 2,
        "name": "Autoencoder reconstruction",
        "kinds": kinds,
        "bottlenecks": bottlenecks,
        "optimizer": "Adam",
        "optimizer_mandated": True,
        "lr": AUTOENCODER_LR,
        "loss": "MSE",
        "tolerance": TOL,
        "stopping_rule": "plateau on windowed best loss (relative)",
        "max_epochs": max_epochs,
        "batch_size": "full",
        "activation": "logistic sigmoid",
        "bottleneck_activation": "linear (A4 mandate)",
        "by_model": {},
    }

    for kind in kinds:
        for b in bottlenecks:
            run_id = f"task2_{kind}_k{b}"
            print(f"\n{'=' * 60}\nTask-2: {KIND_LABEL[kind]} k={b}\n{'=' * 60}")

            tracker = RunTracker(run_id, outdir, total_epochs=max_epochs)
            model = build_autoencoder(kind, b).to(device)

            res = train_autoencoder(model, data, f"{kind}_k{b}", tracker=tracker,
                                    max_epochs=max_epochs, device=device)

            trained = build_autoencoder(kind, b).to(device)
            trained.load_state_dict(res["model_state"])

            recon = res["recon_error"]
            print(f"  recon error  train={recon['train']:.6f}  "
                  f"val={recon['val']:.6f}  test={recon['test']:.6f}")
            print(f"  epochs={res['epochs_run']}  "
                  f"early_stop={res['stopped_early']}  "
                  f"params={count_params(model):,}")

            grid_paths = {}
            for split in ("train", "val", "test"):
                idx, _ = one_per_class(data, split)
                orig, rec, _ = reconstruction_grid(trained, data, split, idx,
                                                  device=device)
                p = plots.reconstruction_grid(
                    orig, rec, CLASS_NAMES,
                    f"{KIND_LABEL[kind]}, bottleneck k={b} — {split} split "
                    f"(one image per class)",
                    os.path.join(dirs["recon"],
                                 f"task2_{kind}_k{b}_recon_{split}.png"))
                grid_paths[split] = p

            plots.loss_and_accuracy(
                res["history"],
                f"Task-2: {KIND_LABEL[kind]} k={b} — reconstruction loss vs epoch",
                os.path.join(dirs["training"],
                             f"task2_{kind}_k{b}_loss_accuracy.png"),
                tol=TOL)
            plots.loss_curve(
                res["history"],
                f"Task-2: {KIND_LABEL[kind]} k={b} — reconstruction MSE vs epoch",
                os.path.join(dirs["training"],
                             f"task2_{kind}_k{b}_loss.png"),
                ylabel="reconstruction MSE (train)")

            plots.recon_error_bars(
                {"reconstruction error": recon},
                f"Task-2: {KIND_LABEL[kind]} k={b} — reconstruction error per split",
                os.path.join(dirs["recon_error"],
                             f"task2_{kind}_k{b}_recon_error.png"))

            idx, _ = one_per_class(data, "test")
            o, r, _ = reconstruction_grid(trained, data, "test", [idx[0]],
                                          device=device)
            plots.reconstruction_grid(
                o, r, [CLASS_NAMES[0]],
                f"Task-2: {KIND_LABEL[kind]} k={b} — reconstruction of a single "
                f"test digit '{CLASS_NAMES[0]}'",
                os.path.join(dirs["recon"],
                             f"task2_{kind}_k{b}_single_digit.png"))

            results["by_model"][f"{kind}_{b}"] = {
                "kind": kind,
                "bottleneck": b,
                "params": count_params(model),
                "recon_error": recon,
                "epochs_run": res["epochs_run"],
                "stopped_early": res["stopped_early"],
                "stopping_rule": res.get("stopping_rule"),
                "history": res["history"],
                "grids": grid_paths,
                "model_state": res["model_state"],
            }

    plots.recon_error_bars(
        {name: e["recon_error"] for name, e in results["by_model"].items()},
        "Task-2c: reconstruction error per split, all architectures",
        os.path.join(dirs["recon_error"], "task2_recon_error_all.png"))

    for kind in kinds:
        subset = {f"k={b}": results["by_model"][f"{kind}_{b}"]["recon_error"]
                  for b in bottlenecks}
        plots.recon_error_bars(
            subset, f"Task-2c: {KIND_LABEL[kind]} — reconstruction error vs bottleneck",
            os.path.join(dirs["recon_error"], f"task2_recon_error_{kind}.png"))
        plots.dimension_bars(
            {"train": [1000 * results["by_model"][f"{kind}_{b}"]["recon_error"]["train"]
                       for b in bottlenecks],
             "val": [1000 * results["by_model"][f"{kind}_{b}"]["recon_error"]["val"]
                     for b in bottlenecks],
             "test": [1000 * results["by_model"][f"{kind}_{b}"]["recon_error"]["test"]
                      for b in bottlenecks]},
            [str(b) for b in bottlenecks],
            f"Task-2c: {KIND_LABEL[kind]} — reconstruction error (x1000) vs bottleneck",
            os.path.join(dirs["recon_error"],
                         f"task2_recon_vs_bottleneck_{kind}.png"),
            ylabel="reconstruction error x1000 (lower is better)")

    plots.dimension_bars(
        {"1-hidden AE": [1000 * results["by_model"][f"1hidden_{b}"]["recon_error"]["test"]
                         for b in bottlenecks],
         "3-hidden AE": [1000 * results["by_model"][f"3hidden_{b}"]["recon_error"]["test"]
                         for b in bottlenecks]},
        [str(b) for b in bottlenecks],
        "Task-2c: test reconstruction error (x1000), 1-hidden vs 3-hidden",
        os.path.join(dirs["recon_error"],
                     "task2_depth_comparison.png"),
        ylabel="reconstruction error x1000 (lower is better)")

    for kind in kinds:
        plots.dimension_bars(
            {"params (k)": [results["by_model"][f"{kind}_{b}"]["params"] / 1000
                            for b in bottlenecks],
             "test recon (x1000)": [1000 * results["by_model"][f"{kind}_{b}"]["recon_error"]["test"]
                                    for b in bottlenecks]},
            [str(b) for b in bottlenecks],
            f"Task-2: {KIND_LABEL[kind]} — parameter count vs reconstruction error",
            os.path.join(dirs["recon_error"],
                         f"task2_{kind}_params_vs_error.png"))

    for kind in kinds:
        hs, ls = [], []
        for b in bottlenecks:
            h = results["by_model"][f"{kind}_{b}"]["history"]
            if h:
                hs.append(h)
                ls.append(f"k={b}")
        plots.superimposed_curves(
            hs, ls,
            f"Task-2: {KIND_LABEL[kind]} — reconstruction loss vs epoch, all bottlenecks",
            os.path.join(dirs["comparison"],
                         f"task2_{kind}_loss_across_bottlenecks.png"),
            ylabel="reconstruction MSE (train)")

    plots.generalisation_gap(
        {name: 1000 * (e["recon_error"]["train"] - e["recon_error"]["test"])
         for name, e in results["by_model"].items()},
        "Task-2: reconstruction generalisation gap (train - test, x1000)",
        os.path.join(dirs["comparison"], "task2_generalisation_gap.png"),
        ylabel="train - test recon error (x1000)")

    for kind in kinds:
        best = min((results["by_model"][f"{kind}_{b}"] for b in bottlenecks),
                   key=lambda e: e["recon_error"]["test"])
        results.setdefault("best_by_kind", {})[kind] = {
            "bottleneck": best["bottleneck"],
            "test_recon_error": best["recon_error"]["test"],
        }
        print(f"\n  best bottleneck ({kind}): k={best['bottleneck']} "
              f"test_recon={best['recon_error']['test']:.6f}")

    atomic_write_text(json.dumps(_jsonable(results), indent=2, default=float),
                      os.path.join(outdir, "task2.json"))
    _update_selection(outdir, results)
    print(f"\nwrote {os.path.join(outdir, 'task2.json')}")
    return results


def _jsonable(obj):
    """Tensors/numpy -> JSON-safe; model_state dropped (checkpoints hold it)."""
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


def _update_selection(outdir, entry):
    path = os.path.join(outdir, "selection.json")
    sel = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                sel = json.load(fh)
        except (json.JSONDecodeError, OSError):
            sel = {}
    sel["2"] = entry.get("best_by_kind", {})
    atomic_write_text(json.dumps(_jsonable(sel), indent=2, default=float), path)
