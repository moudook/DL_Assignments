"""
task5_denoising.py — Assignment-4 Task-5: denoising autoencoders.

A4 Task-5 requirements:
  a. 1-hidden denoising autoencoders at 20% and 40% noise. Bottleneck size is
     NOT free: "Consider number of neurons for the bottleneck (middle) layer
     based on the best test accuracy of the best reduced dimensional
     representation from 1-hidden layer autoencoder." That is Task-3's winner, so
     this reads selection.json["3"]["best_bottleneck"] rather than hardcoding a
     value. If Task-3 has not run, this raises with a clear message instead of
     silently guessing.
  b. Average reconstruction errors on train/val/test, post-training.
  c. One image per class from train/val/test, with reconstructions.
  d. Classify the reduced representation using the SAME best architecture as
     Task-3, reporting validation AND test accuracy.

Optimizer note: A4 mandates Adam for Task-2 autoencoders but says nothing about
Task-5's. Using Adam is an inference from the same model family, not an A4
requirement — the report must say so rather than imply A4 demanded it.

Noise model: masking corruption (pixels zeroed at random), implemented in
models.make_noise. Masking rather than additive Gaussian keeps values in [0,1] so
the sigmoid output layer stays matched to the target distribution. This is a
stated modeling choice, not something A4 specifies.

Writes results/task5.json and selection.json["task5"].
"""

import json
import os

import torch

import plots
from data import CLASS_NAMES, one_per_class
from evaluate import evaluate_classifier, reconstruction_grid
from models import build_denoising_autoencoder, build_classifier, count_params
from run_tracker import RunTracker, atomic_write_text, thin_history
from train import train_autoencoder, train_classifier, encode_all, MAX_EPOCHS, \
    AUTOENCODER_LR, CLASSIFIER_LR, TOL

# A4 Task-5a mandates exactly these two noise levels.
NOISE_LEVELS = [0.2, 0.4]


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


def resolve_bottleneck(outdir):
    """
    Read Task-3's winning bottleneck from selection.json.

    A4 Task-5a makes this a hard dependency, so a missing file must fail loudly.
    Guessing a value here would silently produce a wrong denoising experiment
    that still looks plausible in the report.
    """
    path = os.path.join(outdir, "selection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Task-5 depends on Task-3's best bottleneck "
            "(A4 Task-5a). Run Task-3 first.")
    with open(path, encoding="utf-8") as fh:
        sel = json.load(fh)
    t3 = sel.get("3")
    if not t3 or t3.get("best_bottleneck") is None:
        raise KeyError(
            f"selection.json has no Task-3 entry with a best_bottleneck. "
            f"Found keys: {list(sel)}. Task-5a requires Task-3's result.")
    return int(t3["best_bottleneck"]), t3.get("best_arch")


def run_task5(data, outdir="results", max_epochs=MAX_EPOCHS, device=None,
              noise_levels=NOISE_LEVELS, bottleneck=None, task3_arch=None):
    """
    Train denoising AEs at each noise level, then classify their representation
    with Task-3's winning architecture.

    bottleneck: override; None resolves from Task-3's selection.json.
    task3_arch: override; None resolves from Task-3's selection.json. A4 Task-5d
                requires "the same best architecture you have considered in
                Task-3".
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = plots.task_dirs(outdir, 5)["root"]
    dirs = plots.task_dirs(outdir, 5)

    if bottleneck is None:
        bottleneck, sel_arch = resolve_bottleneck(outdir)
        task3_arch = task3_arch or sel_arch
        print(f"  bottleneck from Task-3 selection: k={bottleneck}")
    if task3_arch is None:
        raise ValueError("task3_arch required (A4 Task-5d reuses Task-3's best arch)")

    print(f"Task-5: denoising AE, bottleneck k={bottleneck} "
          f"(from Task-3), classifier arch {task3_arch} (from Task-3)")

    results = {
        "task": 5,        "name": "Denoising autoencoders",
        "bottleneck": bottleneck,
        "bottleneck_source": "Task-3 best test accuracy (A4 Task-5a)",
        "classifier_arch": task3_arch,
        "classifier_arch_source": "Task-3 best architecture (A4 Task-5d)",
        "noise_levels": noise_levels,
        "noise_model": "masking (pixels zeroed at random)",
        "optimizer": "Adam",
        "optimizer_note": ("A4 mandates Adam for Task-2 autoencoders but is "
                           "silent for Task-5; Adam used here as an inference "
                           "from the same model family."),
        "lr": AUTOENCODER_LR,
        "tolerance": TOL,
        "max_epochs": max_epochs,
        "by_noise": {},
    }

    for noise in noise_levels:
        tag = f"noise{int(noise * 100)}"
        print(f"\n{'=' * 60}\nTask-5: {tag}% noise, k={bottleneck}\n{'=' * 60}")

        # Seeded generator: the corruption pattern must be reproducible, or the
        # reported reconstruction error for the same model is not repeatable.
        gen = torch.Generator(device=device).manual_seed(42)

        run_id = f"task5_dae_{tag}_k{bottleneck}"
        tracker = RunTracker(run_id, outdir, total_epochs=max_epochs)
        model = build_denoising_autoencoder(bottleneck, noise=noise).to(device)

        res = train_autoencoder(model, data, f"dae_{tag}_k{bottleneck}",
                                tracker=tracker, max_epochs=max_epochs,
                                device=device, noise_level=noise,
                                noise_generator=gen)

        trained = build_denoising_autoencoder(bottleneck, noise=noise).to(device)
        trained.load_state_dict(res["model_state"])

        recon = res["recon_error"]
        print(f"  recon error  train={recon['train']:.6f}  "
              f"val={recon['val']:.6f}  test={recon['test']:.6f}")

        # Task-5c: reconstruction grids per split, with originals. Also a
        # triptych per split showing what the network was FED (corrupted input),
        # since "reconstruction" alone is ambiguous for a denoising model.
        grids = {}
        for split in ("train", "val", "test"):
            idx, _ = one_per_class(data, split)
            orig, rec, _ = reconstruction_grid(trained, data, split, idx,
                                              device=device)
            grids[split] = plots.reconstruction_grid(
                orig, rec, CLASS_NAMES,
                f"Denoising AE {tag}% noise, k={bottleneck} — {split} split "
                f"(one image per class)",
                os.path.join(dirs["recon"],
                             f"task5_dae_{tag}_recon_{split}.png"))

            # Clean / corrupted / reconstructed: the noise used to build the
            # model. Sampled with a seeded generator so the figure shows the same
            # corruption the model would have seen.
            X = data[f"X_{split}"]
            sel = X[torch.as_tensor(idx, device=X.device)]
            from models import make_noise
            gen2 = torch.Generator(device=X.device).manual_seed(42)
            noisy = make_noise(sel, noise, gen2)
            with torch.no_grad():
                rec2 = trained(noisy)
            clean = sel.reshape(-1, 28, 28).cpu().numpy()
            nois = noisy.reshape(-1, 28, 28).cpu().numpy()
            recs = rec2.reshape(-1, 28, 28).cpu().numpy()
            plots.denoise_triptych(
                clean, nois, recs,
                f"Task-5: denoising AE {tag}% noise, k={bottleneck} — {split} "
                f"(clean / corrupted / reconstructed)",
                os.path.join(dirs["recon"],
                             f"task5_dae_{tag}_triptych_{split}.png"))

        # Task-5d: classify the representation with Task-3's best architecture.
        red = encode_all(trained, data)
        run_id_c = f"task5_clf_{tag}_k{bottleneck}_{task3_arch}"
        tracker_c = RunTracker(run_id_c, outdir, total_epochs=max_epochs)
        clf = build_classifier(task3_arch, input_dim=bottleneck,
                               num_classes=len(CLASS_NAMES)).to(device)
        cres = train_classifier(clf, red, f"task5_{tag}", tracker=tracker_c,
                                max_epochs=max_epochs, device=device)

        clf_trained = build_classifier(task3_arch, input_dim=bottleneck,
                                       num_classes=len(CLASS_NAMES)).to(device)
        clf_trained.load_state_dict(cres["model_state"])
        test_res = evaluate_classifier(clf_trained, red, device=device,
                                       split="test")
        train_clf_res = evaluate_classifier(clf_trained, red, device=device,
                                            split="train")

        print(f"  classifier ({task3_arch}): val={100 * cres['final_val_acc']:.2f}%  "
              f"TEST={100 * test_res['accuracy']:.2f}%")

        # Confusion matrices (raw + normalised) and per-class breakdown.
        plots.confusion_matrix(
            test_res["confusion_matrix"],
            f"Task-5: denoising AE {tag}% noise, k={bottleneck} — test "
            f"confusion matrix\ntest accuracy {100 * test_res['accuracy']:.2f}%",
            os.path.join(dirs["confusion"], f"task5_confusion_{tag}_test.png"))
        plots.confusion_matrix(
            test_res["confusion_matrix"],
            f"Task-5: denoising AE {tag}% noise, k={bottleneck} — normalised "
            f"by true class\ntest accuracy {100 * test_res['accuracy']:.2f}%",
            os.path.join(dirs["confusion"], f"task5_confusion_{tag}_test_norm.png"),
            normalize=True)
        plots.per_class_accuracy(
            test_res["per_class"],
            f"Task-5: denoising AE {tag}% noise, k={bottleneck} — per-class "
            f"test accuracy",
            os.path.join(dirs["accuracy"], f"task5_per_class_{tag}.png"))

        # Autoencoder training curve + classifier training/accuracy curves.
        plots.loss_curve(
            res["history"],
            f"Task-5: denoising AE {tag}% noise — reconstruction MSE vs epoch",
            os.path.join(dirs["training"], f"task5_dae_{tag}_loss.png"),
            ylabel="reconstruction MSE (train)")
        plots.loss_and_accuracy(
            cres["train_loss"],
            f"Task-5: {tag}% noise — classifier loss and accuracy vs epoch",
            os.path.join(dirs["training"], f"task5_clf_{tag}_loss_accuracy.png"),
            tol=TOL)
        plots.accuracy_curve(
            cres["train_loss"],
            f"Task-5: {tag}% noise — classifier accuracy vs epoch",
            os.path.join(dirs["accuracy"], f"task5_clf_{tag}_accuracy.png"))
        plots.recon_error_bars(
            {"reconstruction error": recon},
            f"Task-5: denoising AE {tag}% noise — reconstruction error per split",
            os.path.join(dirs["recon_error"], f"task5_{tag}_recon_error.png"))

        results["by_noise"][tag] = {
            "noise": noise,
            "bottleneck": bottleneck,
            "params": count_params(model),
            # Kept in the returned dict (and stripped only when serialising to
            # JSON by _jsonable) so run_all can hand the trained denoiser to
            # Task-6 without retraining it.
            "model_state": res["model_state"],
            "recon_error": recon,
            "epochs_run": res["epochs_run"],
            "stopped_early": res["stopped_early"],
            "history": res["history"],
            "grids": grids,
            # Weights stay out of the JSON. The classifier checkpoint already
            # persists them (checkpoint ~50 KB), and inlining them here bloated
            # task5.json to ~350 KB for no analytical value.
            "classifier_model_recoverable_from": "checkpoint",
            "classifier": {
                "arch": task3_arch,
                "val_acc": cres["final_val_acc"],
                "train_acc": train_clf_res["accuracy"],
                "test_acc": test_res["accuracy"],
                "epochs_run": cres["epochs_run"],
                "confusion_matrix": test_res["confusion_matrix"],
                "per_class": test_res["per_class"],
                "history": cres["train_loss"],
            },
        }

    # ── Task-5 comparison figures ────────────────────────────────────────────
    plots.recon_error_bars(
        {name: e["recon_error"] for name, e in results["by_noise"].items()},
        f"Task-5b: denoising AE reconstruction error, k={bottleneck}",
        os.path.join(dirs["recon_error"], "task5_recon_error_by_noise.png"))

    plots.dimension_bars(
        {"validation accuracy": [100 * e["classifier"]["val_acc"]
                                 for e in results["by_noise"].values()],
         "test accuracy": [100 * e["classifier"]["test_acc"]
                           for e in results["by_noise"].values()]},
        [f"{int(n * 100)}% noise" for n in noise_levels],
        f"Task-5d: classification from denoising AE representation\n"
        f"(k={bottleneck}, {task3_arch} — architecture from Task-3)",
        os.path.join(dirs["accuracy"], "task5_accuracy_by_noise.png"))

    # Reconstruction error against noise level, which is the quantity that
    # reveals the tradeoff Task-5 is about: more corruption is harder to
    # reconstruct, but forces the model to learn noise-invariant features.
    plots.dimension_bars(
        {"train": [1000 * e["recon_error"]["train"]
                   for e in results["by_noise"].values()],
         "val": [1000 * e["recon_error"]["val"]
                 for e in results["by_noise"].values()],
         "test": [1000 * e["recon_error"]["test"]
                  for e in results["by_noise"].values()]},
        [f"{int(n * 100)}% noise" for n in noise_levels],
        f"Task-5b: reconstruction error (x1000) vs noise level, k={bottleneck}",
        os.path.join(dirs["recon_error"], "task5_recon_vs_noise.png"),
        ylabel="reconstruction error x1000 (lower is better)")

    # Generalisation gap of the classifiers, per noise level.
    plots.generalisation_gap(
        {f"{int(e['noise'] * 100)}% noise":
            100 * (e["classifier"]["train_acc"] - e["classifier"]["test_acc"])
         for e in results["by_noise"].values()},
        "Task-5: train minus test accuracy, classifier generalisation by noise",
        os.path.join(dirs["comparison"], "task5_generalisation_gap.png"))

    # Superimposed AE training curves: the two noise levels must be compared
    # against each other, which is the whole point of Task-5's design.
    hist, labels = [], []
    for name, e in results["by_noise"].items():
        if e["history"]:
            hist.append(e["history"])
            labels.append(f"{name} noise")
    plots.superimposed_curves(
        hist, labels, "Task-5: denoising AE training reconstruction loss by noise",
        os.path.join(dirs["comparison"], "task5_loss_curves_by_noise.png"))
    for name, e in results["by_noise"].items():
        plots.loss_and_accuracy(
            e["classifier"]["history"],
            f"Task-5: {name} noise — classifier loss and accuracy vs epoch",
            os.path.join(dirs["comparison"],
                         f"task5_clf_{name}_loss_accuracy.png"), tol=TOL)

    best_tag = max(results["by_noise"].items(),
                   key=lambda kv: kv[1]["classifier"]["test_acc"])
    results["best_noise"] = float(best_tag[1]["noise"])
    results["best_test_accuracy"] = best_tag[1]["classifier"]["test_acc"]
    print(f"\n  BEST noise level by test accuracy: {best_tag[1]['noise']:.0%}")

    atomic_write_text(json.dumps(_jsonable(results), indent=2, default=float),
                      os.path.join(outdir, "task5.json"))

    path = os.path.join(outdir, "selection.json")
    sel = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                sel = json.load(fh)
        except (json.JSONDecodeError, OSError):
            sel = {}
    sel["5"] = {"best_noise": results["best_noise"],
                "bottleneck": bottleneck,
                "classifier_arch": task3_arch,
                "best_test_accuracy": results["best_test_accuracy"]}
    atomic_write_text(json.dumps(_jsonable(sel), indent=2, default=float), path)

    print(f"\nwrote {os.path.join(outdir, 'task5.json')}")
    return results