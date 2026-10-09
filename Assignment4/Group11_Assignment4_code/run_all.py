"""
run_all.py — orchestrator for Assignment-4.

Runs Tasks 1 through 6 in order, because they are genuinely dependent:

    Task-1 (PCA)         ─┐
    Task-2 (AEs)         ─┤ independent
                          │
    Task-3 (1-hidden AE) ──┼─► Task-3's winning bottleneck ─► Task-5's bottleneck
    Task-4 (3-hidden AE) ──┘                                └─► Task-6's bottleneck
    Task-5 (denoising)   ───► its two AEs ─────────────────────► Task-6
    Task-6 (weights)     ─── needs Task-3 and Task-5

A4 states these dependencies explicitly (Task-5a's bottleneck comes from
Task-3's result; Task-6a says "best compressed representation in one hidden layer
autoencoder"), so the pipeline is sequential by requirement, not by convenience.
Each task writes selection.json entries the next one reads, which keeps the
dependency visible in the artifacts instead of hidden in a long function.

Autoencoders are trained ONCE in Task-2 and their weights are reused by Tasks 3,
4 and 6, rather than retrained per downstream task. That is a pure efficiency
choice: an autoencoder is deterministic given seed 42, so Task-3's "present
training data to each of the encoders built" is satisfied by the encoder Task-2
already built. Pass --retrain-aes to force retraining if you want to verify that.

Usage:
    python run_all.py                      # everything
    python run_all.py --only 1 2 3         # selected tasks (dependencies auto-included)
    python run_all.py --quick              # tiny epoch budget, for a smoke test
    python run_all.py --max-epochs 2000
    python run_all.py --outdir results
"""

import argparse
import json
import os
import sys
import time

import torch

from data import load_splits, resolve_data_dir
from models import BOTTLENECKS, CLASSIFIER_ARCHS
from run_tracker import atomic_write_text
from train import MAX_EPOCHS

TASKS = {
    1: "PCA dimension reduction",
    2: "Autoencoder reconstruction",
    3: "Classification from 1-hidden AE representation",
    4: "Classification from 3-hidden AE representation",
    5: "Denoising autoencoders",
    6: "Weight visualisation",
}

# Task N implies these must have run first.
DEPENDENCIES = {1: [], 2: [], 3: [2], 4: [2], 5: [3], 6: [3, 5]}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Run the Assignment-4 pipeline (Tasks 1-6).")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data", default="Group_11")
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    choices=sorted(TASKS),
                    help="run only these tasks (dependencies added automatically)")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--quick", action="store_true",
                    help="15-epoch budget for a smoke test of the whole pipeline")
    ap.add_argument("--dims", nargs="*", type=int, default=BOTTLENECKS)
    ap.add_argument("--archs", nargs="*", default=None,
                    help=f"classifier archs (default all of {list(CLASSIFIER_ARCHS)})")
    ap.add_argument("--retrain-aes", action="store_true",
                    help="retrain autoencoders per task instead of reusing Task-2")
    args = ap.parse_args()

    if args.quick:
        args.max_epochs = 15
    archs = args.archs or list(CLASSIFIER_ARCHS)
    dims = args.dims

    # Expand --only with the dependencies each requested task needs.
    if args.only:
        requested = set()
        stack = list(args.only)
        while stack:
            t = stack.pop()
            if t in requested:
                continue
            requested.add(t)
            stack.extend(DEPENDENCIES.get(t, []))
        todo = sorted(requested)
    else:
        todo = sorted(TASKS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Assignment 4 — Group 11 — full pipeline")
    print("=" * 70)
    print(f"device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"torch       : {torch.__version__}")
    print(f"max epochs  : {args.max_epochs}" + ("  (QUICK)" if args.quick else ""))
    print(f"dimensions  : {dims}")
    print(f"architectures: {archs}")
    print(f"tasks       : {todo}")
    print("=" * 70, flush=True)

    os.makedirs(args.outdir, exist_ok=True)

    # Create the whole figure tree BEFORE training starts. Every task, every
    # subfolder, even ones a run may never reach, so the structure is browsable
    # from the first moment and the numbered layout is guaranteed present.
    import plots
    plots.prepare_all_dirs(args.outdir, sorted(TASKS))

    # Record the configuration alongside the results so every number in the
    # report is traceable to the run that produced it.
    atomic_write_text(json.dumps({
        "tasks": todo,
        "max_epochs": args.max_epochs,
        "quick": args.quick,
        "dimensions": dims,
        "architectures": archs,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "data_dir": resolve_data_dir(args.data),
        "batch_size": "full",
    }, indent=2), os.path.join(args.outdir, "run_config.json"))

    log(f"loading dataset from {resolve_data_dir(args.data)}")
    data = load_splits(args.data, device=device)
    log(f"train {tuple(data['X_train'].shape)}  "
        f"val {tuple(data['X_val'].shape)}  test {tuple(data['X_test'].shape)}")

    # Imported here so a syntax error in a later module does not block the early
    # tasks, and so the imports happen after the dataset is on the GPU.
    from task1_pca import run_task1
    from task2_autoencoder import run_task2
    from task3_ae_classify import run_task3, run_task4
    from task5_denoising import run_task5
    from task6_weights import run_task6

    t_start = time.time()
    ae_states = {}          # "kind_b" -> state_dict, from Task-2
    denoise_states = {}     # "noise20"/"noise40" -> state_dict, from Task-5
    summary = {}

    for t in todo:
        log(f"{'#' * 70}")
        log(f"# TASK {t}: {TASKS[t]}")
        log(f"{'#' * 70}")
        t0 = time.time()

        if t == 1:
            r = run_task1(data, outdir=args.outdir, dims=dims,
                          max_epochs=args.max_epochs, device=device,
                          archs=archs)
            summary["task1"] = {
                "best_dimension": r["best_dimension"],
                "best_test_accuracy": r["best_test_accuracy"],
                "vs_a3_pp": r["comparison_to_a3"]["gap_percentage_points"],
            }

        elif t == 2:
            r = run_task2(data, outdir=args.outdir, bottlenecks=dims,
                          max_epochs=args.max_epochs, device=device)
            if not args.retrain_aes:
                ae_states = {k: v["model_state"]
                             for k, v in r["by_model"].items()}
                log(f"cached {len(ae_states)} autoencoder states for Tasks 3/4/6")
            summary["task2"] = {k: v["recon_error"]["test"]
                                for k, v in r["by_model"].items()}

        elif t == 3:
            prior = {}
            t1p = os.path.join(args.outdir, "task1.json")
            if os.path.exists(t1p):
                with open(t1p, encoding="utf-8") as fh:
                    prior["task1"] = json.load(fh).get("best_test_accuracy")
            r = run_task3(data, outdir=args.outdir, bottlenecks=dims,
                          max_epochs=args.max_epochs, device=device,
                          archs=archs, compare_to=prior,
                          ae_epochs=args.max_epochs,
                          ae_state=ae_states or None)
            summary["task3"] = {
                "best_bottleneck": r["best_bottleneck"],
                "best_test_accuracy": r["best_test_accuracy"],
                "gap_vs_a3_pp": r["comparison"]["gap_vs_a3_pp"],
            }

        elif t == 4:
            prior = {}
            for name, fname in (("task1", "task1.json"), ("task3", "task3.json")):
                p = os.path.join(args.outdir, fname)
                if os.path.exists(p):
                    with open(p, encoding="utf-8") as fh:
                        prior[name] = json.load(fh).get("best_test_accuracy")
            r = run_task4(data, outdir=args.outdir, bottlenecks=dims,
                          max_epochs=args.max_epochs, device=device,
                          archs=archs, compare_to=prior,
                          ae_epochs=args.max_epochs,
                          ae_state=ae_states or None)
            summary["task4"] = {
                "best_bottleneck": r["best_bottleneck"],
                "best_test_accuracy": r["best_test_accuracy"],
                "gap_vs_a3_pp": r["comparison"]["gap_vs_a3_pp"],
            }

        elif t == 5:
            r = run_task5(data, outdir=args.outdir,
                          max_epochs=args.max_epochs, device=device)
            # task6 looks denoise_state up by "noise20"/"noise40", which is the
            # same key task5 uses in by_noise.
            denoise_states = {}
            for key, entry in r["by_noise"].items():
                st = entry.get("model_state")
                if st is not None:
                    denoise_states[key] = st
            summary["task5"] = {
                "best_noise": r["best_noise"],
                "best_test_accuracy": r["best_test_accuracy"],
                "bottleneck": r["bottleneck"],
            }

        elif t == 6:
            # Task-6a wants the BEST 1-hidden AE representation - meaning the
            # autoencoder at Task-3's WINNING bottleneck, not just any
            # 1-hidden model. Handing over the first 1hidden_* state found
            # loaded a k=32 encoder into a k=64 slot whenever the winning
            # bottleneck was 64, and load_state_dict raised a shape mismatch that
            # killed Task-6 after Tasks 1-5 had already completed.
            # The state MUST therefore be keyed by the resolved bottleneck.
            from task6_weights import resolve_selection
            sel_bottleneck, _ = resolve_selection(args.outdir)
            plain_key = f"1hidden_{sel_bottleneck}"
            plain_ae_state = (ae_states or {}).get(plain_key)
            if plain_ae_state is None:
                log(f"no cached state for {plain_key}; Task-6 will train it "
                    f"itself (or --retrain-aes was set)")
            else:
                log(f"Task-6 reusing {plain_key} from Task-2")

            # Denoising states must match Task-5's bottleneck too.
            denoise_for_task6 = {}
            for noise in (20, 40):
                st = (denoise_states or {}).get(f"noise{noise}")
                if st is not None:
                    denoise_for_task6[f"noise{noise}"] = st

            r = run_task6(data, outdir=args.outdir,
                          max_epochs=args.max_epochs, device=device,
                          bottleneck=sel_bottleneck,
                          ae_state=plain_ae_state,
                          denoise_state=denoise_for_task6 or None)
            summary["task6"] = {
                "bottleneck": r["bottleneck"],
                "variants": list(r["variants"]),
            }

        log(f"TASK {t} finished in {(time.time() - t0) / 60:.1f} min")

    # ── cross-task figures: comparisons that only exist across tasks ──────────
    log("# building cross-task summary figures")
    try:
        from summary_figures import build_summary
        build_summary(args.outdir)
    except Exception as exc:
        # Summary figures are analysis, not results. Losing them must not
        # invalidate a completed pipeline, so the failure is reported loudly and
        # the pipeline still ends in a usable state.
        print(f"[WARN] summary figures failed: {exc!r}")

    total = (time.time() - t_start) / 60
    atomic_write_text(json.dumps(summary, indent=2, default=float),
                      os.path.join(args.outdir, "summary.json"))
    # Results live ONLY under <outdir>. An earlier ad-hoc run left summary.json
    # and selection.json beside the source; they went stale while results_final/
    # moved on, and an independent audit read the stale copy and reported numbers
    # that matched nothing in the real run. Remove any such strays so the only
    # summary a reader can find is the current one.
    for _stray in ("summary.json", "selection.json"):
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), _stray)
        if os.path.exists(_p):
            os.remove(_p)
            print(f"Removed stale root {_stray} (results live in {args.outdir}/)")

    # Drop figure subfolders that never received a figure, so browsing the tree
    # shows only folders that actually hold something.
    try:
        import plots
        pruned = plots.prune_empty_dirs(args.outdir)
        if pruned:
            print(f"Pruned {len(pruned)} empty figure subfolder(s)")
    except Exception as exc:
        print(f"[WARN] could not prune empty figure folders: {exc!r}")

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — {total:.1f} min total")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nFigures : {os.path.join(args.outdir, 'plots')}")
    print(f"Results : {args.outdir}/task{{1..6}}.json, summary.json")
    print(f"Monitor : python monitor.py --outdir {args.outdir} --watch --gpu")



if __name__ == "__main__":
    main()