"""
summary_figures.py — cross-task comparison figures.

Everything else lives under a task's own folder. This module produces the
figures that only exist ACROSS tasks, which are the ones the report's
conclusions rest on: every representation method against the A3 baseline, every
task's best against every other task's best, and per-class comparison so the
error structure of each representation is comparable.

Outputs to <outdir>/plots/00_summary/.
"""

import json
import os

import numpy as np
import torch

import plots
from data import CLASS_NAMES
from run_tracker import atomic_write_text

# The A3 baseline, cited from the submitted Group11_Assignment3_report.pdf
# (the submission of record). The Assignment-3 results/ folder disagrees with
# this and is a partial leftover, so it is deliberately not used.
A3_TEST_ACC = 0.9876
A3_VAL_ACC = 0.9884


def load_task(outdir, task):
    path = os.path.join(outdir, f"task{task}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def best_accuracy(t):
    """Best test accuracy for a task's result dict, plus its label."""
    if t is None:
        return None, None
    if t["task"] == 1:
        return (t["best_test_accuracy"],
                f"PCA k={t['best_dimension']} ({t['best_dimension_arch']})")
    if t["task"] in (3, 4):
        kind = "1-hidden" if t["task"] == 3 else "3-hidden"
        return (t["best_test_accuracy"],
                f"{kind} AE k={t['best_bottleneck']} ({t['best_arch']})")
    if t["task"] == 5:
        return (t["best_test_accuracy"],
                f"denoising AE {int(t['best_noise'] * 100)}% "
                f"k={t['bottleneck']} ({t['classifier_arch']})")
    if t["task"] == 2:
        # Task-2 has no accuracy; its best k is by reconstruction error.
        kinds = t.get("best_by_kind", {})
        if kinds:
            b = min(kinds.items(), key=lambda kv: kv[1]["test_recon_error"])
            return (None, f"best recon: {b[0]} k={b[1]['bottleneck']}")
    return None, None


def build_summary(outdir):
    """Produce every cross-task figure. Safe to call with missing tasks."""
    summary_dir = plots.ensure_dir(os.path.join(plots.fig_root(outdir),
                                                "00_summary"))
    tasks = {t: load_task(outdir, t) for t in range(1, 7)}
    present = [t for t, v in tasks.items() if v]
    print(f"\nSummary figures from tasks: {present}")

    # ── 1. every method's best against the A3 baseline ──────────────────────
    best = {}
    for t in present:
        acc, label = best_accuracy(tasks[t])
        if acc is not None:
            best[f"task{t}"] = acc

    if best:
        series = dict(best)
        series["A3 baseline"] = A3_TEST_ACC
        plots.comparison_bars(
            series,
            "Best test accuracy per task, against the Assignment-3 baseline",
            os.path.join(summary_dir, "summary_best_vs_a3.png"),
            labels=[{"task1": "Task-1\nPCA", "task3": "Task-3\n1-hidden AE",
                     "task4": "Task-4\n3-hidden AE",
                     "task5": "Task-5\ndenoising AE"}.get(k, k)
                    for k in series])

        # Same comparison showing the gap in percentage points, which is the
        # number the report's conclusion actually turns on.
        plots.generalisation_gap(
            {k: 100 * (v - A3_TEST_ACC) for k, v in series.items()},
            "Gap to the Assignment-3 baseline (positive = better than A3)",
            os.path.join(summary_dir, "summary_gap_vs_a3.png"),
            ylabel="percentage points vs A3")

    # ── 2. per-dimension / per-bottleneck accuracy for every task ────────────
    per_task_curves = {}
    if tasks[1]:
        d = tasks[1]["by_dimension"]
        per_task_curves["Task-1 PCA"] = {
            int(k): v["test_accuracy"] for k, v in d.items()}
    if tasks[3]:
        d = tasks[3]["by_bottleneck"]
        per_task_curves["Task-3 1-hidden AE"] = {
            int(k): v["test_accuracy"] for k, v in d.items()}
    if tasks[4]:
        d = tasks[4]["by_bottleneck"]
        per_task_curves["Task-4 3-hidden AE"] = {
            int(k): v["test_accuracy"] for k, v in d.items()}

    if per_task_curves:
        all_dims = sorted({d for c in per_task_curves.values() for d in c})
        plots.series_curve(
            per_task_curves,
            "Test accuracy vs representation size: PCA against both autoencoders",
            os.path.join(summary_dir, "summary_accuracy_vs_dimension.png"),
            xlabel="retained dimension / bottleneck size",
            ylabel="test accuracy (%)",
            ref=100 * A3_TEST_ACC, ref_label="A3 baseline 98.76%")

        plots.dimension_bars(
            per_task_curves,
            [str(d) for d in all_dims],
            "Test accuracy by representation size and method",
            os.path.join(summary_dir, "summary_accuracy_bars.png"),
            ref=100 * A3_TEST_ACC,
            ref_label="A3 baseline 98.76%")

    # ── 3. per-class accuracy compared across tasks ──────────────────────────
    per_class = {}
    for t in present:
        data = tasks[t]
        if t == 1:
            key = f"Task-1 PCA k={data['best_dimension']}"
            # JSON keys are strings after a round-trip; best_dimension is an int.
            src = data["by_dimension"][str(data["best_dimension"])]
        elif t in (3, 4):
            key = f"Task-{t} {'1-hidden' if t == 3 else '3-hidden'} " \
                  f"k={data['best_bottleneck']}"
            src = data["by_bottleneck"][str(data["best_bottleneck"])]
        elif t == 5:
            key = f"Task-5 denoising {int(data['best_noise'] * 100)}%"
            src = data["by_noise"][f"noise{int(data['best_noise'] * 100)}"]
        else:
            continue
        pc = (src.get("test_per_class") or
              src.get("classifier", {}).get("per_class"))
        if pc:
            per_class[key] = {p["digit"]: 100 * p["accuracy"] for p in pc}

    if per_class:
        keys = list(per_class)
        width = 0.8 / len(keys)
        fig, ax = plots.plt.subplots(figsize=plots.FIGSIZE_WIDE)
        x = np.arange(len(CLASS_NAMES))
        for i, k in enumerate(keys):
            ax.bar(x + i * width - 0.4 + width / 2,
                   [per_class[k][d] for d in CLASS_NAMES], width, label=k)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_xlabel("digit")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Per-class test accuracy, best configuration per task")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
        plots.save_atomic(fig, os.path.join(
            summary_dir, "summary_per_class_by_task.png"))

    # ── 4. task discoverability index ────────────────────────────────────────
    # A listing of which figure folder answered which question, written beside
    # the figures so the report can point at a path and a reader can find it.
    write_index(outdir, summary_dir, present)

    summary_rows = []
    for t in present:
        acc, label = best_accuracy(tasks[t])
        summary_rows.append({
            "task": t,
            "best": label,
            "test_accuracy": acc,
            "gap_vs_a3_pp": (100 * (acc - A3_TEST_ACC)) if acc is not None
                            else None,
        })
    atomic_write_text(json.dumps(summary_rows, indent=2, default=float),
                      os.path.join(outdir, "summary_best.json"))
    return summary_rows


# Maps a report question to the folder that answers it. Kept deliberately short:
# it is a navigation aid, not documentation of every figure.
INDEX = {
    1: "01 PCA decomposition and the representations built from it",
    2: "02 autoencoder reconstruction quality, per architecture and bottleneck",
    3: "03 classification from the 1-hidden autoencoder representation",
    4: "04 classification from the 3-hidden autoencoder representation",
    5: "05 denoising autoencoders at 20% and 40% corruption",
    6: "06 maximally-activating inputs and encoder weights, plain vs denoising",
}


def write_index(outdir, summary_dir, present):
    """
    Write a plain-text figure index.

    The figure tree is numbered so listing order matches conceptual grouping;
    this file records the mapping so navigating it never depends on memorising
    a directory name.
    """
    lines = ["Assignment 4 — figure index", "=" * 40, ""]
    lines.append("Each task folder is numbered and split by concept "
                 "(training curves, accuracy,")
    lines.append("confusion matrices, reconstructions, reconstruction error, "
                 "weights, comparisons),")
    lines.append("so the same question has the same subfolder in every task.")
    lines.append("")
    for t in present:
        name = INDEX.get(t, f"task {t}")
        lines.append(f"plots/task{t}/   {name}")
    lines.append("")
    lines.append("plots/00_summary/  cross-task comparisons and the A3 baseline")
    lines.append("")
    lines.append("Subfolders (identical in every task folder):")
    for sub, desc in (
        ("01_training_curves", "loss vs epoch, loss + accuracy"),
        ("02_accuracy", "accuracy vs epoch, per-class accuracy"),
        ("03_confusion_matrices", "raw and row-normalised confusion matrices"),
        ("04_reconstructions", "original vs reconstructed image grids"),
        ("05_reconstruction_error", "reconstruction error per split/architecture"),
        ("06_weights_and_activations", "weights and activation distributions"),
        ("07_representations", "variance retained, eigen spectrum, projections"),
        ("08_comparisons", "what the report's conclusions rest on"),
    ):
        lines.append(f"  {sub:<28} {desc}")
    atomic_write_text("\n".join(lines) + "\n",
                      os.path.join(summary_dir, "README.txt"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    rows = build_summary(args.outdir)
    print(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()