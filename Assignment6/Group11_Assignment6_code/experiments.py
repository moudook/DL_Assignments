"""Running the hyperparameter grid, collecting results and choosing the best model.

A network is described by two lists (see config.GRID):
  dense_sizes:  neurons in each hidden layer applied at every time step, before the recurrent layers
  hidden_sizes: neurons in each stacked recurrent (RNN/LSTM) layer
"""
import csv
import json
import time

import numpy as np
import torch

import config
from data import HW_NORMALISERS, dataset_stats, make_loader, read_hw_file
from models import SeqClassifier
from plots import (plot_acc_vs_length, plot_confusion_pair, plot_curve_overlay, plot_error_curve,
                   plot_hw_gallery, plot_length_hist)
from train import compute_metrics, evaluate, make_optimizer, save_checkpoint, train_until_converged

EVAL_BATCH_SIZE = 256  # batch size for evaluation only; does not affect results


def sizes_label(sizes):
    return "[" + ", ".join(str(h) for h in sizes) + "]"


def net_label(dense_sizes, hidden_sizes):
    """e.g. "[64, 64]" (recurrent layers only) or "[32] -> [64]" (hidden layer per time step, then recurrent)."""
    if not dense_sizes:
        return sizes_label(hidden_sizes)
    return sizes_label(dense_sizes) + " -> " + sizes_label(hidden_sizes)


def run_name(dense_sizes, hidden_sizes, seed):
    """Folder name of one run, e.g. "h64-64_s11" or "d32_h64_s11"."""
    name = "h" + "-".join(str(h) for h in hidden_sizes) + f"_s{seed}"
    if dense_sizes:
        name = "d" + "-".join(str(d) for d in dense_sizes) + "_" + name
    return name


def network_order(dense_sizes, hidden_sizes):
    """Sort key: recurrent-only networks first (by depth, then width), then those with per-step hidden layers."""
    return (len(dense_sizes), dense_sizes, len(hidden_sizes), hidden_sizes)


def current_settings(device):
    return {
        "optimizer": config.OPTIMIZER, "lr": config.LR, "batch_size": config.BATCH_SIZE,
        "tol": config.TOL, "max_epochs": config.MAX_EPOCHS, "clip": config.CLIP,
        "epoch_error": config.EPOCH_ERROR, "hw_norm": config.HW_NORM, "cv_norm": config.CV_NORM,
        "val_frac": config.VAL_FRAC, "split_seed": config.SPLIT_SEED, "device": str(device),
        "torch": torch.__version__,
    }


def run_one(spec, data, cell, dense_sizes, hidden_sizes, seed, device, force=False):
    """Train one network until convergence, evaluate it on train/val/test and save everything."""
    label = net_label(dense_sizes, hidden_sizes)
    run_dir = config.dataset_out_dir(spec) / cell / run_name(dense_sizes, hidden_sizes, seed)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"  [skip] {spec.name} {cell} {label} seed {seed}: already done")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    print(f"  [run ] {spec.name} {cell} {label} seed {seed}", flush=True)
    torch.manual_seed(seed)
    classes = data["classes"]
    model = SeqClassifier(cell, spec.input_dim, hidden_sizes, len(classes), dense_sizes).to(device)
    optimizer = make_optimizer(config.OPTIMIZER, model.parameters(), config.LR)
    train_loader = make_loader(data["train"], config.BATCH_SIZE, shuffle=True, seed=seed)

    start = time.perf_counter()
    history = train_until_converged(model, train_loader, optimizer, device, config.TOL,
                                    config.MAX_EPOCHS, clip=config.CLIP, log_every=100)
    train_time = time.perf_counter() - start

    metrics = {
        "dataset": spec.name, "cell": cell, "dense_sizes": dense_sizes, "hidden_sizes": hidden_sizes,
        "seed": seed, "classes": classes, "params": model.count_params(),
        "epochs": len(history.epoch_error), "stop_reason": history.stop_reason,
        "final_error": history.epoch_error[-1], "train_time_s": train_time,
        "mean_epoch_time_s": float(np.mean(history.epoch_time)),
        "settings": current_settings(device),
    }
    for split in ("train", "val", "test"):
        loader = make_loader(data[split], EVAL_BATCH_SIZE, shuffle=False)
        result = evaluate(model, loader, device)
        split_metrics = compute_metrics(result["y_true"], result["y_pred"], len(classes))
        split_metrics["loss"] = result["loss"]
        if split == "test":  # kept for the accuracy-vs-length figure
            split_metrics["y_true"] = result["y_true"]
            split_metrics["y_pred"] = result["y_pred"]
            split_metrics["lengths"] = result["lengths"]
        metrics[split] = split_metrics

    print(f"         {history.stop_reason} after {metrics['epochs']} epochs, E = {metrics['final_error']:.5f}, "
          f"train {metrics['train']['accuracy']:.3f}  val {metrics['val']['accuracy']:.3f}  "
          f"test {metrics['test']['accuracy']:.3f}  ({train_time:.0f} s)", flush=True)

    info = {
        "cell": cell, "input_dim": spec.input_dim, "dense_sizes": dense_sizes, "hidden_sizes": hidden_sizes,
        "classes": classes, "standardiser": data["standardiser"].state_dict(), "hw_norm": config.HW_NORM,
    }
    save_checkpoint(run_dir / "model.pt", model, info)
    (run_dir / "history.json").write_text(
        json.dumps({"epoch_error": history.epoch_error, "epoch_time": history.epoch_time,
                    "stop_reason": history.stop_reason}), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def run_sweep(spec, data, cell, device, grid=None, seeds=None, force=False):
    grid = grid or config.GRID
    seeds = seeds or config.SEEDS
    for dense_sizes, hidden_sizes in grid:
        for seed in seeds:
            run_one(spec, data, cell, dense_sizes, hidden_sizes, seed, device, force)


# ---------------- Collecting results ----------------

def collect_results(spec, cell):
    """Finished runs of one (dataset, cell) for the seeds in config.SEEDS, as a list of metrics dicts."""
    cell_dir = config.dataset_out_dir(spec) / cell
    rows = []
    for metrics_path in sorted(cell_dir.glob("*/metrics.json")):
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        row.setdefault("dense_sizes", [])  # runs saved before per-time-step hidden layers were added had none
        if row["seed"] in config.SEEDS:
            rows.append(row)
    return rows


def seeds_suffix():
    """File-name suffix for tables: empty for a single seed, e.g. "_seeds11-12-13" for several."""
    if len(config.SEEDS) == 1:
        return ""
    return "_seeds" + "-".join(str(s) for s in config.SEEDS)


def summarise(rows):
    """One row per network: mean (and std) over seeds. With a single seed, the mean is that run's value."""
    networks = []
    for row in rows:
        network = (row["dense_sizes"], row["hidden_sizes"])
        if network not in networks:
            networks.append(network)
    networks.sort(key=lambda n: network_order(n[0], n[1]))

    def mean(values):
        return float(np.mean(values))

    def std(values):
        return float(np.std(values)) if len(values) > 1 else 0.0

    summary = []
    for dense_sizes, hidden_sizes in networks:
        runs = [row for row in rows if row["dense_sizes"] == dense_sizes and row["hidden_sizes"] == hidden_sizes]
        val_accs = [r["val"]["accuracy"] for r in runs]
        test_accs = [r["test"]["accuracy"] for r in runs]
        train_accs = [r["train"]["accuracy"] for r in runs]
        summary.append({
            "dense_sizes": dense_sizes,
            "hidden_sizes": hidden_sizes,
            "n_seeds": len(runs),
            "params": runs[0]["params"],
            "epochs": mean([r["epochs"] for r in runs]),
            "stop_reasons": sorted({r["stop_reason"] for r in runs}),
            "final_error": mean([r["final_error"] for r in runs]),
            "train_acc": mean(train_accs), "train_acc_std": std(train_accs),
            "val_acc": mean(val_accs), "val_acc_std": std(val_accs),
            "val_loss": mean([r["val"]["loss"] for r in runs]),
            "test_acc": mean(test_accs), "test_acc_std": std(test_accs),
            "epoch_time": mean([r["mean_epoch_time_s"] for r in runs]),
        })
    return summary


def pick_best(summary):
    """Highest validation accuracy; ties broken by lower validation loss, then fewer parameters."""
    return sorted(summary, key=lambda r: (-r["val_acc"], r["val_loss"], r["params"]))[0]


def write_tables(spec, cell, summary, best):
    """CSV and LaTeX table of every hyperparameter set (sheet item b); best row marked.
    Only when several seeds are used (separate files, see seeds_suffix) are the std columns added."""
    out_dir = config.dataset_out_dir(spec) / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    several_seeds = len(config.SEEDS) > 1

    header = ["dense_sizes", "recurrent_sizes", "params", "epochs", "stop", "final_E",
              "train_acc", "val_acc", "test_acc", "s_per_epoch", "best"]
    if several_seeds:
        header += ["n_seeds", "train_acc_std", "val_acc_std", "test_acc_std"]
    with open(out_dir / f"summary{seeds_suffix()}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in summary:
            values = [sizes_label(r["dense_sizes"]), sizes_label(r["hidden_sizes"]), r["params"],
                      f"{r['epochs']:.0f}", "/".join(r["stop_reasons"]), f"{r['final_error']:.5f}",
                      f"{r['train_acc']:.4f}", f"{r['val_acc']:.4f}", f"{r['test_acc']:.4f}",
                      f"{r['epoch_time']:.3f}", "yes" if r is best else ""]
            if several_seeds:
                values += [r["n_seeds"], f"{r['train_acc_std']:.4f}", f"{r['val_acc_std']:.4f}",
                           f"{r['test_acc_std']:.4f}"]
            writer.writerow(values)

    def percent(r, key):
        text = f"{100 * r[key]:.1f}"
        if several_seeds:
            text += rf" $\pm$ {100 * r[key + '_std']:.1f}"
        return text

    lines = [
        r"\begin{tabular}{llrrlrrrr}",
        r"\hline",
        r"Hidden (per step) & Recurrent & Params & Epochs & Stop & Final $E$ & Train acc & Val acc & Test acc \\",
        r"\hline",
    ]
    for r in summary:
        dense = sizes_label(r["dense_sizes"]) if r["dense_sizes"] else "--"
        cells = [dense, sizes_label(r["hidden_sizes"]), str(r["params"]), f"{r['epochs']:.0f}",
                 "/".join(r["stop_reasons"]), f"{r['final_error']:.4f}",
                 percent(r, "train_acc"), percent(r, "val_acc"), percent(r, "test_acc")]
        if r is best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_dir / f"summary{seeds_suffix()}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select(spec, cell):
    rows = collect_results(spec, cell)
    if not rows:
        print(f"  no finished runs for {spec.name} {cell}")
        return None
    summary = summarise(rows)
    best = pick_best(summary)
    write_tables(spec, cell, summary, best)
    best_info = {"dense_sizes": best["dense_sizes"], "hidden_sizes": best["hidden_sizes"],
                 "seeds": config.SEEDS, "val_acc": best["val_acc"],
                 "test_acc": best["test_acc"], "train_acc": best["train_acc"]}
    best_path = config.dataset_out_dir(spec) / cell / f"best{seeds_suffix()}.json"
    best_path.write_text(json.dumps(best_info, indent=1), encoding="utf-8")
    print(f"  {spec.name} {cell}: best {net_label(best['dense_sizes'], best['hidden_sizes'])}  "
          f"val {best['val_acc']:.3f}  test {best['test_acc']:.3f}  ({len(summary)} networks)")
    return best


# ---------------- Pilot: timing before the full sweep ----------------

def pilot(spec, data, device, timing_epochs=3, converge=True):
    """Before the full sweep: time a few epochs of the smallest and largest recurrent network of each cell
    type, then (optionally) train one medium network ([64]) of each cell type to convergence."""
    results = []
    for cell in config.CELLS:
        for hidden_sizes in ([32], [128, 128]):  # smallest and largest recurrent network of the grid
            torch.manual_seed(0)
            model = SeqClassifier(cell, spec.input_dim, hidden_sizes, len(data["classes"])).to(device)
            optimizer = make_optimizer(config.OPTIMIZER, model.parameters(), config.LR)
            loader = make_loader(data["train"], config.BATCH_SIZE, shuffle=True, seed=0)
            history = train_until_converged(model, loader, optimizer, device, tol=0.0,
                                            max_epochs=timing_epochs, clip=config.CLIP)
            seconds = float(np.mean(history.epoch_time[1:] or history.epoch_time))
            results.append({"cell": cell, "hidden_sizes": hidden_sizes, "s_per_epoch": seconds})
            print(f"  {spec.name} {cell:4s} {sizes_label(hidden_sizes):16s} {seconds:.3f} s/epoch")

    if converge:
        for cell in config.CELLS:
            hidden_sizes = [64]
            print(f"  {spec.name} {cell} {sizes_label(hidden_sizes)}: training to convergence")
            torch.manual_seed(0)
            model = SeqClassifier(cell, spec.input_dim, hidden_sizes, len(data["classes"])).to(device)
            optimizer = make_optimizer(config.OPTIMIZER, model.parameters(), config.LR)
            loader = make_loader(data["train"], config.BATCH_SIZE, shuffle=True, seed=0)
            history = train_until_converged(model, loader, optimizer, device, config.TOL,
                                            config.MAX_EPOCHS, clip=config.CLIP, log_every=50)
            val = evaluate(model, make_loader(data["val"], EVAL_BATCH_SIZE, shuffle=False), device)
            val_acc = compute_metrics(val["y_true"], val["y_pred"], len(data["classes"]))["accuracy"]
            results.append({"cell": cell, "hidden_sizes": hidden_sizes, "epochs": len(history.epoch_error),
                            "stop_reason": history.stop_reason, "final_error": history.epoch_error[-1],
                            "time_s": float(np.sum(history.epoch_time)), "val_acc": val_acc,
                            "epoch_error": history.epoch_error})
            print(f"    {history.stop_reason} after {len(history.epoch_error)} epochs "
                  f"({np.sum(history.epoch_time):.0f} s), E = {history.epoch_error[-1]:.5f}, val acc {val_acc:.3f}")

    out_path = config.OUT_DIR / "pilot" / f"pilot_{spec.name}_{device.type}_lr{config.LR:g}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"settings": current_settings(device), "results": results}, indent=1),
                        encoding="utf-8")
    return results


# ---------------- Figures ----------------

def make_data_figures(spec):
    """Dataset statistics, length histograms and (handwriting) the gallery of normalised characters."""
    out_dir = config.dataset_out_dir(spec)
    figure_dir = out_dir / "figures"
    stats = dataset_stats(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")

    for split in ("train", "test"):
        print(f"  {spec.name} {split}:")
        for class_name, s in stats["splits"][split].items():
            print(f"    {class_name:5s} {s['count']:4d} files, length {s['min_len']}-{s['max_len']} "
                  f"(median {s['median_len']:.0f})")

    unit = "points" if spec.name == "hw" else "frames"
    lengths = {name: s["lengths"] for name, s in stats["splits"]["train"].items()}
    plot_length_hist(lengths, f"{spec.name.upper()} train: sequence length per class",
                     f"Sequence length ({unit})", figure_dir / f"len_hist_{spec.name}")

    if spec.name == "hw":
        # The first N_GALLERY train files of each class (sorted by file name).
        normalised, raw = {}, {}
        for class_name in stats["classes"]:
            folder = config.DATA_DIR / spec.folder / class_name / spec.train_dir
            files = sorted(folder.glob(spec.pattern))[:config.N_GALLERY]
            raw[class_name] = [read_hw_file(f) for f in files]
            normalised[class_name] = [HW_NORMALISERS[config.HW_NORM](points) for points in raw[class_name]]
        plot_hw_gallery(normalised, f"Handwritten characters after normalisation ({config.HW_NORM})",
                        figure_dir / "hw_gallery")
        plot_hw_gallery(raw, "Handwritten characters before normalisation (raw coordinates)",
                        figure_dir / "hw_gallery_raw", unit_square=False)


def load_history(spec, cell, dense_sizes, hidden_sizes, seed):
    path = config.dataset_out_dir(spec) / cell / run_name(dense_sizes, hidden_sizes, seed) / "history.json"
    return json.loads(path.read_text(encoding="utf-8"))["epoch_error"]


def make_result_figures(spec):
    """Error curves (every run), best-model confusion matrices, RNN-vs-LSTM comparison (first seed only)."""
    figure_dir = config.dataset_out_dir(spec) / "figures"
    seed = config.SEEDS[0]  # curves are drawn for the first seed
    train_title = f"Train ({100 * (1 - config.VAL_FRAC):.0f}% of the train folder)"
    best_rows = {}
    for cell in config.CELLS:
        rows = collect_results(spec, cell)
        if not rows:
            continue
        overlay = []
        for row in sorted(rows, key=lambda r: network_order(r["dense_sizes"], r["hidden_sizes"])):
            if row["seed"] != seed:
                continue
            errors = load_history(spec, cell, row["dense_sizes"], row["hidden_sizes"], seed)
            label = f"{cell.upper()} {net_label(row['dense_sizes'], row['hidden_sizes'])}"
            file_name = f"curve_{spec.name}_{cell}_{run_name(row['dense_sizes'], row['hidden_sizes'], seed)}"
            plot_error_curve(errors, f"{spec.name.upper()} {label} ({row['stop_reason']})",
                             figure_dir / "curves" / file_name)
            overlay.append((label, errors))
        plot_curve_overlay(overlay, f"{spec.name.upper()} {cell.upper()}: all hyperparameter sets (seed {seed})",
                           figure_dir / f"curves_{spec.name}_{cell}")

        best = pick_best(summarise(rows))
        best_row = [r for r in rows if r["dense_sizes"] == best["dense_sizes"]
                    and r["hidden_sizes"] == best["hidden_sizes"] and r["seed"] == seed][0]
        best_rows[cell] = best_row
        title = (f"{spec.name.upper()} best {cell.upper()} "
                 f"{net_label(best_row['dense_sizes'], best_row['hidden_sizes'])}: "
                 f"train {100 * best_row['train']['accuracy']:.1f}%, test {100 * best_row['test']['accuracy']:.1f}%")
        plot_confusion_pair(best_row["train"]["confusion_matrix"], best_row["test"]["confusion_matrix"],
                            best_row["classes"], title, figure_dir / f"confusion_{spec.name}_{cell}",
                            train_title=train_title)

    if len(best_rows) == 2:
        curves = [(f"{cell.upper()} {net_label(r['dense_sizes'], r['hidden_sizes'])}",
                   load_history(spec, cell, r["dense_sizes"], r["hidden_sizes"], seed))
                  for cell, r in best_rows.items()]
        plot_curve_overlay(curves, f"{spec.name.upper()}: best RNN vs best LSTM",
                           figure_dir / f"rnn_vs_lstm_{spec.name}")

        # Accuracy per length bucket: tertiles of the test-set lengths.
        lengths = best_rows["rnn"]["test"]["lengths"]
        edges = list(np.quantile(lengths, [0, 1 / 3, 2 / 3, 1]))
        models = []
        for cell, r in best_rows.items():
            correct = [int(t == p) for t, p in zip(r["test"]["y_true"], r["test"]["y_pred"])]
            models.append((f"{cell.upper()} {net_label(r['dense_sizes'], r['hidden_sizes'])}",
                           r["test"]["lengths"], correct))
        plot_acc_vs_length(models, edges, f"{spec.name.upper()}: test accuracy vs sequence length",
                           figure_dir / f"acc_vs_length_{spec.name}")
        write_best_summary(spec, best_rows)


def write_best_summary(spec, best_rows):
    """Best RNN vs best LSTM: the numbers the comparison is based on (sheet item c + comparison)."""
    path = config.dataset_out_dir(spec) / "best_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dense_sizes", "recurrent_sizes", "params", "epochs", "stop",
                         "train_acc", "val_acc", "test_acc", "per_class_test_recall"])
        for cell, r in best_rows.items():
            recall = " ".join(f"{name}={100 * value:.1f}" for name, value
                              in zip(r["classes"], r["test"]["per_class_recall"]))
            writer.writerow([cell, sizes_label(r["dense_sizes"]), sizes_label(r["hidden_sizes"]), r["params"],
                             r["epochs"], r["stop_reason"], f"{r['train']['accuracy']:.4f}",
                             f"{r['val']['accuracy']:.4f}", f"{r['test']['accuracy']:.4f}", recall])
