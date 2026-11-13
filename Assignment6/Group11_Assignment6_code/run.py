"""Single entry point for Assignment 6 (works the same on Linux and Windows).

Examples:
    python run.py smoke                          correctness checks (run this first)
    python run.py inspect                        dataset statistics + handwriting gallery
    python run.py pilot --dataset hw             timing + one run to convergence per cell
    python run.py sweep --dataset cv --cell lstm the full grid for one (dataset, cell)
    python run.py select                         tables + best model per (dataset, cell)
    python run.py figures                        all report figures
    python run.py all                            inspect, sweep, select, figures

Finished runs are skipped, so an interrupted sweep can simply be started again.
"""
import argparse
import platform
import sys
from pathlib import Path

import torch

import config
import experiments
from data import prepare_data


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available")
    return torch.device(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Assignment 6: RNN and LSTM sequence classification")
    parser.add_argument("command", choices=["smoke", "inspect", "pilot", "sweep", "select", "figures", "all"])
    parser.add_argument("--dataset", choices=["hw", "cv", "both"], default="both")
    parser.add_argument("--cell", choices=["rnn", "lstm", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads used by torch")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override config.SEEDS")
    parser.add_argument("--configs", type=int, nargs="+", default=None,
                        help="indices into config.GRID to run (default: all), e.g. to split work between machines")
    parser.add_argument("--data", type=str, default=None, help="data folder (default: ../data)")
    parser.add_argument("--out", type=str, default=None, help="output folder (default: ../out)")
    parser.add_argument("--force", action="store_true", help="re-run configs that already finished")
    parser.add_argument("--lr", type=float, default=None, help="override config.LR")
    parser.add_argument("--batch-size", type=int, default=None, help="override config.BATCH_SIZE")
    parser.add_argument("--max-epochs", type=int, default=None, help="override config.MAX_EPOCHS")
    parser.add_argument("--hw-norm", choices=["minmax_axis", "minmax_joint", "none"], default=None,
                        help="override config.HW_NORM")
    parser.add_argument("--cv-norm", choices=["zscore_train", "none"], default=None, help="override config.CV_NORM")
    parser.add_argument("--no-converge", action="store_true", help="pilot: timing only")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data:
        config.DATA_DIR = Path(args.data).resolve()
    if args.out:
        config.OUT_DIR = Path(args.out).resolve()
    if args.seeds:
        config.SEEDS = args.seeds
    if args.lr:
        config.LR = args.lr
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.max_epochs:
        config.MAX_EPOCHS = args.max_epochs
    if args.hw_norm:
        config.HW_NORM = args.hw_norm
    if args.cv_norm:
        config.CV_NORM = args.cv_norm
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.command == "smoke":
        import smoke_test
        sys.exit(smoke_test.main())

    device = choose_device(args.device)
    print(f"Python {platform.python_version()} on {platform.system()}, torch {torch.__version__}, "
          f"device {device}, {torch.get_num_threads()} CPU threads")
    print(f"data: {config.DATA_DIR}")
    print(f"out:  {config.OUT_DIR}")

    dataset_names = ["hw", "cv"] if args.dataset == "both" else [args.dataset]
    cells = config.CELLS if args.cell == "both" else [args.cell]
    grid = config.GRID if args.configs is None else [config.GRID[i] for i in args.configs]

    for name in dataset_names:
        spec = config.DATASETS[name]
        print(f"\n=== {name} ===")
        if args.command in ("inspect", "all"):
            experiments.make_data_figures(spec)
        if args.command == "pilot":
            data = prepare_data(spec)
            experiments.pilot(spec, data, device, converge=not args.no_converge)
        if args.command in ("sweep", "all"):
            data = prepare_data(spec)
            for cell in cells:
                experiments.run_sweep(spec, data, cell, device, grid=grid, force=args.force)
        if args.command in ("select", "all"):
            for cell in cells:
                experiments.select(spec, cell)
        if args.command in ("figures", "all"):
            if args.command == "figures":
                experiments.make_data_figures(spec)
            experiments.make_result_figures(spec)


if __name__ == "__main__":
    main()
