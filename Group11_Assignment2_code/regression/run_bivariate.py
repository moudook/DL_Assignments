from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared.data import load_bivariate_data, train_val_test_split
from _common import RegressionRun, RegRunSpec


HYPERPARAMS = {
    "hidden_sizes": [2, 4, 8, 16, 32],
    "lr": 0.05,
    "epochs": 1500,
    "seed": 42,
    "log_every": 100,
}

DATASET_TAG = "Bivariate"
OUTPUT_ROOT_BASE = os.path.join(ROOT, "outputs", "regression", "bivariate")


def _parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=HYPERPARAMS["hidden_sizes"])
    p.add_argument("--lr", type=float, default=HYPERPARAMS["lr"])
    p.add_argument("--epochs", type=int, default=HYPERPARAMS["epochs"])
    p.add_argument("--seed", type=int, default=HYPERPARAMS["seed"])
    p.add_argument("--log_every", type=int, default=HYPERPARAMS["log_every"])
    p.add_argument("--hidden_activation", type=str, choices=["sigmoid", "tanh"], default="sigmoid")
    p.add_argument("--output_activation", type=str, choices=["linear"], default="linear")
    p.add_argument("--layers", type=str, choices=["1HL", "2HL", "all"], default="all")
    p.add_argument("--quiet", action="store_true")
    return vars(p.parse_args())


def main() -> None:
    cfg = _parse_args()
    print(f"config: {cfg}")

    X, y = load_bivariate_data()
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        X, y, ratios=(0.6, 0.2, 0.2), seed=cfg["seed"], stratify=False,
    )
    print(f"data: train={X_tr.shape[0]}  val={X_va.shape[0]}  test={X_te.shape[0]}")

    layers_to_run = ["1HL", "2HL"] if cfg["layers"] == "all" else [cfg["layers"]]

    for layer_label in layers_to_run:
        output_root = os.path.join(OUTPUT_ROOT_BASE, layer_label.lower())
        spec = RegRunSpec(
            dataset_tag=f"{DATASET_TAG}_{layer_label}",
            layer_label=layer_label,
            output_root=output_root,
            X_tr=X_tr, y_tr=y_tr,
            X_va=X_va, y_va=y_va,
            X_te=X_te, y_te=y_te,
            hidden_sizes=cfg["hidden_sizes"],
            hidden_activation=cfg["hidden_activation"],
            output_activation=cfg["output_activation"],
            lr=cfg["lr"],
            epochs=cfg["epochs"],
            seed=cfg["seed"],
            log_every=cfg["log_every"],
            quiet=cfg["quiet"],
        )
        RegressionRun(spec).execute()


if __name__ == "__main__":
    main()
