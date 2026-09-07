from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from models.fcnn import FCNN
from optimizers.sgd import SGDTrainer
from shared.metrics import regression_summary
from shared.plotting import (
    plot_error_curve,
    plot_model_output_superimposed,
    plot_scatter_target_vs_model,
    plot_node_surfaces,
    plot_node_curves_1d,
)


@dataclass
class RegArchResult:
    hidden: int
    train_mse: float
    train_rmse: float
    train_percent_rmse: float
    val_mse: float
    val_rmse: float
    val_percent_rmse: float


@dataclass
class RegRunSpec:
    dataset_tag: str
    layer_label: str
    output_root: str
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    hidden_sizes: Sequence[int]
    hidden_activation: str
    output_activation: str
    lr: float
    epochs: int
    seed: int
    log_every: int
    quiet: bool = False


def _dump_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _cv_table(results: list[RegArchResult], best: RegArchResult, tag: str, layer_label: str) -> str:
    header = f"  {'h':>3s}  {'tr_rmse':>10s}  {'tr_%rmse':>10s}  {'val_rmse':>10s}  {'val_%rmse':>10s}"
    rule = "  " + "-" * (len(header) - 2)
    body = "\n".join(
        f"  {r.hidden:>3d}  {r.train_rmse:>10.4f}  {r.train_percent_rmse:>10.4f}  {r.val_rmse:>10.4f}  {r.val_percent_rmse:>10.4f}"
        for r in results
    )
    return (
        f"\n{tag} cross-validation ({layer_label}):\n"
        f"{header}\n{rule}\n{body}\n"
        f"  best -> h={best.hidden}  val_mse={best.val_mse:.6f}"
    )


class RegressionRun:
    def __init__(self, spec: RegRunSpec) -> None:
        self.spec = spec
        self._results: list[RegArchResult] = []
        self._models: list[FCNN] = []
        self._best_index: int = -1

    def execute(self) -> RegArchResult:
        for h in self.spec.hidden_sizes:
            model, result = self._train_one(h)
            self._models.append(model)
            self._results.append(result)
            if self._best_index < 0 or result.val_mse < self._results[self._best_index].val_mse:
                self._best_index = len(self._results) - 1

        best = self._results[self._best_index]
        print(_cv_table(self._results, best, self.spec.dataset_tag, self.spec.layer_label))
        self._render_best()
        return best

    def _layer_sizes(self, h: int) -> list[int]:
        n_in = self.spec.X_tr.shape[1]
        n_out = 1
        if self.spec.layer_label == "1HL":
            return [n_in, h, n_out]
        if self.spec.layer_label == "2HL":
            h2 = max(1, h // 2)
            return [n_in, h, h2, n_out]
        raise ValueError(f"Unsupported layer label: {self.spec.layer_label}")

    def _train_one(self, h: int) -> tuple[FCNN, RegArchResult]:
        arch_dir = os.path.join(self.spec.output_root, f"h{h}")
        os.makedirs(arch_dir, exist_ok=True)

        model = FCNN(self._layer_sizes(h), hidden_activation=self.spec.hidden_activation, output_activation=self.spec.output_activation, seed=self.spec.seed)

        # y needs to be column vectors for fcnn regression
        y_tr_col = self.spec.y_tr.reshape(-1, 1)
        y_va_col = self.spec.y_va.reshape(-1, 1)

        history = SGDTrainer(
            model,
            lr=self.spec.lr,
            epochs=self.spec.epochs,
            seed=self.spec.seed,
            X_val=self.spec.X_va,
            y_val=y_va_col,
            log_every=self.spec.log_every,
            verbose=not self.spec.quiet,
        ).fit(self.spec.X_tr, y_tr_col)

        train_pred = model.predict(self.spec.X_tr).ravel()
        train_summary = regression_summary(
            self.spec.y_tr, train_pred,
            name=f"h{h} train",
        )

        val_pred = model.predict(self.spec.X_va).ravel()
        val_summary = regression_summary(
            self.spec.y_va, val_pred,
            name=f"h{h} validation",
        )

        plot_error_curve(
            history["train_mse"], history["val_mse"],
            title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {h} - error vs epoch",
            save_path=os.path.join(arch_dir, "error_curve.png"),
        )
        
        plot_model_output_superimposed(
            model, self.spec.X_va, self.spec.y_va,
            title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {h} - Validation superimposed",
            save_path=os.path.join(arch_dir, "superimposed_val.png"),
        )
        
        plot_scatter_target_vs_model(
            self.spec.y_va, val_pred,
            title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {h} - Validation scatter",
            save_path=os.path.join(arch_dir, "scatter_val.png"),
        )

        _dump_json(os.path.join(arch_dir, "metrics_tr.json"), train_summary)
        _dump_json(os.path.join(arch_dir, "metrics_val.json"), val_summary)

        result = RegArchResult(
            hidden=h,
            train_mse=history["train_mse"][-1],
            train_rmse=train_summary["rmse"],
            train_percent_rmse=train_summary["percent_rmse"],
            val_mse=history["val_mse"][-1],
            val_rmse=val_summary["rmse"],
            val_percent_rmse=val_summary["percent_rmse"],
        )
        return model, result

    def _render_best(self) -> None:
        best = self._results[self._best_index]
        best_model = self._models[self._best_index]
        best_dir = os.path.join(self.spec.output_root, "best")
        os.makedirs(best_dir, exist_ok=True)

        splits = [
            ("train", self.spec.X_tr, self.spec.y_tr),
            ("val", self.spec.X_va, self.spec.y_va),
            ("test", self.spec.X_te, self.spec.y_te),
        ]

        for split_name, X_split, y_split in splits:
            pred_split = best_model.predict(X_split).ravel()
            summary_split = regression_summary(
                y_split, pred_split,
                name=f"h{best.hidden} {split_name}",
            )
            
            plot_model_output_superimposed(
                best_model, X_split, y_split,
                title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {best.hidden} - {split_name.capitalize()} Superimposed",
                save_path=os.path.join(best_dir, f"superimposed_{split_name}.png"),
            )
            
            plot_scatter_target_vs_model(
                y_split, pred_split,
                title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {best.hidden} - {split_name.capitalize()} Scatter",
                save_path=os.path.join(best_dir, f"scatter_{split_name}.png"),
            )

            _dump_json(os.path.join(best_dir, f"metrics_{split_name}.json"), summary_split)

        # Plot node surfaces/curves for each split
        hidden_layer_indices = [1, 2] if self.spec.layer_label == "2HL" else [1]
        if self.spec.X_tr.shape[1] == 2:
            for split_name, X_split, y_split in splits:
                split_dir = os.path.join(best_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)
                for li in hidden_layer_indices:
                    n_nodes = best_model.layer_sizes[li]
                    plot_node_surfaces(
                        best_model, X_split, None,
                        layer_idx=li,
                        node_indices=list(range(n_nodes)),
                        title_prefix=f"{self.spec.dataset_tag} ({split_name}) - best {self.spec.layer_label} x {best.hidden}",
                        save_dir=split_dir,
                        kind="hidden",
                    )

                n_out = best_model.layer_sizes[-1]
                plot_node_surfaces(
                    best_model, X_split, y_split,
                    layer_idx=best_model.n_layers,
                    node_indices=list(range(n_out)),
                    title_prefix=f"{self.spec.dataset_tag} ({split_name}) - best {self.spec.layer_label} x {best.hidden}",
                    save_dir=split_dir,
                    kind="output",
                )
        elif self.spec.X_tr.shape[1] == 1:
            for split_name, X_split, y_split in splits:
                split_dir = os.path.join(best_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)
                for li in hidden_layer_indices:
                    n_nodes = best_model.layer_sizes[li]
                    plot_node_curves_1d(
                        best_model, X_split, None,
                        layer_idx=li,
                        node_indices=list(range(n_nodes)),
                        title_prefix=f"{self.spec.dataset_tag} ({split_name}) - best {self.spec.layer_label} x {best.hidden}",
                        save_dir=split_dir,
                        kind="hidden",
                    )

                n_out = best_model.layer_sizes[-1]
                plot_node_curves_1d(
                    best_model, X_split, y_split,
                    layer_idx=best_model.n_layers,
                    node_indices=list(range(n_out)),
                    title_prefix=f"{self.spec.dataset_tag} ({split_name}) - best {self.spec.layer_label} x {best.hidden}",
                    save_dir=split_dir,
                    kind="output",
                )

        test_pred = best_model.predict(self.spec.X_te).ravel()
        test_summary = regression_summary(self.spec.y_te, test_pred, name="test")
        print(f"  test -> rmse={test_summary['rmse']:.4f}  "
              f"%rmse={test_summary['percent_rmse']:.4f}")
