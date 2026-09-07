from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from models.fcnn import FCNN
from optimizers.sgd import SGDTrainer
from shared.data import to_one_hot
from shared.metrics import classification_summary
from shared.plotting import (
    plot_confusion_matrix_heatmap,
    plot_decision_regions,
    plot_error_curve,
    plot_node_surfaces,
)


@dataclass
class ArchResult:
    hidden: int
    train_mse: float
    train_acc: float
    val_mse: float
    val_acc: float
    precisions: np.ndarray
    recalls: np.ndarray
    f1_scores: np.ndarray
    mean_precision: float
    mean_recall: float
    mean_f1: float
    confusion_matrix: np.ndarray


@dataclass
class RunSpec:
    dataset_tag: str
    layer_label: str
    output_root: str
    n_classes: int
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


def _metrics_to_json(summary: dict) -> dict:
    return {
        "name": summary["name"],
        "accuracy": summary["accuracy"],
        "precisions": summary["precisions"].tolist(),
        "recalls": summary["recalls"].tolist(),
        "f1_scores": summary["f1_scores"].tolist(),
        "mean_precision": summary["mean_precision"],
        "mean_recall": summary["mean_recall"],
        "mean_f1": summary["mean_f1"],
        "confusion_matrix": summary["confusion_matrix"].tolist(),
    }


def _cv_table(results: list[ArchResult], best: ArchResult, tag: str, layer_label: str) -> str:
    n_classes = len(results[0].f1_scores) if results else 0
    f1_headers = "  ".join(f"F1_C{i:>d}" for i in range(n_classes))
    header = f"  {'h':>3s}  {'val_mse':>10s}  {'val_acc':>8s}  {'m_prec':>8s}  {'m_rec':>8s}  {'mean_f1':>8s}  {f1_headers}"
    rule = "  " + "-" * (len(header) - 2)
    
    lines = []
    for r in results:
        f1_vals = "  ".join(f"{f:>4.4f}" for f in r.f1_scores)
        lines.append(f"  {r.hidden:>3d}  {r.val_mse:>10.6f}  {r.val_acc:>8.4f}  {r.mean_precision:>8.4f}  {r.mean_recall:>8.4f}  {r.mean_f1:>8.4f}  {f1_vals}")
    body = "\n".join(lines)
    
    return (
        f"\n{tag} cross-validation ({layer_label}):\n"
        f"{header}\n{rule}\n{body}\n"
        f"  best -> h={best.hidden}  val_mse={best.val_mse:.6f}"
    )


class ClassifierRun:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self._results: list[ArchResult] = []
        self._models: list[FCNN] = []
        self._best_index: int = -1

    def execute(self) -> ArchResult:
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
        n_out = self.spec.n_classes
        if self.spec.layer_label == "1HL":
            return [n_in, h, n_out]
        if self.spec.layer_label == "2HL":
            h2 = max(1, h // 2)
            return [n_in, h, h2, n_out]
        raise ValueError(f"Unsupported layer label: {self.spec.layer_label}")

    def _train_one(self, h: int) -> tuple[FCNN, ArchResult]:
        arch_dir = os.path.join(self.spec.output_root, f"h{h}")
        os.makedirs(arch_dir, exist_ok=True)

        model = FCNN(self._layer_sizes(h), hidden_activation=self.spec.hidden_activation, output_activation=self.spec.output_activation, seed=self.spec.seed)
        y_tr_oh = to_one_hot(self.spec.y_tr, self.spec.n_classes)
        y_va_oh = to_one_hot(self.spec.y_va, self.spec.n_classes)

        history = SGDTrainer(
            model,
            lr=self.spec.lr,
            epochs=self.spec.epochs,
            seed=self.spec.seed,
            X_val=self.spec.X_va,
            y_val=y_va_oh,
            log_every=self.spec.log_every,
            verbose=not self.spec.quiet,
        ).fit(self.spec.X_tr, y_tr_oh)

        tr_summary = classification_summary(
            self.spec.y_tr, model.predict(self.spec.X_tr),
            name=f"h{h} train",
        )
        val_summary = classification_summary(
            self.spec.y_va, model.predict(self.spec.X_va),
            name=f"h{h} validation",
        )

        plot_error_curve(
            history["train_mse"], history["val_mse"],
            title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {h} - error vs epoch",
            save_path=os.path.join(arch_dir, "error_curve.png"),
        )
        plot_confusion_matrix_heatmap(
            val_summary["confusion_matrix"],
            title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {h} - validation confusion matrix",
            save_path=os.path.join(arch_dir, "cm_val.png"),
        )
        _dump_json(os.path.join(arch_dir, "metrics_tr.json"), _metrics_to_json(tr_summary))
        _dump_json(os.path.join(arch_dir, "metrics_val.json"), _metrics_to_json(val_summary))

        result = ArchResult(
            hidden=h,
            train_mse=history["train_mse"][-1],
            train_acc=tr_summary["accuracy"],
            val_mse=history["val_mse"][-1],
            val_acc=val_summary["accuracy"],
            precisions=val_summary["precisions"],
            recalls=val_summary["recalls"],
            f1_scores=val_summary["f1_scores"],
            mean_precision=val_summary["mean_precision"],
            mean_recall=val_summary["mean_recall"],
            mean_f1=val_summary["mean_f1"],
            confusion_matrix=val_summary["confusion_matrix"],
        )
        return model, result

    def _render_best(self) -> None:
        best = self._results[self._best_index]
        best_model = self._models[self._best_index]
        best_dir = os.path.join(self.spec.output_root, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Render decision regions for Train, Val, and Test
        for split_name, X_split, y_split in [
            ("train", self.spec.X_tr, self.spec.y_tr),
            ("val", self.spec.X_va, self.spec.y_va),
            ("test", self.spec.X_te, self.spec.y_te),
        ]:
            plot_decision_regions(
                best_model, X_split, y_split,
                title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {best.hidden} - decision regions ({split_name})",
                save_path=os.path.join(best_dir, f"decision_regions_{split_name}.png"),
            )

        # Test set summary and confusion matrix
        test_summary = classification_summary(
            self.spec.y_te, best_model.predict(self.spec.X_te),
            name=f"h{best.hidden} test",
        )
        plot_confusion_matrix_heatmap(
            test_summary["confusion_matrix"],
            title=f"{self.spec.dataset_tag} - {self.spec.layer_label} x {best.hidden} - test confusion matrix",
            save_path=os.path.join(best_dir, "cm_test.png"),
        )
        _dump_json(os.path.join(best_dir, "metrics_test.json"), _metrics_to_json(test_summary))

        # Node output plots for Train, Val, Test splits
        splits = [
            ("train", self.spec.X_tr, self.spec.y_tr),
            ("val", self.spec.X_va, self.spec.y_va),
            ("test", self.spec.X_te, self.spec.y_te),
        ]
        hidden_layer_indices = [1, 2] if self.spec.layer_label == "2HL" else [1]

        for sname, X_s, y_s in splits:
            for li in hidden_layer_indices:
                n_nodes = best_model.layer_sizes[li]
                plot_node_surfaces(
                    best_model, X_s, y_s,
                    layer_idx=li,
                    node_indices=list(range(n_nodes)),
                    title_prefix=f"{self.spec.dataset_tag} ({sname}) - best {self.spec.layer_label} x {best.hidden}",
                    save_dir=os.path.join(best_dir, sname),
                    kind="hidden",
                )

            n_out = best_model.layer_sizes[-1]
            plot_node_surfaces(
                best_model, X_s, y_s,
                layer_idx=best_model.n_layers,
                node_indices=list(range(n_out)),
                title_prefix=f"{self.spec.dataset_tag} ({sname}) - best {self.spec.layer_label} x {best.hidden}",
                save_dir=os.path.join(best_dir, sname),
                kind="output",
            )

        print(f"  test -> acc={test_summary['accuracy']:.4f}  "
              f"mean_f1={test_summary['mean_f1']:.4f}")
