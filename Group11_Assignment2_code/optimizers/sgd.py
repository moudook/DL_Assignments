from __future__ import annotations

import numpy as np

from models.fcnn import FCNN


class SGDTrainer:
    # one sample per weight update, epoch = one shuffled pass. val_mse/val_acc
    # are computed at the end of each epoch if X_val/y_val are given.

    def __init__(
        self,
        model: FCNN,
        lr: float = 0.1,
        epochs: int = 1000,
        seed: int = 42,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        log_every: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.X_val = X_val
        self.y_val = y_val
        self.log_every = log_every
        self.verbose = verbose

    def _val_acc(self) -> float:
        if self.X_val is None or self.y_val is None:
            return float("nan")
        preds = self.model.predict(self.X_val)
        y_true = np.argmax(np.asarray(self.y_val), axis=1) if self.y_val.ndim > 1 else np.asarray(self.y_val)
        return float(np.mean(preds == y_true))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        n = X_train.shape[0]
        rng = np.random.default_rng(self.seed)

        history: dict[str, list[float]] = {
            "train_mse": [],
            "val_mse": [],
            "val_acc": [],
        }

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for idx in order:
                self.model.train_one_sample(X_train[idx], y_train[idx], self.lr)

            train_mse = self.model.batch_mse(X_train, y_train)
            history["train_mse"].append(train_mse)

            if self.X_val is not None and self.y_val is not None:
                val_mse = self.model.batch_mse(self.X_val, self.y_val)
                history["val_mse"].append(val_mse)
                history["val_acc"].append(self._val_acc())
            else:
                history["val_mse"].append(float("nan"))
                history["val_acc"].append(float("nan"))

            if self.verbose and (
                self.log_every <= 0
                or epoch == 0
                or (epoch + 1) % self.log_every == 0
                or epoch == self.epochs - 1
            ):
                msg = f"epoch {epoch + 1:5d}/{self.epochs}  train_mse={train_mse:.6f}"
                if self.X_val is not None:
                    msg += f"  val_mse={history['val_mse'][-1]:.6f}"
                    if not np.isnan(history["val_acc"][-1]):
                        msg += f"  val_acc={history['val_acc'][-1]:.4f}"
                print(msg)

        return history