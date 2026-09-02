"""Verify the remaining math: data split ratios and SGD per-epoch MSE logging."""
import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())

from shared.data import train_val_test_split, to_one_hot
from models.fcnn import FCNN
from optimizers.sgd import SGDTrainer


def check_split_ratios() -> None:
    np.random.seed(0)
    X = np.random.randn(1500, 2)
    y = np.array([0] * 500 + [1] * 500 + [2] * 500)
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        X, y, ratios=(0.6, 0.2, 0.2), seed=42, stratify=True
    )
    assert X_tr.shape[0] == 900
    assert X_va.shape[0] == 300
    assert X_te.shape[0] == 300
    for cls in [0, 1, 2]:
        n_tr = int(np.sum(y_tr == cls))
        n_va = int(np.sum(y_va == cls))
        n_te = int(np.sum(y_te == cls))
        assert (n_tr, n_va, n_te) == (300, 100, 100), \
            f"class {cls}: got {(n_tr, n_va, n_te)}"
    print("  [OK] stratified 60/20/20 split: 900/300/300, per-class 300/100/100")


def check_one_hot() -> None:
    y = np.array([0, 1, 2, 0, 1])
    oh = to_one_hot(y, 3)
    expected = np.eye(3)[y]
    assert np.allclose(oh, expected), "one-hot mismatch"
    print("  [OK] one-hot encoding matches np.eye(n_classes)[y]")


def check_sgd_epoch_mse_logging() -> None:
    np.random.seed(99)
    X = np.random.randn(40, 2)
    y_idx = np.random.randint(0, 3, size=40)
    y_oh = to_one_hot(y_idx, 3)
    X_val = np.random.randn(20, 2)
    y_val_idx = np.random.randint(0, 3, size=20)
    y_val_oh = to_one_hot(y_val_idx, 3)

    model = FCNN([2, 5, 3], seed=0)
    trainer = SGDTrainer(model, lr=0.3, epochs=5, seed=0,
                         X_val=X_val, y_val=y_val_oh)
    history = trainer.fit(X, y_oh)

    assert len(history["train_mse"]) == 5
    assert len(history["val_mse"]) == 5
    assert len(history["val_acc"]) == 5

    expected = float(np.mean((model.forward(X)[-1] - y_oh) ** 2))
    assert abs(history["train_mse"][-1] - expected) < 1e-10, \
        f"logged train_mse {history['train_mse'][-1]} != recomputed {expected}"
    expected_val = float(np.mean((model.forward(X_val)[-1] - y_val_oh) ** 2))
    assert abs(history["val_mse"][-1] - expected_val) < 1e-10, \
        f"logged val_mse {history['val_mse'][-1]} != recomputed {expected_val}"
    print("  [OK] SGD trainer's per-epoch train_mse and val_mse match hand recomputation")

    for i in range(1, len(history["train_mse"])):
        pass
    losses_decreased = sum(
        1 for i in range(1, len(history["train_mse"]))
        if history["train_mse"][i] <= history["train_mse"][i - 1] + 1e-9
    )
    assert losses_decreased >= 3, \
        f"train_mse not monotonically decreasing on average: {history['train_mse']}"
    print(f"  [OK] train_mse trend over 5 epochs: "
          f"{[f'{v:.4f}' for v in history['train_mse']]}")


def check_argmax_predict() -> None:
    np.random.seed(0)
    model = FCNN([2, 4, 3], seed=0)
    X = np.random.randn(7, 2)
    acts = model.forward(X)[-1]
    expected = np.argmax(acts, axis=1)
    got = model.predict(X)
    assert np.array_equal(got, expected), "argmax predict mismatch"
    print("  [OK] predict() == argmax(forward(X)[-1], axis=1) for sigmoid output")


if __name__ == "__main__":
    print("Verifying stratified 60/20/20 split ...")
    check_split_ratios()
    print("Verifying one-hot encoding ...")
    check_one_hot()
    print("Verifying SGD per-epoch MSE logging ...")
    check_sgd_epoch_mse_logging()
    print("Verifying argmax predict ...")
    check_argmax_predict()
    print("\nAll auxiliary math checks PASSED.")
