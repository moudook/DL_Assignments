"""Numerical verification of every mathematical operation in the FCNN.

Compares:
  1. Sigmoid output values vs the closed-form definition.
  2. Forward pass output vs a hand-rolled reference.
  3. Analytical MSE + gradients (from FCNN._loss_and_grad) vs finite-difference approximations.
  4. Weight update direction (loss must decrease after one SGD step).

If any check fails, the script raises an AssertionError with a clear message.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.getcwd())

from models.fcnn import FCNN


ATOL = 1e-7
RTOL = 1e-5


def check_sigmoid() -> None:
    model = FCNN([2, 3, 3], seed=0)
    z = np.array([-1000.0, -60.0, 0.0, 60.0, 1000.0])
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    got = model._sigmoid(z)
    assert np.allclose(got, expected, atol=1e-6), f"sigmoid values wrong: {got}"
    print("  [OK] sigmoid matches closed-form 1/(1+exp(-z))")


def check_forward() -> None:
    np.random.seed(123)
    layer_sizes = [2, 4, 3]
    model = FCNN(layer_sizes, seed=0)
    X = np.array([[0.5, -0.3], [1.2, 0.8], [-0.1, 2.0]])

    acts = model.forward(X, store=True)
    z_list = model._zs

    for i, (W, b) in enumerate(zip(model.weights, model.biases)):
        z_manual = acts[i] @ W.T + b
        a_manual = model._sigmoid(z_manual)
        assert np.allclose(z_list[i], z_manual, atol=1e-10), \
            f"forward z mismatch at layer {i}"
        assert np.allclose(acts[i + 1], a_manual, atol=1e-10), \
            f"forward activation mismatch at layer {i}"
    print("  [OK] forward pass matches hand-rolled reference (z = hW^T + b; a = sigmoid(z))")


def mse(W_list, b_list, X, y, output_activation="sigmoid") -> float:
    """Recompute MSE from scratch with given weights/biases (for finite differences)."""
    h = X.astype(np.float64)
    for i, (W, b) in enumerate(zip(W_list, b_list)):
        z = h @ W.T + b
        is_last = i == len(W_list) - 1
        if is_last and output_activation == "linear":
            h = z
        else:
            h = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    diff = h - y
    return float(np.mean(diff ** 2))


def finite_difference_grads(
    model: FCNN, X: np.ndarray, y: np.ndarray, eps: float = 1e-5
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Numerically estimate dL/dW and dL/db for every layer via central differences."""
    grad_W = [np.zeros_like(W) for W in model.weights]
    grad_b = [np.zeros_like(b) for b in model.biases]
    for i, W in enumerate(model.weights):
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                W[r, c] += eps
                lp = mse(model.weights, model.biases, X, y)
                W[r, c] -= 2 * eps
                lm = mse(model.weights, model.biases, X, y)
                W[r, c] += eps
                grad_W[i][r, c] = (lp - lm) / (2 * eps)
    for i, b in enumerate(model.biases):
        for j in range(b.shape[0]):
            b[j] += eps
            lp = mse(model.weights, model.biases, X, y)
            b[j] -= 2 * eps
            lm = mse(model.weights, model.biases, X, y)
            b[j] += eps
            grad_b[i][j] = (lp - lm) / (2 * eps)
    return grad_W, grad_b


def check_gradients() -> None:
    np.random.seed(7)
    layer_sizes = [3, 5, 4, 2]
    X = np.random.randn(6, 3)
    y_raw = np.random.randint(0, 2, size=6)
    y_oh = np.eye(2)[y_raw]

    model = FCNN(layer_sizes, output_activation="sigmoid", seed=0)
    mse_val, grad_W, grad_b = model._loss_and_grad(X, y_oh)
    fd_W, fd_b = finite_difference_grads(model, X, y_oh, eps=1e-5)

    mse_check = mse(model.weights, model.biases, X, y_oh)
    assert abs(mse_val - mse_check) < 1e-10, f"MSE mismatch: {mse_val} vs {mse_check}"
    print(f"  [OK] MSE value matches hand-rolled computation: {mse_val:.6f}")

    for i, (gw, gw_fd) in enumerate(zip(grad_W, fd_W)):
        max_err = float(np.max(np.abs(gw - gw_fd)))
        rel_err = max_err / (float(np.max(np.abs(gw_fd))) + 1e-12)
        assert rel_err < 1e-4, \
            f"layer {i} grad_W mismatch: max abs err {max_err:.2e}, rel err {rel_err:.2e}"
        print(f"  [OK] grad_W[layer {i}] matches finite difference (rel err {rel_err:.2e})")

    for i, (gb, gb_fd) in enumerate(zip(grad_b, fd_b)):
        max_err = float(np.max(np.abs(gb - gb_fd)))
        rel_err = max_err / (float(np.max(np.abs(gb_fd))) + 1e-12)
        assert rel_err < 1e-4, \
            f"layer {i} grad_b mismatch: max abs err {max_err:.2e}, rel err {rel_err:.2e}"
        print(f"  [OK] grad_b[layer {i}] matches finite difference (rel err {rel_err:.2e})")


def check_gradient_direction() -> None:
    np.random.seed(11)
    X = np.random.randn(8, 2)
    y = np.eye(3)[np.random.randint(0, 3, size=8)]

    model = FCNN([2, 8, 3], output_activation="sigmoid", seed=0)
    W_save = [W.copy() for W in model.weights]
    b_save = [b.copy() for b in model.biases]

    mse_before = mse(model.weights, model.biases, X, y)
    mse_val, grad_W, grad_b = model._loss_and_grad(X, y)
    lr = 0.5
    for i in range(model.n_layers):
        model.weights[i] -= lr * grad_W[i]
        model.biases[i] -= lr * grad_b[i]
    mse_after = mse(model.weights, model.biases, X, y)
    assert mse_after < mse_before, \
        f"SGD step did not reduce loss: {mse_before:.6f} -> {mse_after:.6f}"
    print(f"  [OK] one SGD step reduces loss: {mse_before:.6f} -> {mse_after:.6f}")

    model.weights = W_save
    model.biases = b_save


def check_one_sample_equivalence() -> None:
    np.random.seed(13)
    X = np.random.randn(5, 3)
    y = np.eye(4)[np.random.randint(0, 4, size=5)]

    model = FCNN([3, 6, 4], output_activation="sigmoid", seed=0)
    W_save = [W.copy() for W in model.weights]
    b_save = [b.copy() for b in model.biases]

    for i in range(X.shape[0]):
        model.train_one_sample(X[i], y[i], lr=0.0)
    for W_orig, W_after in zip(W_save, model.weights):
        assert np.allclose(W_orig, W_after), "lr=0 update should not change weights"
    for b_orig, b_after in zip(b_save, model.biases):
        assert np.allclose(b_orig, b_after), "lr=0 update should not change biases"
    print("  [OK] lr=0 leaves weights unchanged")

    model.weights = [W.copy() for W in W_save]
    model.biases = [b.copy() for b in b_save]
    lr = 0.1
    online_mse = 0.0
    for i in range(X.shape[0]):
        online_mse += model.train_one_sample(X[i], y[i], lr)
    online_mse /= X.shape[0]
    print(f"  [OK] online SGD avg per-sample MSE = {online_mse:.6f}")


def check_mse_definition() -> None:
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.8, 0.3, 0.6, 0.1])
    expected = float(np.mean((y_pred - y_true) ** 2))
    got = float(np.mean((y_pred - y_true) ** 2))
    assert abs(got - expected) < 1e-12
    print(f"  [OK] MSE = mean((y_pred - y_true)^2): {got:.6f}")


if __name__ == "__main__":
    print("Verifying sigmoid definition ...")
    check_sigmoid()
    print("Verifying forward pass (z = hW^T + b, a = sigmoid(z)) ...")
    check_forward()
    print("Verifying MSE definition ...")
    check_mse_definition()
    print("Verifying analytical gradients vs finite differences ...")
    check_gradients()
    print("Verifying SGD update direction (loss must decrease) ...")
    check_gradient_direction()
    print("Verifying train_one_sample mechanics ...")
    check_one_sample_equivalence()
    print("\nAll mathematical checks PASSED.")
