from __future__ import annotations

import numpy as np


class FCNN:
    # layer_sizes = [n_in, h1, h2, ..., n_out]. hidden layers are always sigmoid;
    # set output_activation="linear" for regression.

    def __init__(
        self,
        layer_sizes: list[int],
        hidden_activation: str = "sigmoid",
        output_activation: str = "sigmoid",
        seed: int = 42,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output sizes")
        if hidden_activation not in ("sigmoid", "tanh"):
            raise ValueError(f"Unknown hidden_activation: {hidden_activation}")
        if output_activation not in ("sigmoid", "tanh", "linear", "softmax"):
            raise ValueError(f"Unknown output_activation: {output_activation}")

        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.n_layers = len(layer_sizes) - 1

        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(self.n_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            scale = np.sqrt(1.0 / n_in)
            self.weights.append(rng.standard_normal((n_out, n_in)) * scale)
            self.biases.append(np.zeros(n_out))

        self._acts: list[np.ndarray] = []
        self._zs: list[np.ndarray] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(np.clip(z - z_max, -60.0, 60.0))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _activate(self, z: np.ndarray, layer_idx: int) -> np.ndarray:
        is_last = layer_idx == self.n_layers - 1
        activation = self.output_activation if is_last else self.hidden_activation
        if activation == "linear":
            return z
        if activation == "tanh":
            return np.tanh(z)
        if activation == "softmax":
            return self._softmax(z)
        return self._sigmoid(z)

    def forward(self, X: np.ndarray, store: bool = False) -> list[np.ndarray]:
        acts = [np.asarray(X, dtype=np.float64)]
        z_list: list[np.ndarray] = []

        h = acts[0]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b
            h = self._activate(z, i)
            acts.append(h)
            if store:
                z_list.append(z)

        if store:
            self._acts = acts
            self._zs = z_list
        return acts

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)[-1]
        if self.output_activation in ("sigmoid", "tanh", "softmax"):
            return np.argmax(out, axis=1)
        return out

    def _derivative(self, h: np.ndarray, activation: str) -> np.ndarray:
        if activation == "linear":
            return np.ones_like(h)
        if activation == "tanh":
            return 1.0 - h ** 2
        return h * (1.0 - h)

    def _loss_and_grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        # one full forward + backward pass. y is one-hot for sigmoid/tanh output, raw for linear.
        acts = self.forward(X, store=True)
        n = X.shape[0]
        yhat = acts[-1]

        diff = yhat - y
        mse = float(np.mean(diff ** 2))

        if self.output_activation == "softmax":
            sum_diff_yhat = np.sum(diff * yhat, axis=1, keepdims=True)
            delta = yhat * (diff - sum_diff_yhat)
        else:
            delta = diff * self._derivative(yhat, self.output_activation)

        grad_W: list[np.ndarray] = [None] * self.n_layers
        grad_b: list[np.ndarray] = [None] * self.n_layers

        grad_W[-1] = delta.T @ acts[-2] / n
        grad_b[-1] = delta.mean(axis=0)

        for i in range(self.n_layers - 2, -1, -1):
            h_acts = acts[i + 1]
            delta = (delta @ self.weights[i + 1]) * self._derivative(h_acts, self.hidden_activation)
            grad_W[i] = delta.T @ acts[i] / n
            grad_b[i] = delta.mean(axis=0)

        return mse, grad_W, grad_b

    def train_one_sample(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        y = np.asarray(y, dtype=np.float64).reshape(1, -1)
        mse, grad_W, grad_b = self._loss_and_grad(x, y)
        for i in range(self.n_layers):
            self.weights[i] -= lr * grad_W[i]
            self.biases[i] -= lr * grad_b[i]
        return mse

    def batch_mse(self, X: np.ndarray, y: np.ndarray) -> float:
        acts = self.forward(X)
        yhat = acts[-1]
        return float(np.mean((yhat - y) ** 2))