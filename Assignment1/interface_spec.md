# Assignment 1 — Final Interface Specification

This is the agreed contract between all modules before implementation begins.

---

## `Assignment1/data_loader.py`

```python
def load_ls_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads Class1.txt, Class2.txt, Class3.txt from data_dir.
    Returns:
        X : (1500, 2)  — stacked, float
        y : (1500,)    — class labels {0, 1, 2}
    """

def load_nls_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads NLS_Group11.txt. Skips the plain-text header line (line 0).
    Returns:
        X : (1700, 2)  — float
        y : (1700,)    — class labels {0, 1, 2}  (500 | 500 | 700)
    """

def load_univariate_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a 2-column CSV: x, y
    Returns:
        X : (n, 1)  — input feature
        y : (n,)    — target
    """

def load_bivariate_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a 3-column CSV: x1, x2, y
    Returns:
        X : (n, 2)  — input features
        y : (n,)    — target
    """
```

> **Rationale (Q5):** Two explicit functions rather than auto-detect — callers are unambiguous,
> and wrong-shape bugs fail loudly at load time rather than silently downstream.

---

## `shared/utils/data.py`

```python
def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffles and splits X and y.
    Returns: X_train, X_test, y_train, y_test
    """
```

---

## `Assignment1/models/perceptron.py`

```python
class Perceptron:
    def __init__(self, n_features: int, activation: str = 'sigmoid'):
        """
        activation: 'sigmoid' | 'tanh' | 'linear'
        Weights initialised to small random values ~ N(0, 0.01).
        """
        self.weights : np.ndarray  # shape (n_features,)
        self.bias    : float

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes net = X @ w + b, then applies activation.
        X   : (n_samples, n_features)
        out : (n_samples,)  — post-activation, NOT thresholded
        """

    def gradient(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Computes weight and bias gradients given the forward output.
        y      : (n_samples,)  — true targets
        output : (n_samples,)  — result of forward(X), passed in explicitly

        delta  = (output - y) * activation_derivative(output)
        grad_w = X.T @ delta / n
        grad_b = delta.mean()

        Returns:
            grad_w : (n_features,)
            grad_b : float
        """

    def activation_derivative(self, output: np.ndarray) -> np.ndarray:
        """
        Derivative of the activation w.r.t. net input, expressed in terms of output.
        sigmoid : output * (1 - output)
        tanh    : 1 - output ** 2
        linear  : ones_like(output)
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns decision-ready output:
        sigmoid → threshold at 0.5 → {0, 1}
        tanh    → threshold at 0   → {-1, +1}
        linear  → forward() as-is  (regression)
        """
```

> **Rationale (Q1):** Optimizer's job is *how to use gradients*, not *what this model's derivative is*.
> `gradient()` and `activation_derivative()` are independently unit-testable.

> **Rationale (Q2):** `predict()` for binary sub-classifiers returns the activation-specific encoding
> ({0,1} or {-1,+1}). The OAO wrapper — not the data loader — handles re-encoding class labels
> before calling `fit()`. Data labels stay as {0,1,2} throughout.

---

## `Assignment1/optimizers/gradient_descent.py`

```python
class GradientDescent:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def fit(
        self,
        model: Perceptron,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100
    ) -> list[float]:
        """
        Full-batch gradient descent. Updates model.weights and model.bias in-place.

        Each epoch:
            output         = model.forward(X)
            grad_w, grad_b = model.gradient(X, y, output)   # delegate to model
            model.weights -= lr * grad_w
            model.bias    -= lr * grad_b

        Vectorised — no Python loops over samples.

        Returns:
            mse_history : list[float]  — mean((y - output)²) per epoch

        Classification metrics (accuracy, precision, recall) are computed
        separately after training using shared/metrics/classification.py.
        """
```

> **Rationale (Q3):** Full-batch is the standard interpretation of "gradient descent" (SGD is named
> explicitly in literature). Vectorised `X.T @ delta / n` is both faster and mathematically clear.

> **Rationale (Q4):** Optimizer returns only MSE — it is an optimiser, not an evaluator.
> Classification metrics belong in dedicated evaluation functions called after training.

---

## `Assignment1/models/one_vs_one.py`

```python
class OneAgainstOne:
    def __init__(self, n_classes: int, n_features: int, activation: str = 'sigmoid'):
        """
        Creates n*(n-1)/2 Perceptron instances, one per class pair.
        Pairs stored as dict keyed by (i, j) with i < j.
        """
        self.classifiers : dict[tuple[int,int], Perceptron]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,            # labels {0, 1, ..., n_classes-1}
        optimizer: GradientDescent,
        epochs: int = 100
    ) -> dict[tuple[int,int], list[float]]:
        """
        For each pair (i, j):
          1. Filter X, y to only samples from class i and class j.
          2. Re-encode: class i → positive label, class j → negative label
             (0/1 for sigmoid, -1/+1 for tanh — determined by classifier's activation).
          3. Call optimizer.fit(classifier, X_pair, y_pair, epochs).

        Returns dict of {(i,j): mse_history} for plotting per-pair error curves.
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Majority vote across all classifiers.
        Each classifier votes for one of its two classes per sample.
        Class with most votes wins.

        Tie-breaking rule: lowest class index wins.
        (Deterministic, documented, simple.)

        Returns: (n_samples,) — predicted class labels {0, 1, ..., n_classes-1}
        """

    def predict_pair(self, X: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
        """
        Evaluates X using only the sub-classifier for class pair (i, j).
        Used for plotting pairwise decision boundaries as required by the assignment.

        pair : (i, j) with i < j
        Returns: (n_samples,) — binary predictions in the classifier's encoding
        """
```

---

## Summary Table

| Module | Key method | In → Out |
|---|---|---|
| `data_loader` | `load_ls_data(dir)` | path → `(X: n×2, y: n)` |
| `data_loader` | `load_nls_data(file)` | path → `(X: n×2, y: n)` |
| `data_loader` | `load_univariate_data(file)` | path → `(X: n×1, y: n)` |
| `data_loader` | `load_bivariate_data(file)` | path → `(X: n×2, y: n)` |
| `shared/utils` | `train_test_split(X, y, ratio)` | arrays → 4 arrays |
| `Perceptron` | `forward(X)` | `(n,d)` → `(n,)` raw output |
| `Perceptron` | `gradient(X, y, output)` | arrays → `(grad_w, grad_b)` |
| `Perceptron` | `predict(X)` | `(n,d)` → `(n,)` decisions |
| `GradientDescent` | `fit(model, X, y, epochs)` | — → `mse_hist` |
| `OneAgainstOne` | `fit(X, y, optimizer, epochs)` | — → `{pair: mse_hist}` |
| `OneAgainstOne` | `predict(X)` | `(n,d)` → `(n,)` class labels |
| `OneAgainstOne` | `predict_pair(X, pair)` | `(n,d)` + pair → `(n,)` binary |
