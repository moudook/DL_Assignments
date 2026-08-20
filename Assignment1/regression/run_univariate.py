import sys
import os
import numpy as np

assignment1_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(assignment1_dir)
repo_root = os.path.dirname(assignment1_dir)
sys.path.append(repo_root)
np.random.seed(42)

from data_loader import load_univariate_data
from models.perceptron import Perceptron
from optimizers.gradient_descent import GradientDescent
from shared.utils.data import train_test_split
from shared.metrics.regression import get_rmse, get_percent_rmse
from shared.utils.plotting import (
    plot_error_curves,
    plot_regression_univariate,
    plot_scatter_target_vs_output,
)

DATA_DIR = "data/Group11/Regression/UnivariateData/11.csv"
X, y = load_univariate_data(DATA_DIR)

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ── Train ──────────────────────────────────────────────────────────────────
model = Perceptron(n_features=1, activation='linear')
optimizer = GradientDescent(lr=0.01)
histories = optimizer.fit(model, X_train, y_train, epochs=100)

train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

# ── Metrics ────────────────────────────────────────────────────────────────
train_rmse         = get_rmse(y_train, train_pred)
test_rmse          = get_rmse(y_test,  test_pred)
train_percent_rmse = get_percent_rmse(y_train, train_pred)
test_percent_rmse  = get_percent_rmse(y_test,  test_pred)

print("\n============ Univariate Regression ============")
print(f"Train RMSE:      {train_rmse:.4f}")
print(f"Train % RMSE:    {train_percent_rmse:.2f}%")
print(f"Test  RMSE:      {test_rmse:.4f}")
print(f"Test  % RMSE:    {test_percent_rmse:.2f}%")
print("===============================================\n")

# ── Plot 1: Error vs Epochs ────────────────────────────────────────────────
plot_error_curves(histories,
                  title="Univariate Regression — Error vs Epochs",
                  save_path="outputs/regression/univariate/error_curve.png")

# ── Plot 3: Model output superimposed on target ────────────────────────────
plot_regression_univariate(X_train, y_train, train_pred,
                            title="Univariate — Train: Model vs Target",
                            save_path="outputs/regression/univariate/model_vs_target_train.png")

plot_regression_univariate(X_test, y_test, test_pred,
                            title="Univariate — Test: Model vs Target",
                            save_path="outputs/regression/univariate/model_vs_target_test.png")

# ── Plot 4: Scatter — target (x) vs model output (y) ──────────────────────
plot_scatter_target_vs_output(y_train, train_pred,
                               title="Univariate — Train: Target vs Output",
                               save_path="outputs/regression/univariate/scatter_train.png")

plot_scatter_target_vs_output(y_test, test_pred,
                               title="Univariate — Test: Target vs Output",
                               save_path="outputs/regression/univariate/scatter_test.png")