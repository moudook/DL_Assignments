import numpy as np

def get_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred)**2))

def get_percent_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (get_rmse(y_true, y_pred) / np.mean(y_true)) * 100
    