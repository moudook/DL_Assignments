import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
import itertools
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_error_curves(histories, title: str, save_path: str = None):
    """histories: dict[(i,j) -> list] for classification, or plain list for regression."""
    plt.figure(figsize=(8, 5))
    if isinstance(histories, dict):
        for key, mse in histories.items():
            label = f'Class {key[0]} vs {key[1]}' if isinstance(key, tuple) else str(key)
            plt.plot(mse, label=label)
    else:
        plt.plot(histories, label='Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_decision_regions(model, X: np.ndarray, y: np.ndarray,
                          title: str, save_path: str = None, pair: tuple = None,
                          margin: float = 0.5, resolution: int = 400):
    """
    Plots 2D decision regions for a classifier.
    
    Args:
        model: The trained classifier.
        X (np.ndarray): 2D array of features.
        y (np.ndarray): 1D array of labels.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the figure.
        pair (tuple, optional): (class_i, class_j) for pairwise OvO plotting.
        margin (float): Extra space around the min/max data points.
        resolution (int): Number of grid points per axis.
    """
    if X.shape[1] != 2:
        raise ValueError(f"X must have exactly 2 features for this 2D plot, got {X.shape[1]}")

    # 1. Build grid dynamically based on resolution and margin
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 2. Predict on grid
    if pair is None:
        Z = model.predict(grid)
        classes = np.unique(y)
    else:
        Z = model.predict_pair(grid, pair)
        classes = np.array(list(pair))
    
    Z = Z.reshape(xx.shape)

    # 3. Dynamic Colors and Markers (supports up to 10 classes easily)
    base_cmap = plt.get_cmap('tab10')
    unique_z = np.unique(Z)
    
    # Create light background colormap by adding transparency (alpha)
    bg_colors = [base_cmap(i) for i in range(len(unique_z))]
    cmap = ListedColormap(bg_colors)
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'])

    # 4. Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    for idx, cls in enumerate(classes):
        mask = y == cls
        plt.scatter(X[mask, 0], X[mask, 1],
                    color=base_cmap(idx), marker=next(markers),
                    edgecolor='black',
                    label=f'Class {cls}', s=40, alpha=0.9)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()

def plot_regression_univariate(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                                title: str, save_path: str = None):
    x_vals = X.ravel()
    sort_idx = np.argsort(x_vals)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals, y_true, s=15, alpha=0.5, label='Actual', color='steelblue')
    plt.plot(x_vals[sort_idx], y_pred[sort_idx], color='crimson',
             linewidth=2, label='Model output')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regression_bivariate(model, X: np.ndarray, y_true: np.ndarray,
                               title: str, save_path: str = None):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    zz = model.predict(grid).reshape(xx1.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx1, xx2, zz, alpha=0.4, color='orange', label='Model surface')
    ax.scatter(X[:, 0], X[:, 1], y_true, color='steelblue', s=10, alpha=0.6)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_target_vs_output(y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str, save_path: str = None):
    """Scatter plot with target on x-axis and model output on y-axis."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=15, alpha=0.5, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction (y=x)')
    plt.xlabel('Target (y_true)')
    plt.ylabel('Model output (y_pred)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

