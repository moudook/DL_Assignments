import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
import itertools

def plot_error_curves(histories: dict, title: str, save_path: str = None):
    plt.figure(figsize=(8, 5))
    for (i, j), mse in histories.items():
        plt.plot(mse, label=f'Class {i} vs {j}')
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