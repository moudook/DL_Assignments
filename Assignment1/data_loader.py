import numpy as np
import os

def load_ls_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.txt'))

    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}. Did you extract the data?")

    X_parts, y_parts = [], []
    for label, filename in enumerate(files):
        data = np.loadtxt(os.path.join(data_dir, filename))
        X_parts.append(data)
        y_parts.append(np.full(len(data), label))

    return np.vstack(X_parts), np.concatenate(y_parts)

def load_nls_data(filepath: str, class_sizes: list[int]=[500, 500, 700]) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = np.loadtxt(filepath, skiprows=1)

    X_parts, y_parts = [], []
    start = 0
    for label, size in enumerate(class_sizes):
        X_parts.append(data[start:start+size])
        y_parts.append(np.full(size, label))
        start+=size
    return np.vstack(X_parts), np.concatenate(y_parts)

def load_univariate_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.loadtxt(filepath, delimiter=',')
    return data[:, :-1], data[:, -1]

def load_bivariate_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    # Keeping both functions separate just in case
    return load_univariate_data(filepath)