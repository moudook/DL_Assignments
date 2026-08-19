# Mostly for test_train_split kinda work, doesn't make sense 
# to have shared code for loading the data as it'll just depend
# on the assingment on the type of data being dealt with
# But could have some funcitions regarding data loading like dealing with csvs or something

import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: float=0.7, seed: int=42):
    if(train_ratio<=0 or train_ratio>=1):
        raise ValueError("Train ratio must be between 0 and 1")
    num_rows = X.shape[0]
    indices = np.arange(num_rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_count = int(num_rows*train_ratio)
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]