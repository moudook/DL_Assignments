# Mostly for test_train_split kinda work, doesn't make sense 
# to have shared code for loading the data as it'll just depend
# on the assingment on the type of data being dealt with
# But could have some funcitions regarding data loading like dealing with csvs or something
import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, stratify: bool = False, seed: int = 42):
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("Train ratio must be between 0 and 1")
        
    rng = np.random.default_rng(seed)
    
    if not stratify:
        # Standard random split (Use this for Regression)
        num_rows = X.shape[0]
        indices = np.arange(num_rows)
        rng.shuffle(indices)
        train_count = int(num_rows * train_ratio)
        train_idx = indices[:train_count]
        test_idx = indices[train_count:]
        
    else:
        # Stratified split (Use this for Classification)
        train_idx, test_idx = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            n_train = int(len(idx) * train_ratio)
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])
            
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        
        # Crucial: Shuffle the final combined indices 
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]