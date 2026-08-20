from models.perceptron import Perceptron
import numpy as np
from optimizers.gradient_descent import GradientDescent

VALID_ACTIVATIONS = ['sigmoid', 'tanh']

class OneAgainstOne:
    def __init__(self, n_classes:int, n_features:int, activation: str='sigmoid'):
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f'Invalid activation function: {activation}')
        if not isinstance(n_classes, int) or n_classes<=1:
            raise ValueError(f'Number of classes must be an integer greater than 1: {n_classes}')
        if not isinstance(n_features, int) or n_features<=0:
            raise ValueError(f'Number of features must be a positive integer: {n_features}')
        self.n_classes = n_classes
        self.n_features = n_features
        self.classifiers: dict[tuple[int, int], Perceptron] = {
            (i, j): Perceptron(n_features, activation)
            for i in range(n_classes)
            for j in range(i+1, n_classes)
        }
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: GradientDescent,
        epochs: int = 100
    ) -> dict[tuple[int,int], list[float]]:
        histories = {}
        for (i, j), classifier in self.classifiers.items():
            mask = (y==i) | (y==j)
            X_pair = X[mask]
            y_pair = y[mask]
            if classifier.activation=='sigmoid':
                y_encoded = np.where(y_pair==i, 1, 0)
            elif classifier.activation=='tanh':
                y_encoded = np.where(y_pair==i, 1, -1)
            histories[(i, j)] = optimizer.fit(classifier, X_pair, y_encoded, epochs)
        return histories
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.zeros((X.shape[0], self.n_classes), dtype=int)
        for (i, j), classifier in self.classifiers.items():
            pred = classifier.predict(X)
            votes[:, i] += (pred == 1).astype(int)
            if classifier.activation == 'sigmoid':
                votes[:, j] += (pred == 0).astype(int)
            else:
                votes[:, j] += (pred == -1).astype(int)
        return np.argmax(votes, axis=1)
    
    def predict_pair(self, X: np.ndarray, pair: tuple[int, int])->np.ndarray:
        i,j = pair
        classifier = self.classifiers[(i, j)]
        pred = classifier.predict(X)
        return np.where(pred==1, i, j)
