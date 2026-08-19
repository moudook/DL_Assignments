import numpy as np

class Perceptron:
    def __init__(self, n_features:int, activation: str="sigmoid"):
        self.weights: np.ndarray = np.random.randn(n_features) * 0.01
        self.bias: float = 0.0
        self.activation = activation
    
    def sigmoid(self, x):
        return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, X):
        linear = np.dot(X, self.weights) + self.bias
        if self.activation == "sigmoid":
            return self.sigmoid(linear)
        elif self.activation=="tanh":
            return self.tanh(linear)
        elif self.activation=="linear":
            return linear
        else:
            raise ValueError("Unknown activation function")

    def gradient(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> tuple[np.ndarray, float]:
        delta = (output-y)*self.activation_derivative(output)
        grad_w = np.dot(X.T, delta)/X.shape[0]
        grad_b = delta.mean()
        return grad_w, grad_b
    
    def activation_derivative(self, output: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return output*(1-output)
        elif self.activation=="tanh":
            return 1-output**2
        elif self.activation=="linear":
            return np.ones_like(output)
        else:
            raise ValueError(f"Unknown activation function {self.activation}")

    def predict(self, X: np.ndarray):
        output = self.forward(X)
        if self.activation=='sigmoid':
            return (output>=0.5).astype(int)
        elif self.activation=='tanh':
            return np.where(output>=0.0, 1, -1)
        elif self.activation=='linear':
            return output
        else:
            raise ValueError(f"Unknown activation function {self.activation}")
        