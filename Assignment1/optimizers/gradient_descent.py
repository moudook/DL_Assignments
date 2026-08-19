from models.perceptron import Perceptron
import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def fit(self, model: Perceptron, X: np.ndarray, y: np.ndarray, epochs: int=100)->list[float]:
        mse_history = []
        for epoch in range(epochs):
            output = model.forward(X)
            grad_w, grad_b = model.gradient(X, y, output)
            model.weights -= self.lr*grad_w
            model.bias -= self.lr*grad_b
            mse_history.append(float(np.mean((y-output)**2)))
        return mse_history
