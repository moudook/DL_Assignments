# Imports
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ActivationFunction(y, B):
    # Sigmoid activation, outputs in (0, 1)
    return 1 / (1 + math.exp(-1 * y * B))


# Load data
df = pd.read_csv("data.csv")
print(df)
print(df.head())

# Build feature matrix: inputs x0 (bias), x1, x2
data = df[['x1', 'x2']]
data.insert(loc=0, column='x0', value=1)

# Labels
label = df[['label']]

# Random weight initialization (one per feature)
from random import randrange as rdr
weights = [rdr(-2, 2), rdr(-2, 2), rdr(-2, 2)]

epoch = 200
B = 0.7  # sigmoid slope

# Training: for each epoch, sweep all rows, update on misclassification
for i in range(1, epoch + 1):
    count = 0
    misclass = 0
    eta = 1 / i  # decaying learning rate
    while count < len(data):
        y = np.dot(weights, data.iloc[count])  # weighted sum
        AF = ActivationFunction(y, B)           # sigmoid activation
        error = label.iloc[count].label - y    # prediction error
        if error != 0:
            # Gradient update toward correct label
            sigmoid_derivative = B * AF * (1 - AF)
            weights = weights + np.array(
                eta * error * sigmoid_derivative * data.iloc[count]
            )
            misclass += 1
        count += 1

print("output weights")
print(weights)

# Test on a sample input
inp = [1, 2, 3]
prod = np.dot(weights, inp)
print(1 if prod > 0.5 else 0)

# Plot data and decision boundary
line = (-1) * (weights[1] * df['x1'] + weights[0]) / weights[2]
plt.scatter(df['x1'], df['x2'], c=df['label'])
plt.plot(df['x1'], line, c='red')
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()