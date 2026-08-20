import sys
import os
assignment1_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(assignment1_dir)
repo_root = os.path.dirname(assignment1_dir)
sys.path.append(repo_root)

from data_loader import load_nls_data
from models.one_vs_one import OneAgainstOne
from optimizers.gradient_descent import GradientDescent
from shared.utils.data import train_test_split
from shared.metrics.classificiation import *
from shared.utils.plotting import plot_error_curves, plot_decision_regions

DATA_PATH = "data/Group11/Classification/NLS_Group11.txt"
X, y = load_nls_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

logisitc_model = OneAgainstOne(3, 2, activation='sigmoid')
logistic_histories = logisitc_model.fit(X_train, y_train, optimizer=GradientDescent(lr=0.01), epochs=100)
logistic_pred = logisitc_model.predict(X_test)

logisitc_summary = generate_summary(logistic_pred, y_test, "Perceptron with Logisitc activation function")
print(f"=================={logisitc_summary["model_name"]}==================")
print("Accuracy: ", logisitc_summary["accuracy"])
print("Precisions: ", logisitc_summary["precisions"])
print("Recalls: ", logisitc_summary["recalls"])
print("F1 scores: ", logisitc_summary["f1_scores"])
print("Confusion matrix:\n", logisitc_summary["confusion_matrix"])
print("Mean Precision: ", logisitc_summary["precisions"].mean())
print("Mean Recall: ", logisitc_summary["recalls"].mean())
print("Mean F1 Score: ", logisitc_summary["f1_scores"].mean())
print("\n")
plot_error_curves(logistic_histories, "Perceptron with Logisitc activation function", "outputs/nls/error_curves_logistic.png")
plot_decision_regions(logisitc_model, X_train, y_train, "Perceptron with Logisitc activation function", save_path="outputs/nls/decision_regions_logistic.png")
for pair in [(0,1), (0,2), (1,2)]:
    plot_decision_regions(logisitc_model, X_train, y_train, "Perceptron with Logisitc activation function", save_path=f"outputs/nls/decision_regions_logistic_{pair}.png", pair=pair)


tanh_model = OneAgainstOne(3, 2, activation='tanh')
tanh_histories = tanh_model.fit(X_train, y_train, optimizer=GradientDescent(lr=0.01), epochs=100)
tanh_pred = tanh_model.predict(X_test)

tanh_summary = generate_summary(tanh_pred, y_test, "Perceptron with Tan hyperbolic activation function")
print(f"=================={tanh_summary["model_name"]}==================")
print("Accuracy: ", tanh_summary["accuracy"])
print("Precisions: ", tanh_summary["precisions"])
print("Recalls: ", tanh_summary["recalls"])
print("F1 scores: ", tanh_summary["f1_scores"])
print("Confusion matrix:\n", tanh_summary["confusion_matrix"])
print("Mean Precision: ", tanh_summary["precisions"].mean())
print("Mean Recall: ", tanh_summary["recalls"].mean())
print("Mean F1 Score: ", tanh_summary["f1_scores"].mean())
plot_error_curves(tanh_histories, "Perceptron with Tan hyperbolic activation function", "outputs/nls/error_curves_tanh.png")
plot_decision_regions(tanh_model, X_train, y_train, "Perceptron with Tan hyperbolic activation function", save_path="outputs/nls/decision_regions_tanh.png")


for pair in [(0,1), (0,2), (1,2)]:
    plot_decision_regions(tanh_model, X_train, y_train, "Perceptron with Tan hyperbolic activation function", save_path=f"outputs/nls/decision_regions_tanh_{pair}.png", pair=pair)
