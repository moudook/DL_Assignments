import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os

def evaluate_model(model, data_loader, device):
    """
    Evaluates a model and returns predictions, targets, and accuracy.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return all_targets, all_preds, acc

def plot_confusion_matrix(targets, preds, classes, title, save_path=None):
    """
    Plots and optionally saves a confusion matrix.
    """
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_curves(results, arch_name, save_dir):
    """
    Plots superimposed training loss curves for all optimizers for a given architecture.
    """
    plt.figure(figsize=(10, 6))
    for opt_name, history in results[arch_name].items():
        plt.plot(history['train_loss'], label=opt_name)
    
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Error (Loss)')
    plt.title(f'Training Error vs Epochs for {arch_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{arch_name}_training_error.png')
    plt.savefig(save_path)
    plt.close()
