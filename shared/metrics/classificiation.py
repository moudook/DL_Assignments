import numpy as np
def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean(y_pred == y_true)

def get_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray):
    n_classes = np.unique(np.concatenate([y_pred, y_true])).size
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)]+=1
    return cm

def get_precision(cm: np.ndarray):
    n_classes = cm.shape[0]
    precisions = []
    for i in range(n_classes):
        tp = cm[i, i]
        row_sum = cm[:, i].sum()
        precisions.append(tp / (row_sum) if (row_sum) > 0 else 0)
    return np.array(precisions)

def get_recall(cm: np.ndarray):
    n_classes = cm.shape[0]
    recalls = []
    for i in range(n_classes):
        tp = cm[i, i]
        col_sum = cm[i, :].sum()
        recalls.append(tp/col_sum if col_sum > 0 else 0)
    return np.array(recalls)

def get_f1_score(precisions:list[float], recalls:list[float])->list[float]:
    f1_scores = []
    for p, r in zip(precisions, recalls):
        f1_scores.append((2*p*r)/(p+r) if (p+r) > 0 else 0)
    return np.array(f1_scores)

def generate_summary(y_pred: np.ndarray, y_true: np.ndarray, model_name: str):
    cm = get_confusion_matrix(y_pred, y_true)
    precisions = get_precision(cm)
    recalls = get_recall(cm)
    f1_scores = get_f1_score(precisions, recalls)
    accuracy = calculate_accuracy(y_pred, y_true)
    summary = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "confusion_matrix": cm
    }
    return summary