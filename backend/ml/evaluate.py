import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score


def evaluate_classification(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_regression(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {"mse": mse, "rmse": float(np.sqrt(mse))}
