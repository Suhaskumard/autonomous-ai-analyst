"""Metrics.

Three numbers were not enough to judge a model. Accuracy alone is actively
misleading on imbalanced data — a constant predictor on a 95/5 split scores
95% — so every classification run now also reports balanced accuracy, macro
F1, and ROC-AUC, and selection happens on macro F1 rather than accuracy.
Regression gains MAE, R², and MAPE alongside RMSE.

`accuracy`, `precision`, `recall`, `mse` and `rmse` keep their original
meanings so stored artifacts and the existing UI stay readable.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# The metric each problem type is *selected* on, and its direction.
SELECTION_METRIC = {"classification": "f1_macro", "regression": "rmse"}
HIGHER_IS_BETTER = {"f1_macro": True, "rmse": False}

# Below this share of non-zero actuals, a MAPE is more misleading than absent.
MAPE_MIN_COVERAGE = 0.5


def evaluate_classification(y_true, y_pred, y_proba=None, labels=None) -> dict:
    """Accuracy plus the metrics that survive class imbalance.

    `y_proba` is optional because not every estimator exposes predict_proba;
    ROC-AUC is simply absent when it cannot be computed rather than faked.
    """
    metrics = {
        # Kept for continuity with earlier runs and the existing UI.
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        # The imbalance-aware view.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    roc_auc = _roc_auc(y_true, y_proba, labels)
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    return metrics


def evaluate_regression(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
    }
    mape = _mape(y_true, y_pred)
    if mape is not None:
        metrics["mape"] = mape
    return metrics


def evaluate(problem_type: str, y_true, y_pred, y_proba=None, labels=None) -> dict:
    if problem_type == "classification":
        return evaluate_classification(y_true, y_pred, y_proba=y_proba, labels=labels)
    return evaluate_regression(y_true, y_pred)


def per_class_report(y_true, y_pred, class_names: list[str]) -> list[dict]:
    """Precision/recall/F1/support for each class.

    A macro average hides which class the model is failing on; this is the
    breakdown that shows a minority class scoring zero while accuracy looks
    healthy.
    """
    labels = list(range(len(class_names))) if class_names else sorted(set(np.asarray(y_true).tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    for index, label in enumerate(labels):
        name = class_names[label] if class_names and label < len(class_names) else str(label)
        rows.append(
            {
                "label": str(name),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return rows


def _roc_auc(y_true, y_proba, labels) -> float | None:
    """Binary AUC from the positive column, multiclass as weighted one-vs-rest."""
    if y_proba is None:
        return None
    y_proba = np.asarray(y_proba)
    present = np.unique(np.asarray(y_true))
    if len(present) < 2:
        return None  # AUC is undefined against a single observed class
    try:
        if y_proba.ndim == 1 or y_proba.shape[1] == 2:
            scores = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
            return float(roc_auc_score(y_true, scores))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=labels))
    except Exception as exc:  # a fold missing a class, mismatched columns, ...
        logger.debug("ROC-AUC unavailable: %s", exc)
        return None


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean absolute percentage error over the rows where it is defined.

    MAPE divides by the actual, so zeros make it explode. Reporting a number
    computed from a handful of non-zero rows would be worse than reporting
    nothing, hence the coverage floor.
    """
    if len(y_true) == 0:
        return None
    scale = np.abs(y_true)
    usable = scale > np.finfo(float).eps
    if usable.sum() / len(y_true) < MAPE_MIN_COVERAGE:
        return None
    return float(np.mean(np.abs((y_true[usable] - y_pred[usable]) / scale[usable])) * 100.0)
