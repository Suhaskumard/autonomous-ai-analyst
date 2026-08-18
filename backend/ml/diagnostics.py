"""Model diagnostics the dashboard plots.

A confusion matrix (classification) or residual plot (regression) says what a
single accuracy number cannot: *where* the model is wrong. Computed on the
held-out test split, so these describe generalisation rather than fit.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Plotting every residual of a large test set is slow to send and unreadable;
# a deterministic sample is enough to show the shape.
MAX_RESIDUAL_POINTS = 400
MAX_CONFUSION_CLASSES = 12
MAX_CORRELATION_COLUMNS = 12


def build_confusion_matrix(y_true, y_pred, class_names: list[str]) -> dict[str, Any] | None:
    """Counts per (true, predicted) pair, labelled with real class names."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return None

    labels = np.unique(np.concatenate([y_true, y_pred]))
    truncated = False
    if len(labels) > MAX_CONFUSION_CLASSES:
        # Keep the classes that actually occur most in the truth column.
        counts = pd.Series(y_true).value_counts()
        labels = np.array(sorted(counts.head(MAX_CONFUSION_CLASSES).index))
        truncated = True

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    def name_of(label) -> str:
        index = int(label)
        if class_names and 0 <= index < len(class_names):
            return str(class_names[index])
        return str(label)

    return {
        "labels": [name_of(label) for label in labels],
        "matrix": [[int(value) for value in row] for row in matrix],
        "truncated": truncated,
        "total": int(matrix.sum()),
    }


def build_residuals(y_true, y_pred) -> dict[str, Any] | None:
    """Predicted vs residual points, plus the summary a reader needs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return None

    residuals = y_true - y_pred
    count = y_true.size
    if count > MAX_RESIDUAL_POINTS:
        # Evenly spaced sample: deterministic and preserves the overall shape.
        index = np.linspace(0, count - 1, MAX_RESIDUAL_POINTS).astype(int)
    else:
        index = np.arange(count)

    return {
        "points": [
            {"predicted": float(y_pred[i]), "residual": float(residuals[i]), "actual": float(y_true[i])} for i in index
        ],
        "sampled": bool(count > MAX_RESIDUAL_POINTS),
        "total": int(count),
        "mean_residual": float(residuals.mean()),
        "std_residual": float(residuals.std()) if count > 1 else 0.0,
    }


def build_correlation_matrix(df: pd.DataFrame) -> dict[str, Any] | None:
    """Pearson correlation across numeric columns, for the heatmap."""
    numeric = df.select_dtypes(include=["number"])
    # Constant columns produce NaN correlations and an unreadable row.
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        return None

    truncated = False
    if numeric.shape[1] > MAX_CORRELATION_COLUMNS:
        # Keep the most variable columns — the ones worth looking at.
        keep = numeric.std(numeric_only=True).sort_values(ascending=False).head(MAX_CORRELATION_COLUMNS).index
        numeric = numeric[list(keep)]
        truncated = True

    matrix = numeric.corr(method="pearson").fillna(0.0)
    columns = [str(c) for c in matrix.columns]

    strongest: list[dict[str, Any]] = []
    values = matrix.to_numpy()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            strongest.append({"a": columns[i], "b": columns[j], "r": float(values[i][j])})
    strongest.sort(key=lambda item: abs(item["r"]), reverse=True)

    return {
        "columns": columns,
        "matrix": [[round(float(value), 4) for value in row] for row in values],
        "truncated": truncated,
        "strongest_pairs": strongest[:5],
    }


def build_diagnostics(
    problem_type: str,
    y_test,
    y_pred,
    class_names: list[str],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Everything the diagnostics panel needs, in one payload."""
    payload: dict[str, Any] = {
        "problem_type": problem_type,
        "correlation": build_correlation_matrix(df),
        "confusion_matrix": None,
        "residuals": None,
    }
    if problem_type == "classification":
        payload["confusion_matrix"] = build_confusion_matrix(y_test, y_pred, class_names)
    else:
        payload["residuals"] = build_residuals(y_test, y_pred)
    return payload
