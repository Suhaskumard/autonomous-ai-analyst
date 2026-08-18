"""Class imbalance: name it, then do something about it.

On a 95/5 target, a model that always predicts the majority class scores 95%
accuracy and is worthless. Two defences here, plus the warning that makes the
situation visible in the first place:

* class weights, applied to every estimator that supports them — free, always
  available, and it changes the loss rather than the data;
* SMOTE, optional and off by default, applied *inside* the pipeline so
  resampling only ever touches the training rows of a fold, never the
  validation rows. Resampling before splitting would leak synthetic neighbours
  of validation rows into training and inflate every score.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:  # imbalanced-learn is optional; SMOTE is simply unavailable without it
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline

    SMOTE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    SMOTE_AVAILABLE = False
    logger.info("imbalanced-learn is not installed; SMOTE oversampling is unavailable.")

# Majority share above this is called out to the user.
IMBALANCE_WARN_RATIO = 0.65
# Imbalance ratio (majority/minority) above this is called severe.
SEVERE_IMBALANCE_RATIO = 10.0
# SMOTE interpolates between neighbours, so the smallest class needs members.
SMOTE_MIN_CLASS_SIZE = 6


def describe_imbalance(y_encoded: np.ndarray, class_names: list[str]) -> dict:
    """Class distribution plus a verdict on how skewed it is."""
    y_encoded = np.asarray(y_encoded)
    total = len(y_encoded)
    if total == 0:
        return {"imbalanced": False, "severe": False, "ratio": 1.0, "distribution": [], "warnings": []}

    labels, counts = np.unique(y_encoded, return_counts=True)
    order = np.argsort(-counts)
    distribution = [
        {
            "label": str(class_names[labels[i]]) if class_names and labels[i] < len(class_names) else str(labels[i]),
            "count": int(counts[i]),
            "share": float(counts[i] / total),
        }
        for i in order
    ]

    majority_share = float(counts.max() / total)
    ratio = float(counts.max() / max(counts.min(), 1))
    imbalanced = majority_share > IMBALANCE_WARN_RATIO or ratio >= SEVERE_IMBALANCE_RATIO
    severe = ratio >= SEVERE_IMBALANCE_RATIO

    warnings: list[str] = []
    if imbalanced:
        top = distribution[0]
        warnings.append(
            f"Class imbalance: '{top['label']}' is {top['share']:.1%} of rows "
            f"({ratio:.1f}:1 against the rarest class). Accuracy is misleading here — "
            "the run is selected on macro F1 and reports balanced accuracy alongside it."
        )
    if severe:
        rarest = distribution[-1]
        warnings.append(
            f"The rarest class '{rarest['label']}' has only {rarest['count']:,} rows, "
            "so its per-class scores rest on very few examples."
        )

    return {
        "imbalanced": bool(imbalanced),
        "severe": bool(severe),
        "ratio": ratio,
        "majority_share": majority_share,
        "distribution": distribution,
        "warnings": warnings,
    }


def supports_class_weight(estimator) -> bool:
    return "class_weight" in estimator.get_params()


def apply_class_weight(estimator):
    """Set class_weight='balanced' where the estimator understands it."""
    if supports_class_weight(estimator):
        estimator.set_params(class_weight="balanced")
    return estimator


def sample_weight_for(y_encoded: np.ndarray) -> np.ndarray:
    """Per-row weights inversely proportional to class frequency.

    The fallback for estimators with no class_weight parameter — XGBoost and
    LightGBM take this through fit(sample_weight=...).
    """
    y_encoded = np.asarray(y_encoded)
    labels, counts = np.unique(y_encoded, return_counts=True)
    weight_by_label = {
        label: len(y_encoded) / (len(labels) * count) for label, count in zip(labels, counts, strict=True)
    }
    return np.array([weight_by_label[label] for label in y_encoded], dtype=float)


def can_use_smote(y_encoded: np.ndarray) -> tuple[bool, str]:
    """Whether SMOTE is usable here, and why not when it is not."""
    if not SMOTE_AVAILABLE:
        return False, "imbalanced-learn is not installed, so SMOTE is unavailable."
    _, counts = np.unique(np.asarray(y_encoded), return_counts=True)
    if len(counts) < 2:
        return False, "SMOTE needs at least two classes."
    if counts.min() < SMOTE_MIN_CLASS_SIZE:
        return False, (
            f"The smallest class has {counts.min()} rows; SMOTE needs at least "
            f"{SMOTE_MIN_CLASS_SIZE} to interpolate between neighbours. Used class weights instead."
        )
    return True, ""


def build_pipeline(preprocessor, estimator, y_encoded: np.ndarray | None = None, use_smote: bool = False):
    """(preprocessor -> [SMOTE] -> estimator) as one fittable object.

    When SMOTE is active the pipeline is imbalanced-learn's, which applies
    resamplers on fit and skips them on predict. That is what keeps synthetic
    rows out of every validation fold and out of scoring.
    """
    steps = [("preprocessor", preprocessor)]
    if use_smote and y_encoded is not None:
        _, counts = np.unique(np.asarray(y_encoded), return_counts=True)
        # k must be strictly below the smallest class size.
        k_neighbors = max(1, min(5, int(counts.min()) - 1))
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)))
        steps.append(("model", estimator))
        return ImbalancedPipeline(steps=steps)

    from sklearn.pipeline import Pipeline

    steps.append(("model", estimator))
    return Pipeline(steps=steps)
