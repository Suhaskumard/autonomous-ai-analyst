"""Model training.

Two rules this module exists to enforce:

1. The split happens before anything is fitted. Each candidate is a single
   sklearn Pipeline of (preprocessor -> estimator) fitted on the training split
   only, so imputer/scaler/encoder statistics never see the test rows. The
   fitted Pipeline is also the serving artifact, so predict.py loads one object
   instead of re-implementing the transform order.
2. Classification targets are label-encoded. XGBoost requires 0..n-1 integer
   classes and fails outright on string labels; the encoder is stored with the
   model and inverted at predict time.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from exceptions import TrainingError
from ml.evaluate import evaluate_classification, evaluate_regression

logger = logging.getLogger(__name__)

try:  # xgboost is optional; a missing wheel should not take the whole app down
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost is not installed; XGBoost models will be unavailable.")

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Bumped whenever preprocessing or training changes in a way that makes stored
# artifacts stale. It is part of the cache key, so old artifacts are ignored
# automatically instead of being silently reused.
PIPELINE_VERSION = "2"


def detect_problem_type(y: pd.Series) -> str:
    """Classification vs regression, without the old "few uniques -> classes" trap.

    The previous rule (<=20 uniques or ratio < 0.1 => classification) misread
    free text, ID columns, and genuinely low-cardinality regression targets.
    """
    non_null = y.dropna()

    if pd.api.types.is_bool_dtype(non_null):
        return "classification"

    if not pd.api.types.is_numeric_dtype(non_null):
        return "classification"

    # Continuous values are a regression target no matter how few of them there
    # are — 8 distinct prices are still prices.
    as_float = non_null.astype(float)
    if not np.all(np.isfinite(as_float)):
        as_float = as_float[np.isfinite(as_float)]
    if len(as_float) and not np.allclose(as_float, np.round(as_float)):
        return "regression"

    n_unique = int(non_null.nunique())
    if n_unique <= 20:
        return "classification"
    return "regression"


def get_model_registry(problem_type: str) -> dict:
    if problem_type == "classification":
        registry = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
            "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
            "SVM": SVC(probability=True, kernel="rbf", max_iter=2000),
        }
        if XGBOOST_AVAILABLE:
            registry["XGBoost"] = XGBClassifier(eval_metric="logloss", n_estimators=50, random_state=RANDOM_STATE)
        return registry

    registry = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "SVR": SVR(max_iter=2000),
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoostRegressor"] = XGBRegressor(n_estimators=50, random_state=RANDOM_STATE)
    return registry


def _encode_target(y: pd.Series, problem_type: str) -> tuple[np.ndarray, LabelEncoder | None]:
    """Map string classes to 0..n-1 so every estimator, XGBoost included, works.

    The encoder is fitted on the full target rather than the training split.
    That is not leakage: it learns only the set of class names, no statistic
    about the data. Fitting it on train alone would instead crash scoring the
    moment a class appeared only in the test split.
    """
    if problem_type != "classification":
        return y.to_numpy(), None
    encoder = LabelEncoder()
    return encoder.fit_transform(y.astype(str)), encoder


def _stratify_for(y_encoded: np.ndarray, problem_type: str):
    """Stratify only when every class has enough members for the split."""
    if problem_type != "classification":
        return None
    _, counts = np.unique(y_encoded, return_counts=True)
    if counts.min() < 2:
        logger.info("Skipping stratification: at least one class has a single member.")
        return None
    return y_encoded


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    mode: str = "auto",
    manual_model: str | None = None,
) -> dict:
    """Fit every candidate as preprocessor+estimator on the training split only."""
    problem_type = detect_problem_type(y)
    y_encoded, label_encoder = _encode_target(y, problem_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=_stratify_for(y_encoded, problem_type),
    )

    registry = get_model_registry(problem_type)

    # In 'auto' mode, skip the models that scale badly on large datasets.
    if mode == "auto" and len(X) > 5000:
        registry.pop("SVM", None)
        registry.pop("SVR", None)

    if mode == "manual" and manual_model:
        if manual_model not in registry:
            available = ", ".join(registry)
            raise TrainingError(f"Model '{manual_model}' is not available for {problem_type}. Available: {available}.")
        registry = {manual_model: registry[manual_model]}

    trained_models: dict[str, Pipeline] = {}
    performance_scores: dict[str, dict] = {}
    failed_models: dict[str, str] = {}

    for model_name, estimator in registry.items():
        try:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = (
                evaluate_classification(y_test, y_pred)
                if problem_type == "classification"
                else evaluate_regression(y_test, y_pred)
            )
            trained_models[model_name] = pipeline
            performance_scores[model_name] = metrics
        except Exception as exc:
            logger.warning("Model %s failed to train: %s", model_name, exc, exc_info=True)
            failed_models[model_name] = str(exc)

    if not trained_models:
        detail = "; ".join(f"{name}: {err}" for name, err in failed_models.items())
        raise TrainingError(f"Every candidate model failed to train. Details — {detail}")

    return {
        "problem_type": problem_type,
        "trained_models": trained_models,
        "performance_scores": performance_scores,
        "failed_models": failed_models,
        "label_encoder": label_encoder,
        "class_names": list(label_encoder.classes_) if label_encoder is not None else [],
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "pipeline_version": PIPELINE_VERSION,
    }
