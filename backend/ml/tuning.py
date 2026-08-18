"""Hyperparameter search under a time budget.

`n_estimators=100` hardcoded for every dataset is a guess, not a choice. This
runs a randomized search over each model's parameter space, cross-validated on
the development split only — the held-out test set is never part of a search,
so the reported final score is not the score the tuner optimised against.

Randomized rather than exhaustive, and budgeted rather than open-ended: a grid
search over five models is minutes of CPU the user did not ask to spend. The
budget is wall-clock and checked between models, so a long search degrades to
"tuned the first few, kept defaults for the rest" instead of hanging.
"""

import logging
import time

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)

# Search spaces are deliberately small: a handful of meaningfully different
# settings each, not a sweep. The estimator sits at step "model" of a Pipeline.
SEARCH_SPACES: dict[str, dict] = {
    "LogisticRegression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
    "ElasticNet": {"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.15, 0.5, 0.85]},
    "RandomForest": {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [None, 6, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "ExtraTrees": {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [None, 6, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "DecisionTree": {"model__max_depth": [3, 5, 10, 20, None], "model__min_samples_leaf": [1, 2, 5, 10]},
    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_depth": [2, 3, 5],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_iter": [100, 200],
        "model__max_leaf_nodes": [15, 31, 63],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.03, 0.1, 0.3],
        "model__max_depth": [3, 6, 10],
        "model__subsample": [0.7, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__num_leaves": [15, 31, 63],
    },
    "SVM": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
    "SVR": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
}

# Regressor variants share their classifier's space.
SEARCH_SPACES["RandomForestRegressor"] = SEARCH_SPACES["RandomForest"]
SEARCH_SPACES["ExtraTreesRegressor"] = SEARCH_SPACES["ExtraTrees"]
SEARCH_SPACES["DecisionTreeRegressor"] = SEARCH_SPACES["DecisionTree"]
SEARCH_SPACES["GradientBoostingRegressor"] = SEARCH_SPACES["GradientBoosting"]
SEARCH_SPACES["HistGradientBoostingRegressor"] = SEARCH_SPACES["HistGradientBoosting"]
SEARCH_SPACES["XGBoostRegressor"] = SEARCH_SPACES["XGBoost"]
SEARCH_SPACES["LightGBMRegressor"] = SEARCH_SPACES["LightGBM"]

DEFAULT_BUDGET_SECONDS = 60
MAX_BUDGET_SECONDS = 900
MIN_ITERATIONS = 4
MAX_ITERATIONS = 24


def space_for(model_name: str) -> dict | None:
    return SEARCH_SPACES.get(model_name)


def iterations_for(space: dict, budget_seconds: float) -> int:
    """How many candidates to draw, bounded by the space and by the budget."""
    combinations = 1
    for values in space.values():
        combinations *= len(values)
    by_budget = int(MIN_ITERATIONS + budget_seconds / 10)
    return max(1, min(combinations, MAX_ITERATIONS, by_budget))


class Budget:
    """A wall-clock allowance, checked between models rather than mid-fit.

    Interrupting a running fit would leave a half-trained estimator; asking
    "is there time for another model?" keeps every result complete.
    """

    def __init__(self, seconds: float):
        self.seconds = max(0.0, float(seconds))
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed)

    def exhausted(self, need: float = 0.0) -> bool:
        return self.remaining <= need


def tune_pipeline(
    pipeline,
    model_name: str,
    X_dev,
    y_dev,
    cv,
    scoring: str,
    budget: Budget,
    fit_params: dict | None = None,
) -> tuple[object, dict | None]:
    """Search `pipeline`'s space on the development split.

    Returns (best_pipeline, report). The report is None when tuning was
    skipped — no search space, or no budget left — and the caller keeps the
    default-parameter pipeline rather than pretending a search happened.
    """
    space = space_for(model_name)
    if not space:
        return pipeline, None
    if budget.exhausted():
        logger.info("Tuning budget exhausted before %s; keeping defaults.", model_name)
        return pipeline, None

    n_iter = iterations_for(space, budget.remaining)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=True,
        error_score=np.nan,
    )
    started = time.monotonic()
    try:
        search.fit(X_dev, y_dev, **(fit_params or {}))
    except Exception as exc:
        logger.warning("Tuning failed for %s, keeping defaults: %s", model_name, exc)
        return pipeline, None

    took = time.monotonic() - started
    best_score = float(search.best_score_) if np.isfinite(search.best_score_) else None
    if best_score is None:
        logger.warning("Every tuning candidate failed for %s; keeping defaults.", model_name)
        return pipeline, None

    report = {
        "model": model_name,
        "candidates": int(n_iter),
        "seconds": round(took, 2),
        "best_params": {key.replace("model__", ""): _plain(value) for key, value in search.best_params_.items()},
        "best_cv_score": best_score,
        "scoring": scoring,
    }
    logger.info("Tuned %s in %.1fs over %d candidates", model_name, took, n_iter)
    return search.best_estimator_, report


def _plain(value):
    """numpy scalars do not survive json.dump; unwrap them."""
    if isinstance(value, np.generic):
        return value.item()
    return value
