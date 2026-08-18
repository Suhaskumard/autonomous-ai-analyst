"""A model card: what was trained, on what, how, and with which libraries.

A metrics blob three months old is not interpretable on its own. "RandomForest,
0.87" says nothing about which columns went in, whether the classes were
balanced, what the tuner changed, or which scikit-learn produced it — and a
scikit-learn upgrade can move a score without anything in the project changing.

Everything here is already known at the end of a run; the point is to write it
down in one place, in a shape that still reads years later.
"""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

# The libraries whose version can move a score. Anything absent is recorded as
# absent, because "xgboost was not installed" explains a missing model.
TRACKED_PACKAGES = (
    "scikit-learn",
    "numpy",
    "pandas",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "imbalanced-learn",
)


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def build_model_card(
    *,
    run_key: str,
    timestamp: str,
    dataset: dict,
    prepared: dict,
    trained: dict,
    health: dict,
    explanation_method: str,
    feature_importance: list[dict],
) -> dict:
    """Assemble the card from what the pipeline already computed."""
    problem_type = trained["problem_type"]
    selected = trained["selected_model"]
    selection_metric = trained["selection_metric"]

    return {
        "schema": 1,
        "run_key": run_key,
        "created_at": timestamp,
        "pipeline_version": trained["pipeline_version"],
        "environment": {
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "libraries": library_versions(),
        },
        "data": {
            "filename": dataset.get("filename"),
            "dataset_hash": dataset.get("dataset_hash"),
            "rows": dataset.get("rows"),
            "columns": dataset.get("columns"),
            "target": prepared["target_col"],
            "target_auto_detected": prepared["target_auto_detected"],
            "problem_type": problem_type,
            "class_names": trained["class_names"],
            "class_distribution": trained.get("imbalance", {}).get("distribution", []),
            "imbalance_ratio": trained.get("imbalance", {}).get("ratio"),
            "rows_dropped_missing_target": prepared.get("dropped_target_rows", 0),
            "features": {
                "numeric": prepared["numeric_columns"],
                "one_hot": prepared["categorical_columns"],
                "target_encoded": prepared.get("high_cardinality_columns", []),
                "dropped": prepared["dropped_columns"],
            },
        },
        "training": {
            "mode": dataset.get("mode"),
            "selected_model": selected,
            "candidates": sorted(set(trained["trained_models"]) - {"Ensemble"}),
            "failed_models": trained["failed_models"],
            "ensemble_members": trained.get("ensemble_members", []),
            "selection_metric": selection_metric,
            "holdout_fraction": dataset.get("holdout_fraction"),
            "holdout_rows": trained.get("holdout_rows"),
            "cv_folds": trained.get("cv_folds"),
            "class_weighting": trained.get("class_weighting", {}),
            "smote_used": trained.get("smote_used", False),
            "tuning_budget_seconds": trained.get("tuning_budget_seconds", 0.0),
            "tuning": trained.get("tuning", []),
        },
        "metrics": {
            "selected_model": selected,
            "cross_validated": trained["cv_scores"].get(selected, {}),
            "holdout": trained["performance_scores"].get(selected, {}),
            "baseline_cross_validated": trained.get("baseline_cv", {}),
            "all_models_cv": trained["cv_scores"],
            "all_models_holdout": trained["performance_scores"],
            "health": health,
        },
        "explanation": {
            "method": explanation_method,
            "top_features": feature_importance[:10],
        },
    }
