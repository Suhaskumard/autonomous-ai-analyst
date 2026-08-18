"""Model training.

Four rules this module exists to enforce:

1. Nothing is fitted before the split. Each candidate is one Pipeline of
   (preprocessor -> [SMOTE] -> estimator), so imputer/scaler/encoder statistics
   never see rows they are then scored on. The fitted Pipeline is also the
   serving artifact, so predict.py loads one object instead of re-implementing
   the transform order.
2. A score is a distribution, not a number. Every candidate is cross-validated
   on the development split and reported as mean +/- std, so a lucky split
   cannot win. On top of that sits a held-out test set that no search, no fold,
   and no selection decision ever touches — the only truly clean number.
3. Selection is on macro F1, not accuracy. On a 95/5 target a constant
   predictor scores 95% accuracy and 0.49 macro F1; selecting on accuracy
   picks it, which is exactly the failure this phase exists to remove. Every
   candidate is also compared against a Dummy baseline fitted on the same
   folds, and one that cannot beat it is never put in an ensemble.
4. Classification targets are label-encoded. XGBoost requires 0..n-1 integer
   classes and fails outright on string labels; the encoder is stored with the
   model and inverted at predict time.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import settings
from exceptions import TrainingError
from ml.evaluate import HIGHER_IS_BETTER, SELECTION_METRIC, evaluate
from ml.imbalance import apply_class_weight, build_pipeline, can_use_smote, describe_imbalance, supports_class_weight
from ml.tuning import Budget, tune_pipeline

logger = logging.getLogger(__name__)

try:  # xgboost is optional; a missing wheel should not take the whole app down
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost is not installed; XGBoost models will be unavailable.")

try:  # lightgbm is optional in the same way
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    LIGHTGBM_AVAILABLE = False
    logger.info("lightgbm is not installed; LightGBM models will be unavailable.")

try:  # catboost is heavy, so it is supported-if-present rather than required
    from catboost import CatBoostClassifier, CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    CATBOOST_AVAILABLE = False

# Share of the data withheld from *everything* — CV, tuning, selection.
HOLDOUT_SIZE = 0.2
RANDOM_STATE = 42

CV_FOLDS = 5
CV_FOLDS_LARGE = 3
LARGE_DATASET_ROWS = 20_000
# Above this, the slow superlinear learners are dropped in auto mode.
SLOW_MODEL_ROW_LIMIT = 5_000
# Below this, histogram-based boosting is all fixed cost and no benefit.
SMALL_DATASET_ROWS = 1_000

# Bumped whenever preprocessing or training changes in a way that makes stored
# artifacts stale. It is part of the cache key, so old artifacts are ignored
# automatically instead of being silently reused.
PIPELINE_VERSION = "3"

# CV scorers are sign-flipped by sklearn convention; these come back negated.
_NEGATED_SCORERS = {"rmse", "mae", "mape"}


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
    """Every candidate for this problem type, at default settings.

    Broader than the original four: gradient boosting and extra trees cover
    ground random forests miss, and ridge/lasso/elastic-net give regression a
    regularised linear option instead of only ordinary least squares.
    """
    if problem_type == "classification":
        registry = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
            "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
            "SVM": SVC(probability=True, kernel="rbf", max_iter=2000),
        }
        if XGBOOST_AVAILABLE:
            registry["XGBoost"] = XGBClassifier(eval_metric="logloss", n_estimators=100, random_state=RANDOM_STATE)
        if LIGHTGBM_AVAILABLE:
            registry["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
        if CATBOOST_AVAILABLE:
            registry["CatBoost"] = CatBoostClassifier(verbose=0, random_seed=RANDOM_STATE)
        return registry

    registry = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "SVR": SVR(max_iter=2000),
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoostRegressor"] = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
    if LIGHTGBM_AVAILABLE:
        registry["LightGBMRegressor"] = LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    if CATBOOST_AVAILABLE:
        registry["CatBoostRegressor"] = CatBoostRegressor(verbose=0, random_seed=RANDOM_STATE)
    return registry


def _prune_registry(registry: dict, n_rows: int, mode: str, max_models: int = 0) -> dict:
    """Keep the candidates that suit the data's size, then honour the cap.

    Cross-validation multiplies every fit by the fold count, so breadth costs
    real time and a candidate has to earn its place at both ends of the range:

    * histogram-based boosting is built for tens of thousands of rows and pays
      a fixed binning cost that never comes back on a few hundred — on small
      data it is both slower than exact boosting and no more accurate;
    * kernel SVMs and exact gradient boosting are superlinear, so past a few
      thousand rows they dominate the run for no reliable gain.

    Manual mode is left alone: an explicit choice is not second-guessed.
    """
    if mode == "manual":
        return registry

    pruned = dict(registry)
    if n_rows < SMALL_DATASET_ROWS:
        for name in ("HistGradientBoosting", "HistGradientBoostingRegressor"):
            pruned.pop(name, None)
    if n_rows > SLOW_MODEL_ROW_LIMIT:
        for name in ("SVM", "SVR", "GradientBoosting", "GradientBoostingRegressor"):
            pruned.pop(name, None)

    if max_models and len(pruned) > max_models:
        # Registry order is the priority order: the general-purpose learners
        # come first, the specialists after.
        pruned = dict(list(pruned.items())[:max_models])
    return pruned


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


def build_cv(problem_type: str, y_dev: np.ndarray):
    """Folds sized to the data, stratified whenever the classes allow it.

    Stratification is what keeps a rare class from vanishing out of a fold
    entirely, which would make that fold's macro F1 meaningless.
    """
    folds = CV_FOLDS_LARGE if len(y_dev) > LARGE_DATASET_ROWS else CV_FOLDS
    if problem_type == "classification":
        _, counts = np.unique(y_dev, return_counts=True)
        smallest = int(counts.min())
        if smallest >= 2:
            return StratifiedKFold(n_splits=max(2, min(folds, smallest)), shuffle=True, random_state=RANDOM_STATE)
        logger.info("A class has a single member; falling back to unstratified folds.")
    return KFold(n_splits=max(2, min(folds, len(y_dev))), shuffle=True, random_state=RANDOM_STATE)


def scoring_for(problem_type: str, n_classes: int) -> dict[str, str]:
    """The scorer set cross_validate reports, keyed by the name we publish."""
    if problem_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
        }
        scoring["roc_auc"] = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
        return scoring
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "mape": "neg_mean_absolute_percentage_error",
    }


def summarise_cv(cv_results: dict, scoring: dict) -> dict:
    """Fold scores as {metric: {mean, std, folds}}.

    A fold that errored comes back as NaN and is dropped rather than dragging
    the mean to nothing — but if every fold failed the metric is simply absent.
    """
    summary: dict[str, dict] = {}
    for metric in scoring:
        raw = cv_results.get(f"test_{metric}")
        if raw is None:
            continue
        values = np.asarray(raw, dtype=float)
        if metric in _NEGATED_SCORERS:
            values = -values
        if metric == "mape":
            values = values * 100.0  # sklearn reports a fraction; we publish a percentage
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        summary[metric] = {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "folds": [float(v) for v in finite],
        }
    return summary


def _baseline_cv(problem_type: str, X_dev, y_dev, cv, scoring: dict) -> dict:
    """What a constant predictor scores on the very same folds.

    This is the number every candidate has to beat to be believed, and the
    reason a majority-class predictor can no longer be selected or ensembled.
    """
    dummy = (
        DummyClassifier(strategy="most_frequent")
        if problem_type == "classification"
        else DummyRegressor(strategy="mean")
    )
    try:
        results = cross_validate(dummy, X_dev, y_dev, cv=cv, scoring=scoring, error_score=np.nan)
        return summarise_cv(results, scoring)
    except Exception as exc:
        logger.warning("Could not cross-validate the baseline: %s", exc)
        return {}


def _improvement_over(score: float | None, baseline: float | None, metric: str) -> float | None:
    """Signed distance from the baseline, in the direction that means better."""
    if score is None or baseline is None:
        return None
    return score - baseline if HIGHER_IS_BETTER[metric] else baseline - score


def _apply_imbalance_handling(name: str, estimator, imbalance: dict, n_classes: int):
    """Weight the loss toward the minority class where the estimator allows it."""
    if not imbalance.get("imbalanced"):
        return estimator, None
    if supports_class_weight(estimator):
        return apply_class_weight(estimator), "class_weight=balanced"
    # XGBoost has no class_weight; for two classes scale_pos_weight is the
    # equivalent knob. Multiclass XGBoost relies on SMOTE or on macro-F1
    # selection instead, which is a real limitation rather than a silent one.
    if name.startswith("XGBoost") and n_classes == 2 and "scale_pos_weight" in estimator.get_params():
        ratio = imbalance.get("ratio", 1.0)
        estimator.set_params(scale_pos_weight=float(ratio))
        return estimator, f"scale_pos_weight={ratio:.2f}"
    return estimator, None


def _supports_proba(pipeline) -> bool:
    try:
        return hasattr(pipeline.named_steps["model"], "predict_proba")
    except (AttributeError, KeyError):
        return hasattr(pipeline, "predict_proba")


def build_ensemble(
    members: dict,
    cv_scores: dict,
    baseline_cv: dict,
    problem_type: str,
    selection_metric: str,
):
    """A vote weighted by cross-validated performance, over members worth voting.

    The old ensemble was an unweighted mode/mean over every model that trained,
    so a near-chance model got the same say as the best one. Anything that
    fails to beat the Dummy baseline on the development folds is excluded
    outright, and the rest are weighted by how far past it they got.
    """
    eligible: list[tuple[str, object, float]] = []
    baseline = (baseline_cv.get(selection_metric) or {}).get("mean")
    for name, pipeline in members.items():
        score = (cv_scores.get(name, {}).get(selection_metric) or {}).get("mean")
        improvement = _improvement_over(score, baseline, selection_metric)
        if improvement is None or improvement <= 0:
            logger.info("Excluding %s from the ensemble: it does not beat the baseline.", name)
            continue
        eligible.append((name, pipeline, improvement))

    if len(eligible) < 2:
        return None, [], "Fewer than two models beat the naive baseline, so there was nothing to vote between."

    weights = [improvement for _, _, improvement in eligible]
    estimators = [(name, clone(pipeline)) for name, pipeline, _ in eligible]
    names = [name for name, _, _ in eligible]

    if problem_type == "classification":
        voting = "soft" if all(_supports_proba(pipeline) for _, pipeline, _ in eligible) else "hard"
        ensemble = VotingClassifier(estimators=estimators, voting=voting, weights=weights)
    else:
        ensemble = VotingRegressor(estimators=estimators, weights=weights)
    return ensemble, names, ""


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    mode: str = "auto",
    manual_model: str | None = None,
    tuning_budget_seconds: float = 0.0,
    use_smote: bool = False,
) -> dict:
    """Cross-validate every candidate on the dev split, then score on a holdout."""
    problem_type = detect_problem_type(y)
    y_encoded, label_encoder = _encode_target(y, problem_type)
    class_names = [str(c) for c in label_encoder.classes_] if label_encoder is not None else []
    n_classes = len(class_names)

    imbalance = (
        describe_imbalance(y_encoded, class_names)
        if problem_type == "classification"
        else {"imbalanced": False, "severe": False, "warnings": [], "distribution": []}
    )

    # The holdout is carved out first and touched exactly once, at the end.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y_encoded,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=_stratify_for(y_encoded, problem_type),
    )

    cv = build_cv(problem_type, y_dev)
    scoring = scoring_for(problem_type, n_classes)
    selection_metric = SELECTION_METRIC[problem_type]
    baseline_cv = _baseline_cv(problem_type, X_dev, y_dev, cv, scoring)

    registry = _prune_registry(get_model_registry(problem_type), len(X), mode, max_models=settings.max_candidate_models)
    if mode == "manual" and manual_model:
        if manual_model not in registry:
            available = ", ".join(get_model_registry(problem_type))
            raise TrainingError(f"Model '{manual_model}' is not available for {problem_type}. Available: {available}.")
        registry = {manual_model: registry[manual_model]}

    smote_ok, smote_reason = (False, "")
    if use_smote and problem_type == "classification":
        smote_ok, smote_reason = can_use_smote(y_dev)

    budget = Budget(tuning_budget_seconds)
    trained_models: dict[str, object] = {}
    cv_scores: dict[str, dict] = {}
    holdout_scores: dict[str, dict] = {}
    failed_models: dict[str, str] = {}
    tuning_reports: list[dict] = []
    weighting_notes: dict[str, str] = {}

    for model_name, estimator in registry.items():
        try:
            estimator, note = _apply_imbalance_handling(model_name, clone(estimator), imbalance, n_classes)
            if note:
                weighting_notes[model_name] = note
            pipeline = build_pipeline(clone(preprocessor), estimator, y_dev, use_smote=smote_ok)

            if tuning_budget_seconds > 0:
                pipeline, report = tune_pipeline(
                    pipeline, model_name, X_dev, y_dev, cv, _sklearn_scorer(selection_metric), budget
                )
                if report:
                    tuning_reports.append(report)

            # Cross-validate the configuration we are actually going to ship,
            # so the reported variance describes the final model.
            results = cross_validate(pipeline, X_dev, y_dev, cv=cv, scoring=scoring, error_score=np.nan)
            summary = summarise_cv(results, scoring)
            if selection_metric not in summary:
                raise TrainingError(f"every cross-validation fold failed for {model_name}")

            fitted = clone(pipeline).fit(X_dev, y_dev)
            holdout = _score_on_holdout(fitted, X_test, y_test, problem_type, n_classes)

            trained_models[model_name] = fitted
            cv_scores[model_name] = summary
            holdout_scores[model_name] = holdout
        except Exception as exc:
            logger.warning("Model %s failed to train: %s", model_name, exc, exc_info=True)
            failed_models[model_name] = str(exc)

    if not trained_models:
        detail = "; ".join(f"{name}: {err}" for name, err in failed_models.items())
        raise TrainingError(f"Every candidate model failed to train. Details — {detail}")

    best_model_name = select_best(cv_scores, selection_metric)
    selected_model_name = best_model_name
    ensemble_note = ""
    ensemble_members: list[str] = []

    if mode == "ensemble":
        ensemble, ensemble_members, ensemble_note = build_ensemble(
            trained_models, cv_scores, baseline_cv, problem_type, selection_metric
        )
        if ensemble is not None:
            try:
                results = cross_validate(ensemble, X_dev, y_dev, cv=cv, scoring=scoring, error_score=np.nan)
                fitted_ensemble = clone(ensemble).fit(X_dev, y_dev)
                cv_scores["Ensemble"] = summarise_cv(results, scoring)
                holdout_scores["Ensemble"] = _score_on_holdout(fitted_ensemble, X_test, y_test, problem_type, n_classes)
                trained_models["Ensemble"] = fitted_ensemble
                selected_model_name = "Ensemble"
            except Exception as exc:
                logger.warning("Ensemble failed to fit: %s", exc, exc_info=True)
                failed_models["Ensemble"] = str(exc)
                ensemble_note = f"The ensemble failed to fit ({exc}); the best single model was used instead."

    return {
        "problem_type": problem_type,
        "trained_models": trained_models,
        # Held-out scores keep the original key: this is what the UI has always
        # shown as "performance", and it is now the cleanest number available.
        "performance_scores": holdout_scores,
        "cv_scores": cv_scores,
        "baseline_cv": baseline_cv,
        "selection_metric": selection_metric,
        "selected_model": selected_model_name,
        # Never "Ensemble": attribution needs a preprocessor+model Pipeline.
        "explain_model": best_model_name,
        "ensemble_members": ensemble_members,
        "ensemble_note": ensemble_note,
        "failed_models": failed_models,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "imbalance": imbalance,
        "class_weighting": weighting_notes,
        "smote_used": bool(smote_ok),
        "smote_note": smote_reason,
        "tuning": tuning_reports,
        "tuning_budget_seconds": float(tuning_budget_seconds),
        "cv_folds": int(cv.get_n_splits()),
        "holdout_rows": int(len(y_test)),
        "X_train": X_dev,
        "y_train": y_dev,
        "X_test": X_test,
        "y_test": y_test,
        "pipeline_version": PIPELINE_VERSION,
    }


def select_best(cv_scores: dict, selection_metric: str) -> str:
    """Best cross-validated mean, ties broken by the smaller spread.

    Selecting on the mean alone would let a model that got lucky on one fold
    beat a steadier one with the same average.
    """
    higher_is_better = HIGHER_IS_BETTER[selection_metric]

    def key(name: str):
        entry = cv_scores[name].get(selection_metric) or {}
        mean = entry.get("mean")
        std = entry.get("std", 0.0)
        if mean is None:
            return (1, 0.0, 0.0)
        return (0, -mean if higher_is_better else mean, std)

    return sorted(cv_scores, key=key)[0]


def _sklearn_scorer(selection_metric: str) -> str:
    return {"f1_macro": "f1_macro", "rmse": "neg_root_mean_squared_error"}[selection_metric]


def _score_on_holdout(fitted, X_test, y_test, problem_type: str, n_classes: int) -> dict:
    """The one evaluation on data no fold, search, or selection has seen."""
    y_pred = fitted.predict(X_test)
    y_proba = None
    if problem_type == "classification":
        try:
            y_proba = np.asarray(fitted.predict_proba(X_test))
        except Exception:  # hard-voting ensembles and SVMs without probability
            y_proba = None
    labels = list(range(n_classes)) if n_classes else None
    return evaluate(problem_type, y_test, y_pred, y_proba=y_proba, labels=labels)
