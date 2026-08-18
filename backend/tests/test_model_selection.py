"""Cross-validation, metric choice, imbalance handling, tuning, ensembling.

Phase 4's exit criterion has two halves, and this file exists to hold both:
reported scores are cross-validated means with variance, and an imbalanced
dataset produces a warning and a sensible model rather than a high-accuracy
constant predictor.
"""

import numpy as np
import pandas as pd
import pytest

from config import settings
from ml.evaluate import evaluate_classification, evaluate_regression, per_class_report
from ml.imbalance import can_use_smote, describe_imbalance, sample_weight_for
from ml.preprocess import prepare_dataset
from ml.train import (
    HOLDOUT_SIZE,
    _prune_registry,
    build_cv,
    get_model_registry,
    select_best,
    train_models,
)
from ml.tuning import Budget, iterations_for, space_for
from tests.conftest import upload_and_wait


def _frame(n=400, imbalance=None, seed=0):
    """A learnable binary problem; `imbalance` is the share of the majority class."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n) if imbalance is None else (rng.random(n) > imbalance).astype(int)
    # A signal the model can actually find, plus noise.
    signal = y * 2.0 + rng.normal(0, 0.8, n)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0, 1, n),
            "group": rng.choice(["a", "b", "c"], n),
            "label": np.where(y == 1, "yes", "no"),
        }
    )


def _train(df, **kwargs):
    prepared = prepare_dataset(df, target_column="label")
    return prepared, train_models(prepared["X"], prepared["y"], preprocessor=prepared["preprocessor"], **kwargs)


# --- cross-validation -------------------------------------------------------


def test_scores_are_cross_validated_means_with_variance():
    """The exit criterion, stated directly."""
    _, trained = _train(_frame(), mode="manual", manual_model="RandomForest")
    entry = trained["cv_scores"]["RandomForest"]["f1_macro"]

    assert set(entry) == {"mean", "std", "folds"}
    assert len(entry["folds"]) == trained["cv_folds"] > 1
    assert entry["std"] >= 0.0
    assert entry["mean"] == pytest.approx(float(np.mean(entry["folds"])))
    assert entry["std"] == pytest.approx(float(np.std(entry["folds"])))


def test_the_holdout_is_never_part_of_cross_validation():
    """Selection, folds and tuning all run on the dev split only."""
    df = _frame(n=500)
    _, trained = _train(df, mode="manual", manual_model="DecisionTree")

    assert trained["holdout_rows"] == pytest.approx(len(df) * HOLDOUT_SIZE, abs=2)
    assert len(trained["X_train"]) + len(trained["X_test"]) == len(df)
    # No row appears on both sides.
    assert set(trained["X_train"].index).isdisjoint(set(trained["X_test"].index))


def test_folds_are_stratified_so_a_rare_class_survives_every_fold():
    y = np.array([0] * 90 + [1] * 10)
    cv = build_cv("classification", y)
    for _, validation in cv.split(np.zeros((len(y), 1)), y):
        assert len(np.unique(y[validation])) == 2


def test_fold_count_shrinks_to_the_rarest_class():
    """Five folds are impossible when a class has three members."""
    y = np.array([0] * 90 + [1] * 3)
    assert build_cv("classification", y).get_n_splits() == 3


def test_holdout_and_cv_scores_are_reported_separately():
    _, trained = _train(_frame(), mode="manual", manual_model="RandomForest")
    assert "RandomForest" in trained["cv_scores"]
    assert "RandomForest" in trained["performance_scores"]
    # Different data, so different numbers — the holdout is not a copy of CV.
    assert "f1_macro" in trained["performance_scores"]["RandomForest"]


# --- metrics ----------------------------------------------------------------


def test_classification_reports_the_imbalance_aware_metrics():
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.zeros(10, dtype=int)  # the constant predictor
    metrics = evaluate_classification(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.8)
    # The whole point: the same prediction scores badly on the honest metrics.
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["f1_macro"] < 0.5


def test_roc_auc_is_reported_when_probabilities_are_available():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert evaluate_classification(y_true, y_true, y_proba=proba)["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_absent_rather_than_faked_without_probabilities():
    y_true = np.array([0, 0, 1, 1])
    assert "roc_auc" not in evaluate_classification(y_true, y_true)


def test_regression_reports_mae_r2_and_mape():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 32.0, 39.0])
    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(1.25)
    assert 0.9 < metrics["r2"] <= 1.0
    assert metrics["mape"] == pytest.approx(6.0416666, rel=1e-3)


def test_mape_is_withheld_when_the_actuals_are_mostly_zero():
    """Dividing by zero would produce a number worse than no number."""
    y_true = np.array([0.0, 0.0, 0.0, 5.0])
    assert "mape" not in evaluate_regression(y_true, np.array([1.0, 1.0, 1.0, 5.0]))


def test_per_class_report_exposes_the_class_the_model_fails_on():
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.zeros(10, dtype=int)
    rows = per_class_report(y_true, y_pred, ["no", "yes"])

    assert [row["label"] for row in rows] == ["no", "yes"]
    minority = next(row for row in rows if row["label"] == "yes")
    assert minority["recall"] == 0.0
    assert minority["support"] == 2


# --- selection --------------------------------------------------------------


def test_selection_prefers_the_higher_cross_validated_mean():
    cv = {"A": {"f1_macro": {"mean": 0.7, "std": 0.01}}, "B": {"f1_macro": {"mean": 0.8, "std": 0.01}}}
    assert select_best(cv, "f1_macro") == "B"


def test_ties_are_broken_by_the_steadier_model():
    """Equal means, so the one that does not swing with the split wins."""
    cv = {"Swingy": {"f1_macro": {"mean": 0.8, "std": 0.20}}, "Steady": {"f1_macro": {"mean": 0.8, "std": 0.02}}}
    assert select_best(cv, "f1_macro") == "Steady"


def test_regression_selection_prefers_the_lower_error():
    cv = {"A": {"rmse": {"mean": 5.0, "std": 0.1}}, "B": {"rmse": {"mean": 2.0, "std": 0.1}}}
    assert select_best(cv, "rmse") == "B"


def test_selection_metric_is_macro_f1_not_accuracy():
    _, trained = _train(_frame(imbalance=0.9), mode="auto")
    assert trained["selection_metric"] == "f1_macro"


# --- imbalance --------------------------------------------------------------


def test_imbalance_is_detected_and_described():
    y = np.array([0] * 95 + [1] * 5)
    report = describe_imbalance(y, ["no", "yes"])

    assert report["imbalanced"] and report["severe"]
    assert report["ratio"] == pytest.approx(19.0)
    assert report["distribution"][0] == {"label": "no", "count": 95, "share": pytest.approx(0.95)}
    assert any("Class imbalance" in w for w in report["warnings"])


def test_a_balanced_target_raises_no_warning():
    report = describe_imbalance(np.array([0] * 50 + [1] * 50), ["no", "yes"])
    assert not report["imbalanced"]
    assert report["warnings"] == []


def test_sample_weights_are_inverse_to_class_frequency():
    weights = sample_weight_for(np.array([0] * 90 + [1] * 10))
    assert weights[0] < weights[-1]
    assert weights.sum() == pytest.approx(100.0)


def test_class_weights_are_applied_when_the_target_is_skewed():
    _, trained = _train(_frame(imbalance=0.9), mode="manual", manual_model="RandomForest")
    assert trained["imbalance"]["imbalanced"]
    assert trained["class_weighting"].get("RandomForest") == "class_weight=balanced"


def test_smote_is_refused_with_a_reason_when_the_minority_is_tiny():
    ok, reason = can_use_smote(np.array([0] * 60 + [1] * 3))
    assert not ok
    assert "at least" in reason


def test_smote_resamples_only_inside_the_pipeline():
    """SMOTE must not change the number of rows the caller sees scored."""
    df = _frame(n=400, imbalance=0.85)
    _, trained = _train(df, mode="manual", manual_model="DecisionTree", use_smote=True)

    assert trained["smote_used"]
    # The holdout is untouched by resampling — synthetic rows are never scored.
    assert trained["holdout_rows"] == len(trained["y_test"])
    assert set(np.unique(trained["y_test"])) <= {0, 1}


# --- the exit criterion, end to end ----------------------------------------


def test_an_imbalanced_dataset_warns_and_does_not_ship_a_constant_predictor():
    """Phase 4's exit criterion, run through the whole trainer."""
    _, trained = _train(_frame(n=500, imbalance=0.92), mode="auto")

    assert trained["imbalance"]["imbalanced"]
    assert any("Class imbalance" in w for w in trained["imbalance"]["warnings"])

    selected = trained["selected_model"]
    holdout = trained["performance_scores"][selected]
    # A constant predictor scores ~0.92 accuracy and ~0.48 macro F1 here.
    assert holdout["balanced_accuracy"] > 0.6, "selected a model no better than predicting the majority class"
    assert holdout["f1_macro"] > 0.5


def test_a_baseline_is_cross_validated_on_the_same_folds():
    _, trained = _train(_frame(), mode="auto")
    assert "f1_macro" in trained["baseline_cv"]
    assert trained["baseline_cv"]["f1_macro"]["mean"] >= 0.0


# --- ensembling -------------------------------------------------------------


def test_ensemble_members_are_weighted_and_near_chance_models_excluded():
    _, trained = _train(_frame(n=500), mode="ensemble")

    assert trained["selected_model"] == "Ensemble"
    assert len(trained["ensemble_members"]) >= 2
    baseline = trained["baseline_cv"]["f1_macro"]["mean"]
    for member in trained["ensemble_members"]:
        assert trained["cv_scores"][member]["f1_macro"]["mean"] > baseline


def test_the_ensemble_is_scored_on_its_own_cross_validated_predictions():
    _, trained = _train(_frame(n=500), mode="ensemble")
    assert "Ensemble" in trained["cv_scores"]
    assert "Ensemble" in trained["performance_scores"]
    assert len(trained["cv_scores"]["Ensemble"]["f1_macro"]["folds"]) == trained["cv_folds"]


def test_attribution_never_points_at_the_ensemble():
    """A vote has no preprocessor+model pipeline to explain."""
    _, trained = _train(_frame(n=500), mode="ensemble")
    assert trained["explain_model"] != "Ensemble"
    assert trained["explain_model"] in trained["trained_models"]


# --- tuning -----------------------------------------------------------------


def test_a_zero_budget_means_no_search_happened():
    _, trained = _train(_frame(), mode="manual", manual_model="DecisionTree", tuning_budget_seconds=0)
    assert trained["tuning"] == []


def test_a_budget_produces_a_reported_search():
    _, trained = _train(_frame(), mode="manual", manual_model="DecisionTree", tuning_budget_seconds=30)
    assert len(trained["tuning"]) == 1
    report = trained["tuning"][0]
    assert report["model"] == "DecisionTree"
    assert report["candidates"] >= 1
    assert set(report["best_params"]) <= {"max_depth", "min_samples_leaf"}


def test_an_exhausted_budget_stops_further_searching():
    budget = Budget(0)
    assert budget.exhausted()


def test_iteration_count_never_exceeds_the_space():
    space = {"model__max_depth": [1, 2, 3]}
    assert iterations_for(space, budget_seconds=3600) <= 3


def test_every_registry_model_either_has_a_space_or_is_deliberately_skipped():
    for problem_type in ("classification", "regression"):
        for name in get_model_registry(problem_type):
            if name in {"LinearRegression"}:
                continue  # nothing to tune
            assert space_for(name) is not None, f"{name} has no search space"


# --- registry ---------------------------------------------------------------


def test_registry_is_broader_than_the_original_four():
    classification = get_model_registry("classification")
    regression = get_model_registry("regression")

    assert {"GradientBoosting", "ExtraTrees", "HistGradientBoosting"} <= set(classification)
    assert {"Ridge", "Lasso", "ElasticNet"} <= set(regression)


def test_histogram_boosting_is_dropped_on_small_data():
    """Its binning cost never pays for itself on a few hundred rows."""
    registry = get_model_registry("classification")
    pruned = _prune_registry(registry, n_rows=300, mode="auto")
    assert "HistGradientBoosting" not in pruned
    assert "RandomForest" in pruned


def test_superlinear_models_are_dropped_on_large_data():
    pruned = _prune_registry(get_model_registry("classification"), n_rows=50_000, mode="auto")
    assert "SVM" not in pruned
    assert "GradientBoosting" not in pruned


def test_manual_mode_never_second_guesses_an_explicit_choice():
    registry = get_model_registry("classification")
    assert _prune_registry(registry, n_rows=50_000, mode="manual") == registry


def test_the_candidate_cap_is_honoured():
    pruned = _prune_registry(get_model_registry("classification"), n_rows=2000, mode="auto", max_models=3)
    assert len(pruned) == 3


def test_the_full_registry_trains_end_to_end(monkeypatch):
    """The suite caps candidates for speed; this one pays for real breadth."""
    monkeypatch.setattr(settings, "max_candidate_models", 0)
    _, trained = _train(_frame(n=400), mode="auto")
    # Small data drops HistGradientBoosting; everything else should survive.
    assert len(trained["trained_models"]) >= 6
    assert not trained["failed_models"], trained["failed_models"]


# --- the API surface --------------------------------------------------------


def test_the_result_payload_carries_cv_scores_and_a_model_card(client, fixture_csv):
    result = upload_and_wait(client, fixture_csv("clean_classification.csv"))["result"]

    selected = result["selected_model"]
    assert result["selection_metric"] == "f1_macro"
    assert result["cv_folds"] >= 2
    assert result["holdout_rows"] > 0
    assert result["cv_scores"][selected]["f1_macro"]["std"] >= 0
    assert result["baseline_cv"]["f1_macro"]["mean"] >= 0
    assert result["per_class"], "per-class breakdown missing"

    card = result["model_card"]
    assert card["data"]["target"] == result["target"]
    assert card["training"]["selection_metric"] == "f1_macro"
    assert card["environment"]["libraries"]["scikit-learn"]
    assert card["metrics"]["cross_validated"]


def test_regression_runs_report_mae_and_r2_through_the_api(client, fixture_csv):
    result = upload_and_wait(client, fixture_csv("clean_regression.csv"))["result"]
    holdout = result["all_model_scores"][result["selected_model"]]

    assert result["selection_metric"] == "rmse"
    assert "mae" in holdout and "r2" in holdout


def test_asking_for_a_tuning_budget_retrains_rather_than_replaying(client, fixture_csv):
    """A different configuration is a different run, as with mode and target."""
    first = upload_and_wait(client, fixture_csv("clean_classification.csv"))["result"]
    second = upload_and_wait(client, fixture_csv("clean_classification.csv"), tuning_budget_seconds="20")["result"]

    assert first["run_key"] != second["run_key"]
    assert second["status"] == "trained"
    assert second["tuning_budget_seconds"] == 20.0


def test_asking_for_smote_retrains_rather_than_replaying(client, fixture_csv):
    first = upload_and_wait(client, fixture_csv("clean_classification.csv"))["result"]
    second = upload_and_wait(client, fixture_csv("clean_classification.csv"), use_smote="true")["result"]
    assert first["run_key"] != second["run_key"]


def test_the_tuning_budget_is_capped_before_any_work_is_queued(client, fixture_csv):
    """An unbounded search would hold a web worker indefinitely."""
    from ml.tuning import MAX_BUDGET_SECONDS

    result = upload_and_wait(client, fixture_csv("clean_classification.csv"), tuning_budget_seconds="99999")["result"]
    assert result["tuning_budget_seconds"] == MAX_BUDGET_SECONDS
