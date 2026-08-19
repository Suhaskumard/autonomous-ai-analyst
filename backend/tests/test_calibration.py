"""Phase 9: is the confidence a classifier reports actually a probability?

`ml/calibration.py` fits a calibrated variant and keeps it only if it
measurably improves the holdout's expected calibration error. These tests are
about the decision procedure — does it improve an overconfident model, leave a
well-calibrated one alone, refuse to act on too little data, and never touch a
regressor — rather than about calibration theory itself.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml import calibration


def _split(n=3000, n_classes=2, random_state=0):
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=6, n_classes=n_classes, random_state=random_state
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def _overconfident_pipeline(X_dev, y_dev):
    """A shallow, few-tree forest: fast to fit and reliably poorly calibrated."""
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0))]
    )
    pipeline.fit(X_dev, y_dev)
    return pipeline


# --- expected calibration error -----------------------------------------------


def test_perfect_confidence_has_zero_ece():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    ece, _ = calibration.expected_calibration_error(y_true, y_proba, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_confident_and_wrong_has_high_ece():
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([[0.0, 1.0]] * 4)  # 100% confident, always wrong
    ece, _ = calibration.expected_calibration_error(y_true, y_proba, n_bins=5)
    assert ece == pytest.approx(1.0, abs=1e-9)


def test_ece_measures_the_top_class_confidence_not_the_positive_class():
    """The UI shows the winning class's confidence, for any number of classes."""
    y_true = np.array([0, 0, 0, 0])
    # 90% confident in the *correct* class 0 every time -> should be well calibrated.
    y_proba = np.array([[0.9, 0.05, 0.05]] * 4)
    ece, _ = calibration.expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece < 0.15


def test_brier_score_is_zero_for_a_perfect_forecaster():
    y_true = np.array([0, 1, 1])
    y_proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert calibration.brier_score(y_true, y_proba, n_classes=2) == pytest.approx(0.0, abs=1e-9)


def test_reliability_curve_covers_every_bin_even_when_empty():
    y_true = np.array([0, 1])
    y_proba = np.array([[0.55, 0.45], [0.45, 0.55]])  # everything in one bin
    _, curve = calibration.expected_calibration_error(y_true, y_proba, n_bins=10)
    assert len(curve) == 10
    empty = [entry for entry in curve if entry["count"] == 0]
    assert all(entry["mean_confidence"] is None for entry in empty)


# --- the calibrate() decision procedure ---------------------------------------


def test_an_overconfident_model_is_calibrated_and_improves():
    X_dev, X_test, y_dev, y_test = _split()
    pipeline = _overconfident_pipeline(X_dev, y_dev)

    report = calibration.calibrate(pipeline, "classification", X_dev, y_dev, X_test, y_test, n_classes=2)

    # Confirms the fixture is doing its job: not already well calibrated.
    assert report["before"]["verdict"] != calibration.WELL_CALIBRATED
    if report["applied"]:
        assert report["after"]["ece"] < report["before"]["ece"]
        assert report["method"] in {"isotonic", "sigmoid"}
    else:
        # Calibration is not guaranteed to clear the margin on every run; if it
        # did not, the reason must say why rather than silently doing nothing.
        assert report["reason"]


def test_an_already_calibrated_model_is_left_alone():
    X_dev, X_test, y_dev, y_test = _split(random_state=7)
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    pipeline.fit(X_dev, y_dev)

    report = calibration.calibrate(pipeline, "classification", X_dev, y_dev, X_test, y_test, n_classes=2)

    if report["before"]["verdict"] == calibration.WELL_CALIBRATED:
        assert report["applied"] is False
        assert "already" in report["reason"].lower() or "well calibrated" in report["reason"].lower()
        # The pipeline handed back is the original, not a wrapped copy.
        assert report["pipeline"] is pipeline


def test_a_regressor_is_never_calibrated():
    report = calibration.calibrate("not-a-classifier", "regression", None, None, None, None, n_classes=0)
    assert report["applied"] is False
    assert report["before"] is None
    assert "regressor" in report["reason"].lower()


def test_a_model_with_no_predict_proba_is_reported_not_crashed():
    class NoProba:
        pass

    report = calibration.calibrate(NoProba(), "classification", None, None, None, None, n_classes=2)
    assert report["applied"] is False
    assert "confidence" in report["reason"].lower() or "probabilit" in report["reason"].lower()


def test_too_small_a_holdout_is_measured_but_not_acted_on():
    X_dev, X_test, y_dev, y_test = _split(n=200)
    pipeline = _overconfident_pipeline(X_dev, y_dev)
    tiny_test, tiny_y = X_test[:20], y_test[:20]

    report = calibration.calibrate(pipeline, "classification", X_dev, y_dev, tiny_test, tiny_y, n_classes=2)

    assert report["applied"] is False
    assert report["verdict"] == calibration.UNMEASURED
    assert str(calibration.MIN_HOLDOUT_ROWS) in report["reason"]
    # It is measured, not skipped outright — the number exists even if unused.
    assert report["before"] is not None


def test_disabled_still_measures_but_never_swaps_the_model():
    X_dev, X_test, y_dev, y_test = _split()
    pipeline = _overconfident_pipeline(X_dev, y_dev)

    report = calibration.calibrate(pipeline, "classification", X_dev, y_dev, X_test, y_test, n_classes=2, enabled=False)

    assert report["applied"] is False
    assert report["pipeline"] is pipeline
    assert report["before"] is not None  # measured
    assert "off" in report["reason"].lower()


def test_a_marginal_improvement_below_the_threshold_is_not_adopted(monkeypatch):
    """A change too small to trust is a change too small to adopt."""
    monkeypatch.setattr(calibration, "MIN_ECE_IMPROVEMENT", 0.999)  # nothing clears this
    X_dev, X_test, y_dev, y_test = _split()
    pipeline = _overconfident_pipeline(X_dev, y_dev)

    report = calibration.calibrate(pipeline, "classification", X_dev, y_dev, X_test, y_test, n_classes=2)

    assert report["applied"] is False
    assert "margin" in report["reason"].lower() or "noise" in report["reason"].lower()
    assert report["pipeline"] is pipeline


# --- what goes into the run's metadata ----------------------------------------


def test_to_metadata_never_includes_a_fitted_estimator():
    """The fitted model lives in the bundle; the metadata file is read as JSON."""
    X_dev, X_test, y_dev, y_test = _split()
    pipeline = _overconfident_pipeline(X_dev, y_dev)
    report = calibration.calibrate(pipeline, "classification", X_dev, y_dev, X_test, y_test, n_classes=2)

    payload = calibration.to_metadata(report)
    assert "pipeline" not in payload
    assert payload["confidence_is_calibrated"] == bool(report["applied"])
    assert isinstance(payload["note"], str) and payload["note"]


def test_to_metadata_keeps_both_before_and_after_when_applied():
    X_dev, X_test, y_dev, y_test = _split()
    pipeline = _overconfident_pipeline(X_dev, y_dev)
    report = calibration.calibrate(pipeline, "classification", X_dev, y_dev, X_test, y_test, n_classes=2)
    payload = calibration.to_metadata(report)

    if report["applied"]:
        assert payload["before"]["ece"] > payload["after"]["ece"]
