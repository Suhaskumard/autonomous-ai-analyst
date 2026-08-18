"""Health verdicts, settings parsing, and structured logging."""

import json
import logging

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from config import Settings
from logging_config import ContextFilter, JsonFormatter, job_id_var, request_id_var
from ml.health import assess_run

# --- health verdict ---------------------------------------------------------


IMBALANCED = np.array([0] * 95 + [1] * 5)
BALANCED = np.array([0] * 50 + [1] * 50)


def _constant_predictor_scores(y_test):
    """What always-predict-the-majority-class actually scores on y_test."""
    majority = np.bincount(y_test).argmax()
    constant = np.full_like(y_test, majority)
    return {
        "accuracy": float(accuracy_score(y_test, constant)),
        "f1_macro": float(f1_score(y_test, constant, average="macro", zero_division=0)),
    }


def test_majority_class_predictor_is_not_credited():
    """95% accuracy on a 95/5 split is exactly what predicting nothing scores.

    The verdict is made on macro F1 precisely so this case cannot pass: the
    constant predictor's macro F1 *is* the baseline, so its lift is zero.
    """
    scores = _constant_predictor_scores(IMBALANCED)
    assert scores["accuracy"] == pytest.approx(0.95)

    verdict = assess_run("classification", scores, IMBALANCED, {}, 2)

    assert verdict["verdict"] == "unreliable"
    assert verdict["metric"] == "f1_macro"
    assert verdict["score"] == pytest.approx(verdict["baseline"])
    assert not verdict["beats_baseline"]


def test_high_accuracy_on_an_imbalanced_target_is_called_out():
    """The number that looks good has to be named as the trap it is."""
    verdict = assess_run("classification", _constant_predictor_scores(IMBALANCED), IMBALANCED, {}, 2)
    assert any("constant" in reason and "95.0%" in reason for reason in verdict["reasons"])


def test_exhibit_a_scores_as_unreliable():
    """The committed 1.7%-accuracy run must never read as 'Optimized Model'."""
    y_test = np.arange(200) % 100
    verdict = assess_run("classification", {"accuracy": 0.017, "f1_macro": 0.017}, y_test, {}, 14555)
    assert verdict["verdict"] == "unreliable"
    assert any("classes" in reason for reason in verdict["reasons"])


def test_genuinely_good_run_is_strong():
    verdict = assess_run("classification", {"accuracy": 0.93, "f1_macro": 0.93}, BALANCED, {}, 2)
    assert verdict["verdict"] == "strong"


def test_failed_models_downgrade_to_fair():
    verdict = assess_run("classification", {"accuracy": 0.93, "f1_macro": 0.93}, BALANCED, {"XGBoost": "boom"}, 2)
    assert verdict["verdict"] == "fair"


def test_cross_validated_spread_is_reported_next_to_the_holdout_score():
    verdict = assess_run(
        "classification",
        {"accuracy": 0.93, "f1_macro": 0.93},
        BALANCED,
        {},
        2,
        cv_summary={"f1_macro": {"mean": 0.91, "std": 0.02, "folds": [0.9, 0.92, 0.89, 0.93, 0.91]}},
    )
    assert verdict["cv_mean"] == pytest.approx(0.91)
    assert verdict["cv_std"] == pytest.approx(0.02)
    assert any("0.91" in reason and "5 folds" in reason for reason in verdict["reasons"])


def test_a_wide_cross_validated_spread_is_flagged_as_split_dependent():
    verdict = assess_run(
        "classification",
        {"accuracy": 0.93, "f1_macro": 0.93},
        BALANCED,
        {},
        2,
        cv_summary={"f1_macro": {"mean": 0.60, "std": 0.25, "folds": [0.3, 0.9, 0.5, 0.8, 0.5]}},
    )
    assert any("depends noticeably on which rows" in reason for reason in verdict["reasons"])


def test_imbalance_warnings_reach_the_verdict():
    verdict = assess_run(
        "classification",
        {"accuracy": 0.93, "f1_macro": 0.93},
        BALANCED,
        {},
        2,
        imbalance={"warnings": ["Class imbalance: 'no' is 95.0% of rows."]},
    )
    assert any("Class imbalance" in reason for reason in verdict["reasons"])


def test_regression_is_compared_to_the_mean_predictor():
    y_test = np.array([10.0, 20.0, 30.0, 40.0])
    baseline_rmse = float(np.sqrt(np.mean((y_test - y_test.mean()) ** 2)))
    assert assess_run("regression", {"rmse": baseline_rmse * 0.1}, y_test, {}, 0)["verdict"] == "strong"
    assert assess_run("regression", {"rmse": baseline_rmse * 1.5}, y_test, {}, 0)["verdict"] == "unreliable"


def test_negative_r2_is_called_out_as_worse_than_a_flat_line():
    y_test = np.array([10.0, 20.0, 30.0, 40.0])
    baseline_rmse = float(np.sqrt(np.mean((y_test - y_test.mean()) ** 2)))
    verdict = assess_run("regression", {"rmse": baseline_rmse * 0.1, "r2": -0.4}, y_test, {}, 0)
    assert any("negative" in reason for reason in verdict["reasons"])


# --- settings ---------------------------------------------------------------


def test_cors_origins_parse_from_a_comma_separated_string():
    settings = Settings(cors_allow_origins="http://a.test, http://b.test ,", _env_file=None)
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_upload_ceiling_converts_to_bytes():
    assert Settings(max_upload_mb=7, _env_file=None).max_upload_bytes == 7 * 1024 * 1024


def test_blank_api_key_reads_as_missing():
    assert Settings(gemini_api_key="   ", _env_file=None).gemini_key is None


def test_invalid_limits_are_rejected():
    with pytest.raises(Exception):
        Settings(max_upload_mb=0, _env_file=None)


def test_log_level_is_normalised():
    assert Settings(log_level="debug", _env_file=None).log_level == "DEBUG"


# --- structured logging -----------------------------------------------------


def test_log_lines_carry_request_and_job_ids():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "training started", (), None)
    request_token = request_id_var.set("req-1")
    job_token = job_id_var.set("job-9")
    try:
        ContextFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(request_token)
        job_id_var.reset(job_token)

    assert payload["request_id"] == "req-1"
    assert payload["job_id"] == "job-9"
    assert payload["message"] == "training started"
    assert payload["level"] == "INFO"


def test_extra_fields_survive_into_the_json_line():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "done", (), None)
    record.run_key = "abc123"
    ContextFilter().filter(record)
    assert json.loads(JsonFormatter().format(record))["run_key"] == "abc123"
