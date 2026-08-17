"""Health verdicts, settings parsing, and structured logging."""

import json
import logging

import numpy as np
import pytest

from config import Settings
from logging_config import ContextFilter, JsonFormatter, job_id_var, request_id_var
from ml.health import assess_run

# --- health verdict ---------------------------------------------------------


def test_majority_class_accuracy_is_not_credited():
    """95% on a 95/5 split is exactly the constant predictor."""
    y_test = np.array([0] * 95 + [1] * 5)
    verdict = assess_run("classification", {"accuracy": 0.95}, y_test, {}, 2)
    assert verdict["verdict"] == "unreliable"
    assert verdict["baseline"] == pytest.approx(0.95)


def test_exhibit_a_scores_as_unreliable():
    """The committed 1.7%-accuracy run must never read as 'Optimized Model'."""
    y_test = np.arange(200) % 100
    verdict = assess_run("classification", {"accuracy": 0.017}, y_test, {}, 14555)
    assert verdict["verdict"] == "unreliable"
    assert any("classes" in reason for reason in verdict["reasons"])


def test_genuinely_good_run_is_strong():
    y_test = np.array([0] * 50 + [1] * 50)
    assert assess_run("classification", {"accuracy": 0.93}, y_test, {}, 2)["verdict"] == "strong"


def test_failed_models_downgrade_to_fair():
    y_test = np.array([0] * 50 + [1] * 50)
    verdict = assess_run("classification", {"accuracy": 0.93}, y_test, {"XGBoost": "boom"}, 2)
    assert verdict["verdict"] == "fair"


def test_regression_is_compared_to_the_mean_predictor():
    y_test = np.array([10.0, 20.0, 30.0, 40.0])
    baseline_rmse = float(np.sqrt(np.mean((y_test - y_test.mean()) ** 2)))
    assert assess_run("regression", {"rmse": baseline_rmse * 0.1}, y_test, {}, 0)["verdict"] == "strong"
    assert assess_run("regression", {"rmse": baseline_rmse * 1.5}, y_test, {}, 0)["verdict"] == "unreliable"


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
