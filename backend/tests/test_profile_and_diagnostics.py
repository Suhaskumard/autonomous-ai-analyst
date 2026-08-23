"""The pre-training profile, the diagnostics payloads, and the workspace API."""

import numpy as np
import pandas as pd
import pytest

from ml.diagnostics import (
    MAX_CORRELATION_COLUMNS,
    build_confusion_matrix,
    build_correlation_matrix,
    build_residuals,
)
from ml.profile import profile_dataframe
from tests.conftest import upload_and_wait

# --- column profile ---------------------------------------------------------


def test_profile_reports_type_missing_and_cardinality(fixture_df):
    payload = profile_dataframe(fixture_df("clean_classification.csv"))
    by_name = {column["name"]: column for column in payload["column_profiles"]}

    assert payload["rows"] > 100
    assert payload["suggested_target"] == "churn"
    assert by_name["age"]["dtype"] == "numeric"
    assert by_name["region"]["dtype"] == "categorical"
    assert by_name["region"]["unique"] == 3
    assert by_name["age"]["missing"] == 0


def test_numeric_columns_get_a_histogram_sparkline(fixture_df):
    payload = profile_dataframe(fixture_df("clean_classification.csv"))
    age = next(c for c in payload["column_profiles"] if c["name"] == "age")
    assert age["sparkline"]["kind"] == "histogram"
    assert len(age["sparkline"]["counts"]) == 12
    assert sum(age["sparkline"]["counts"]) > 0
    assert {"min", "max", "mean", "median", "std"} <= set(age["stats"])


def test_categorical_columns_get_category_counts(fixture_df):
    payload = profile_dataframe(fixture_df("clean_classification.csv"))
    region = next(c for c in payload["column_profiles"] if c["name"] == "region")
    assert region["sparkline"]["kind"] == "categories"
    assert set(region["sparkline"]["labels"]) == {"north", "south", "east"}


def test_high_cardinality_tail_folds_into_other():
    """A ninth category is never a new colour — it becomes 'Other'."""
    frame = pd.DataFrame({"c": [f"v{i % 40}" for i in range(400)], "y": [0, 1] * 200})
    payload = profile_dataframe(frame)
    column = next(c for c in payload["column_profiles"] if c["name"] == "c")
    assert column["sparkline"]["labels"][-1] == "Other"
    assert len(column["sparkline"]["labels"]) == 9


def test_profile_flags_columns_that_cannot_be_targets(fixture_df):
    payload = profile_dataframe(fixture_df("free_text_target.csv"))
    findings = next(c for c in payload["column_profiles"] if c["name"] == "Findings")
    assert findings["usable_as_target"] is False
    assert "free text" in findings["target_reason"]
    assert findings["feature_note"] is not None


def test_profile_marks_usable_targets(fixture_df):
    payload = profile_dataframe(fixture_df("clean_classification.csv"))
    churn = next(c for c in payload["column_profiles"] if c["name"] == "churn")
    assert churn["usable_as_target"] is True


def test_profile_missing_percentages_are_accurate():
    frame = pd.DataFrame({"a": [1.0, None, 3.0, None], "b": [1, 2, 3, 4]})
    payload = profile_dataframe(frame)
    a = next(c for c in payload["column_profiles"] if c["name"] == "a")
    assert a["missing"] == 2
    assert a["missing_pct"] == 50.0


def test_profile_endpoint_returns_profile_without_training(client, fixture_csv):
    with fixture_csv("clean_classification.csv").open("rb") as handle:
        response = client.post("/api/profile", files={"file": ("d.csv", handle, "text/csv")})

    assert response.status_code == 200
    body = response.json()
    assert body["suggested_target"] == "churn"
    assert len(body["column_profiles"]) == 4
    assert "parse_report" in body
    # Nothing was trained, so no run was registered.
    assert client.get("/api/runs").json()["count"] == 0


def test_profile_endpoint_enforces_upload_rules(client):
    response = client.post("/api/profile", files={"file": ("x.exe", b"MZ", "application/x-msdownload")})
    assert response.status_code == 400


def test_profile_endpoint_rejects_binary_content_cleanly(client):
    """A file that passes size/extension checks but isn't text CSV.

    Regression test: read_csv_with_report used to raise a bare ValueError for
    this, which routes/profile.py never caught, so the request 500'd with a
    raw traceback instead of the same clean 400 test_profile_endpoint_enforces_
    upload_rules already covers for a too-small file.
    """
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" * 20
    response = client.post("/api/profile", files={"file": ("photo.csv", png_header, "text/csv")})
    assert response.status_code == 400
    assert "binary" in response.json()["detail"].lower()


# --- diagnostics ------------------------------------------------------------


def test_confusion_matrix_uses_real_class_names():
    matrix = build_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], ["no", "yes"])
    assert matrix["labels"] == ["no", "yes"]
    assert matrix["matrix"] == [[1, 1], [0, 2]]
    assert matrix["total"] == 4


def test_confusion_matrix_caps_the_class_count():
    truth = np.arange(60) % 30
    matrix = build_confusion_matrix(truth, truth, [str(i) for i in range(30)])
    assert matrix["truncated"] is True
    assert len(matrix["labels"]) <= 12


def test_residuals_carry_predicted_and_residual():
    payload = build_residuals([10.0, 20.0, 30.0], [12.0, 18.0, 33.0])
    assert [round(p["residual"], 2) for p in payload["points"]] == [-2.0, 2.0, -3.0]
    assert payload["sampled"] is False
    assert payload["total"] == 3


def test_residuals_are_sampled_for_large_test_sets():
    truth = np.linspace(0, 1000, 5000)
    payload = build_residuals(truth, truth + 1)
    assert payload["sampled"] is True
    assert len(payload["points"]) == 400
    assert payload["total"] == 5000


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 3, 4, 1, 2]})
    payload = build_correlation_matrix(frame)
    matrix = payload["matrix"]
    assert all(abs(matrix[i][i] - 1.0) < 1e-9 for i in range(len(matrix)))
    assert abs(matrix[0][1] - matrix[1][0]) < 1e-9
    assert payload["strongest_pairs"][0]["r"] == pytest.approx(1.0)


def test_correlation_needs_two_varying_numeric_columns():
    assert build_correlation_matrix(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})) is None
    assert build_correlation_matrix(pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})) is None


def test_correlation_matrix_caps_its_width():
    frame = pd.DataFrame({f"c{i}": np.random.default_rng(i).normal(0, i + 1, 50) for i in range(20)})
    payload = build_correlation_matrix(frame)
    assert payload["truncated"] is True
    assert len(payload["columns"]) == MAX_CORRELATION_COLUMNS


def test_classification_run_returns_a_confusion_matrix(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    diagnostics = job["result"]["diagnostics"]
    assert diagnostics["problem_type"] == "classification"
    assert diagnostics["residuals"] is None
    matrix = diagnostics["confusion_matrix"]
    assert set(matrix["labels"]) == {"yes", "no"}
    assert matrix["total"] > 0


def test_regression_run_returns_residuals(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_regression.csv"))
    diagnostics = job["result"]["diagnostics"]
    assert diagnostics["problem_type"] == "regression"
    assert diagnostics["confusion_matrix"] is None
    assert len(diagnostics["residuals"]["points"]) > 0


def test_runs_carry_a_correlation_matrix(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_regression.csv"))
    correlation = job["result"]["diagnostics"]["correlation"]
    assert "sqft" in correlation["columns"] and "price" in correlation["columns"]
    assert correlation["strongest_pairs"][0]["r"] > 0.9


def test_result_reports_the_parse_report(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    assert job["result"]["parse_report"]["rows_skipped"] == 0


# --- workspace --------------------------------------------------------------


def test_a_past_run_can_be_reopened(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    run_key = job["result"]["run_key"]

    response = client.get(f"/api/runs/{run_key}/result")

    assert response.status_code == 200
    reopened = response.json()
    assert reopened["status"] == "reopened"
    assert reopened["run_key"] == run_key
    assert reopened["health"]["verdict"] == job["result"]["health"]["verdict"]
    assert reopened["feature_importance"] == job["result"]["feature_importance"]
    assert reopened["diagnostics"]["confusion_matrix"] is not None


def test_reopening_an_unknown_run_is_404(client):
    assert client.get(f"/api/runs/{'a' * 64}/result").status_code == 404


def test_compare_returns_scores_for_several_runs(client, fixture_csv):
    first = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="auto")
    second = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="ensemble")
    keys = f"{first['result']['run_key']},{second['result']['run_key']}"

    body = client.get(f"/api/runs/compare?keys={keys}").json()

    assert len(body["runs"]) == 2
    assert body["comparable"] is True
    assert body["metric"] == "accuracy"
    assert all(item["scores"] for item in body["runs"])


def test_compare_refuses_to_mix_problem_types(client, fixture_csv):
    classification = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    regression = upload_and_wait(client, fixture_csv("clean_regression.csv"))
    keys = f"{classification['result']['run_key']},{regression['result']['run_key']}"

    body = client.get(f"/api/runs/compare?keys={keys}").json()

    # Accuracy and RMSE on one axis would be meaningless.
    assert body["comparable"] is False
    assert body["metric"] is None


def test_compare_is_capped(client):
    keys = ",".join(chr(97 + i) * 64 for i in range(5))
    assert client.get(f"/api/runs/compare?keys={keys}").status_code == 400


def test_compare_rejects_a_malformed_key(client):
    assert client.get("/api/runs/compare?keys=not-a-key").status_code == 400


def test_compare_route_is_not_shadowed_by_the_run_key_route(client, fixture_csv):
    """`/runs/compare` must not be matched as `/runs/{run_key}`."""
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    response = client.get(f"/api/runs/compare?keys={job['result']['run_key']}")
    assert response.status_code == 200
    assert "runs" in response.json()


def test_deleting_a_run_removes_its_row_and_its_artifacts(client, fixture_csv, isolated_storage):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    run_key = job["result"]["run_key"]
    artifacts = [
        isolated_storage["metadata_dir"] / f"{run_key}.json",
        isolated_storage["metadata_dir"] / f"{run_key}_data.csv",
        isolated_storage["model_dir"] / f"{run_key}_model.pkl",
        # Phase 6 signs every bundle, so the sidecar is part of the run and has
        # to go with it — a stale signature outliving its model is at best
        # confusing and at worst validates the wrong file later.
        isolated_storage["model_dir"] / f"{run_key}_model.pkl.sig",
    ]
    assert all(path.exists() for path in artifacts)

    body = client.delete(f"/api/runs/{run_key}").json()

    assert body["deleted"] is True
    assert body["artifacts_removed"] == len(artifacts)
    # Deleting a dataset means the snapshot goes too, not just the registry row.
    assert not any(path.exists() for path in artifacts)
    assert run_key not in [run["run_key"] for run in client.get("/api/runs").json()["runs"]]
    assert client.get(f"/api/runs/{run_key}/result").status_code == 404


def test_deleting_an_unknown_run_is_404(client):
    assert client.delete(f"/api/runs/{'a' * 64}").status_code == 404


def test_deleting_rejects_a_malformed_key(client):
    assert client.delete("/api/runs/not-a-key").status_code == 400
