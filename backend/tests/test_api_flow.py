"""End-to-end HTTP behaviour: upload -> poll -> predict -> insights -> runs."""

import io

import pytest

from tests.conftest import upload_and_wait


@pytest.fixture
def trained_job(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="auto")
    assert job["state"] == "completed", job.get("error")
    return job


def test_upload_trains_and_reports_a_run_key(trained_job):
    result = trained_job["result"]
    assert result["status"] == "trained"
    assert len(result["run_key"]) == 64
    assert result["problem_type"] == "classification"
    assert result["target"] == "churn"


def test_result_carries_a_health_verdict(trained_job):
    health = trained_job["result"]["health"]
    assert health["verdict"] in {"strong", "fair", "unreliable"}
    assert "baseline" in health and "score" in health


def test_importance_labels_are_source_columns(trained_job):
    reported = {item["feature"] for item in trained_job["result"]["feature_importance"]}
    assert reported <= {"age", "region", "spend"}


def test_changing_mode_retrains_instead_of_replaying(client, fixture_csv, trained_job):
    ensemble = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="ensemble")
    assert ensemble["result"]["status"] == "trained"
    assert ensemble["result"]["mode"] == "ensemble"
    assert ensemble["result"]["run_key"] != trained_job["result"]["run_key"]


def test_identical_configuration_hits_the_cache(client, fixture_csv, trained_job):
    again = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="auto")
    assert again["result"]["status"] == "reused"
    assert again["result"]["run_key"] == trained_job["result"]["run_key"]


def test_changing_target_retrains(client, fixture_csv, trained_job):
    other = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="auto", target_column="region")
    assert other["result"]["status"] == "trained"
    assert other["result"]["target"] == "region"


def test_ensemble_is_scored_on_its_own_predictions(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="ensemble")
    scores = job["result"]["all_model_scores"]
    assert "Ensemble" in scores, "the vote itself must be scored, not the best member"


def test_predictions_come_back_as_original_labels(client, fixture_csv, trained_job):
    run_key = trained_job["result"]["run_key"]
    df = __import__("pandas").read_csv(fixture_csv("clean_classification.csv")).drop(columns=["churn"]).head(5)
    payload = df.to_csv(index=False).encode()
    response = client.post(f"/api/predict/{run_key}", files={"file": ("p.csv", io.BytesIO(payload), "text/csv")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert all(isinstance(p, str) and p in {"yes", "no"} for p in body["predictions"])
    assert len(body["confidences"]) == 5


def test_predict_rejects_missing_feature_columns(client, fixture_csv, trained_job):
    run_key = trained_job["result"]["run_key"]
    response = client.post(f"/api/predict/{run_key}", files={"file": ("p.csv", io.BytesIO(b"age\n30\n"), "text/csv")})
    assert response.status_code == 400
    assert "Missing required column(s)" in response.json()["detail"]


def test_insights_endpoint_mirrors_the_result(client, trained_job):
    run_key = trained_job["result"]["run_key"]
    body = client.get(f"/api/insights/{run_key}").json()
    assert body["health"]["verdict"] == trained_job["result"]["health"]["verdict"]
    assert body["explanation_method"] in {"shap", "native", "unavailable"}


def test_regression_dataset_trains_and_beats_the_mean(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_regression.csv"), mode="auto")
    assert job["state"] == "completed", job.get("error")
    result = job["result"]
    assert result["problem_type"] == "regression"
    assert result["health"]["verdict"] in {"strong", "fair"}
    assert result["health"]["score"] < result["health"]["baseline"]


def test_free_text_target_fails_the_job_with_a_target_kind(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("free_text_target.csv"), mode="auto")
    assert job["state"] == "failed"
    assert job["error_kind"] == "target"
    assert "Findings" in job["error"]


def test_unknown_job_id_is_404(client):
    assert client.get("/api/upload/status/does-not-exist").status_code == 404


def test_health_and_readiness_probes(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] == "ready"


def test_every_response_carries_a_request_id(client):
    response = client.get("/healthz")
    assert response.headers.get("x-request-id")


def test_supplied_request_id_is_echoed(client):
    response = client.get("/healthz", headers={"x-request-id": "trace-me"})
    assert response.headers["x-request-id"] == "trace-me"
