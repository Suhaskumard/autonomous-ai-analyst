"""The finished prediction flow: probabilities, tables, and single-row entry."""

import io

import pandas as pd
import pytest

from tests.conftest import upload_and_wait


@pytest.fixture
def trained(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    assert job["state"] == "completed", job.get("error")
    return job["result"]


@pytest.fixture
def sample_csv(fixture_csv):
    frame = pd.read_csv(fixture_csv("clean_classification.csv")).drop(columns=["churn"]).head(6)
    return frame.to_csv(index=False).encode()


def test_predictions_come_back_as_rows_with_probabilities(client, trained, sample_csv):
    response = client.post(
        f"/api/predict/{trained['run_key']}", files={"file": ("p.csv", io.BytesIO(sample_csv), "text/csv")}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["rows"]) == 6
    first = body["rows"][0]
    assert first["prediction"] in {"yes", "no"}
    assert 0.0 <= first["confidence"] <= 1.0
    # Per-class probabilities, so the UI can show what came second.
    assert set(first["class_probabilities"]) == {"yes", "no"}
    assert sum(first["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_response_carries_the_input_rows_for_the_results_table(client, trained, sample_csv):
    body = client.post(
        f"/api/predict/{trained['run_key']}", files={"file": ("p.csv", io.BytesIO(sample_csv), "text/csv")}
    ).json()
    assert len(body["input_rows"]) == 6
    assert set(body["features"]) == {"age", "region", "spend"}
    assert body["target"] == "churn"


def test_flat_prediction_list_is_still_present(client, trained, sample_csv):
    body = client.post(
        f"/api/predict/{trained['run_key']}", files={"file": ("p.csv", io.BytesIO(sample_csv), "text/csv")}
    ).json()
    assert body["predictions"] == [row["prediction"] for row in body["rows"]]


def test_a_single_pasted_row_can_be_predicted(client, trained):
    response = client.post(
        f"/api/predict/{trained['run_key']}/row",
        json={"row": {"age": "70", "region": "north", "spend": "51.2"}},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["rows"]) == 1
    assert body["rows"][0]["prediction"] in {"yes", "no"}
    assert body["rows"][0]["confidence"] is not None


def test_single_row_numeric_strings_are_coerced(client, trained):
    """Form values arrive as strings; they must not reach the scaler as text."""
    typed = client.post(
        f"/api/predict/{trained['run_key']}/row", json={"row": {"age": "70", "region": "north", "spend": "51.2"}}
    ).json()
    numeric = client.post(
        f"/api/predict/{trained['run_key']}/row", json={"row": {"age": 70, "region": "north", "spend": 51.2}}
    ).json()
    assert typed["rows"][0]["prediction"] == numeric["rows"][0]["prediction"]


def test_single_row_requires_every_feature(client, trained):
    response = client.post(f"/api/predict/{trained['run_key']}/row", json={"row": {"age": 30}})
    assert response.status_code == 400
    assert "Missing required features" in response.json()["detail"]


def test_single_row_rejects_an_empty_body(client, trained):
    assert client.post(f"/api/predict/{trained['run_key']}/row", json={"row": {}}).status_code == 400


def test_single_row_validates_the_run_key(client):
    assert client.post("/api/predict/not-a-key/row", json={"row": {"a": 1}}).status_code == 400


def test_regression_predictions_have_no_probabilities(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_regression.csv"))
    frame = pd.read_csv(fixture_csv("clean_regression.csv")).drop(columns=["price"]).head(3)
    response = client.post(
        f"/api/predict/{job['result']['run_key']}",
        files={"file": ("p.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["confidences"] is None
    assert all(isinstance(row["prediction"], float) for row in body["rows"])
    assert "class_probabilities" not in body["rows"][0]


def test_ensemble_predictions_report_no_confidence(client, fixture_csv, sample_csv):
    """A majority vote has no calibrated probability to report."""
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), mode="ensemble")
    body = client.post(
        f"/api/predict/{job['result']['run_key']}", files={"file": ("p.csv", io.BytesIO(sample_csv), "text/csv")}
    ).json()
    assert body["confidences"] is None
    assert all(row["prediction"] in {"yes", "no"} for row in body["rows"])
