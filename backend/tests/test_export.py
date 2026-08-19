"""Phase 10: exporting a trained pipeline (ONNX and the portable bundle)."""

import zipfile
from io import BytesIO

import numpy as np
import pandas as pd

from tests.conftest import upload_and_wait


def _numeric_only_csv(tmp_path):
    """A dataset with no categorical column, so ONNX conversion can succeed."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(scale=5, size=n) + 10,
            "target": rng.choice(["yes", "no"], size=n),
        }
    )
    path = tmp_path / "numeric_only.csv"
    df.to_csv(path, index=False)
    return path


def test_bundle_export_always_available(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), target_column="churn")
    run_key = job["result"]["run_key"]

    availability = client.get(f"/api/runs/{run_key}/export/availability")
    assert availability.status_code == 200, availability.text
    assert availability.json()["bundle"]["available"] is True

    response = client.get(f"/api/runs/{run_key}/export?format=bundle")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"

    archive = zipfile.ZipFile(BytesIO(response.content))
    names = archive.namelist()
    assert any(name.endswith("_model.pkl") for name in names)
    assert "score.py" in names
    assert "requirements.txt" in names
    assert "README.md" in names


def test_onnx_export_refused_for_categorical_features(client, fixture_csv):
    """clean_classification.csv has a `region` column, which is one-hot encoded —
    skl2onnx's SimpleImputer converter cannot handle that branch (see ml/export.py)."""
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), target_column="churn")
    run_key = job["result"]["run_key"]

    availability = client.get(f"/api/runs/{run_key}/export/availability")
    assert availability.status_code == 200
    onnx_availability = availability.json()["onnx"]
    assert onnx_availability["available"] is False
    assert "categorical" in onnx_availability["reason"]

    response = client.get(f"/api/runs/{run_key}/export?format=onnx")
    assert response.status_code == 422
    assert "categorical" in response.json()["detail"]


def test_onnx_export_succeeds_for_a_numeric_only_run(client, tmp_path):
    csv_path = _numeric_only_csv(tmp_path)
    job = upload_and_wait(client, csv_path, mode="manual", manual_model="RandomForest", target_column="target")
    run_key = job["result"]["run_key"]

    availability = client.get(f"/api/runs/{run_key}/export/availability")
    assert availability.status_code == 200, availability.text
    assert availability.json()["onnx"]["available"] is True

    response = client.get(f"/api/runs/{run_key}/export?format=onnx")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/octet-stream"

    import onnx

    model = onnx.load_from_string(response.content)
    onnx.checker.check_model(model)


def test_export_requires_ownership(client, fixture_csv, monkeypatch):
    import auth
    from config import settings

    monkeypatch.setattr(settings, "auth_enabled", True)
    _, alice_key = auth.create_user("alice-export@example.com")
    _, bob_key = auth.create_user("bob-export@example.com")
    alice_headers = {"Authorization": f"Bearer {alice_key}"}
    bob_headers = {"Authorization": f"Bearer {bob_key}"}

    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), headers=alice_headers, target_column="churn")
    run_key = job["result"]["run_key"]

    response = client.get(f"/api/runs/{run_key}/export", headers=bob_headers)
    assert response.status_code == 404
