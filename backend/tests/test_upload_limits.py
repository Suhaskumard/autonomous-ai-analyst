"""Upload ceilings and type checks.

Regression guard: `await file.read()` pulled the whole upload into RAM with no
size cap, type check, or row limit.
"""

import glob
import io
import os
import tempfile

import pytest

from config import settings


@pytest.fixture(autouse=True)
def small_caps(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    monkeypatch.setattr(settings, "max_upload_rows", 50)


def post(client, name, payload, content_type="text/csv"):
    return client.post("/api/upload", files={"file": (name, io.BytesIO(payload), content_type)})


def test_oversized_upload_is_refused(client):
    """Few rows, many bytes — isolates the byte cap from the row cap."""
    payload = b"a,b\n" + b"x" * 2_000_000 + b"\n"
    response = post(client, "big.csv", payload)
    assert response.status_code == 413
    assert "MB limit" in response.json()["detail"]


def test_row_cap_is_enforced_while_streaming(client):
    payload = b"a,b\n" + b"1,2\n" * 200
    response = post(client, "rows.csv", payload)
    assert response.status_code == 413
    assert "row limit" in response.json()["detail"]


@pytest.mark.parametrize(
    ("name", "content_type"),
    [("evil.exe", "application/x-msdownload"), ("archive.zip", "application/zip"), ("noext", "text/csv")],
)
def test_disallowed_file_types_are_refused(client, name, content_type):
    assert post(client, name, b"a,b\n1,2\n", content_type).status_code == 400


def test_empty_upload_is_refused(client):
    assert post(client, "empty.csv", b"").status_code == 400


def test_accepted_upload_leaves_no_temp_file(client, fixture_csv):
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "aia_upload_*")))
    payload = fixture_csv("clean_classification.csv").read_bytes()
    # Raise the caps for this one case so the fixture actually gets through.
    settings.max_upload_rows = 100_000
    response = post(client, "ok.csv", payload)
    assert response.status_code == 200
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "aia_upload_*")))
    assert after <= before


def test_rejected_upload_leaves_no_temp_file(client):
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "aia_upload_*")))
    post(client, "big.csv", b"a,b\n" + b"x" * 2_000_000 + b"\n")
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "aia_upload_*")))
    assert after <= before


def test_tiny_dataset_is_rejected_by_the_pipeline(client):
    """Under the row floor: the job fails with a dataset-kind error."""
    response = post(client, "tiny.csv", b"a,b\n1,2\n3,4\n")
    assert response.status_code == 200
    job = client.get(f"/api/upload/status/{response.json()['job_id']}").json()
    assert job["state"] == "failed"
    assert job["error_kind"] == "dataset"
