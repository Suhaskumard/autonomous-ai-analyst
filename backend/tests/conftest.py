"""Shared fixtures.

Every test runs against a temporary database and temporary artifact
directories, so a test run never touches (or is influenced by) real models.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Modules that captured METADATA_DIR / MODEL_DIR at import time. Redirecting
# storage means rebinding the name in each of them, not just in utils.helpers.
_DIR_HOLDERS = ("utils.helpers", "routes.upload", "routes.predict", "routes.insights", "routes.chat", "routes.runs")


@pytest.fixture
def fixture_csv():
    """Path to a checked-in fixture CSV."""

    def _load(name: str) -> Path:
        path = FIXTURE_DIR / name
        assert path.exists(), f"missing fixture {name}"
        return path

    return _load


@pytest.fixture
def fixture_df(fixture_csv):
    def _load(name: str) -> pd.DataFrame:
        return pd.read_csv(fixture_csv(name))

    return _load


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Temporary artifact dirs + a temporary SQLite database."""
    import importlib

    metadata_dir = tmp_path / "metadata"
    model_dir = tmp_path / "saved_models"
    metadata_dir.mkdir()
    model_dir.mkdir()

    for module_name in _DIR_HOLDERS:
        module = importlib.import_module(module_name)
        if hasattr(module, "METADATA_DIR"):
            monkeypatch.setattr(module, "METADATA_DIR", metadata_dir, raising=False)
        if hasattr(module, "MODEL_DIR"):
            monkeypatch.setattr(module, "MODEL_DIR", model_dir, raising=False)

    import db
    from config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    # No test may reach a real LLM. backend/.env is loaded at import, so without
    # this the analyst tests would bill (and leak) the operator's actual key.
    # Tests that need a model install analyst.providers.set_provider_override.
    monkeypatch.setattr(settings, "gemini_api_key", None)
    # Cross-validation multiplies every fit by the fold count, so a full
    # ten-candidate sweep costs ~12s per pipeline run and most tests here are
    # exercising plumbing, not model breadth. Four candidates keeps every code
    # path live at a fraction of the cost; test_model_selection.py clears the
    # cap where the breadth itself is what is under test.
    monkeypatch.setattr(settings, "max_candidate_models", 4)
    db.reset_engine()
    db.init_db()

    yield {"metadata_dir": metadata_dir, "model_dir": model_dir, "db_path": tmp_path / "test.db"}

    db.reset_engine()


@pytest.fixture
def client(isolated_storage):
    """A TestClient whose storage and database are isolated."""
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as test_client:
        yield test_client


def upload_and_wait(client, csv_path: Path, **form):
    """POST a CSV and return the finished job. BackgroundTasks run inline under
    TestClient, so the job is already terminal by the time status is polled."""
    with csv_path.open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": (csv_path.name, handle, "text/csv")},
            data={"mode": form.pop("mode", "auto"), **{k: v for k, v in form.items() if v is not None}},
        )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    return client.get(f"/api/upload/status/{job_id}").json()
