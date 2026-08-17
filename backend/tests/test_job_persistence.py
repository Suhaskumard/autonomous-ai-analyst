"""Job state must survive a process restart.

Regression guard: UPLOAD_JOBS was a module-level dict — lost on restart,
unbounded, and invisible to a second uvicorn worker.
"""

from datetime import timedelta

import db
from tests.conftest import upload_and_wait
from utils.helpers import now_utc


def test_job_survives_a_simulated_restart(client, fixture_csv):
    """Drop every in-process cache the way a restart would, then read it back."""
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    job_id = job["job_id"]

    db.reset_engine()  # as if the process had exited and come back

    restored = client.get(f"/api/upload/status/{job_id}").json()
    assert restored["state"] == "completed"
    assert restored["result"]["run_key"] == job["result"]["run_key"]
    assert restored["status_log"], "the progress log must persist too"


def test_interrupted_jobs_are_recovered_as_failed(isolated_storage):
    """A job left "running" by a crash must not poll forever."""
    db.create_job("orphan", "Upload received")
    db.append_step("orphan", "Training models", 60)
    assert db.get_job("orphan")["state"] == "running"

    recovered = db.recover_interrupted_jobs()

    assert recovered == 1
    job = db.get_job("orphan")
    assert job["state"] == "failed"
    assert job["error_kind"] == "interrupted"
    assert "restarted" in job["error"]


def test_recovery_leaves_finished_jobs_alone(isolated_storage):
    db.create_job("done", "Upload received")
    db.complete_job("done", {"status": "trained"}, "a" * 64)
    db.recover_interrupted_jobs()
    assert db.get_job("done")["state"] == "completed"


def test_old_jobs_are_purged(isolated_storage):
    db.create_job("ancient", "Upload received")
    with db.session_scope() as session:
        record = session.get(db.JobRecord, "ancient")
        record.created_at = now_utc() - timedelta(hours=200)
        session.add(record)

    assert db.purge_old_jobs(max_age_hours=72) == 1
    assert db.get_job("ancient") is None


def test_completed_run_lands_in_the_registry(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    run_key = job["result"]["run_key"]

    listing = client.get("/api/runs").json()
    assert listing["count"] >= 1
    assert run_key in {run["run_key"] for run in listing["runs"]}

    detail = client.get(f"/api/runs/{run_key}").json()
    assert detail["target"] == "churn"
    assert detail["filename"] == "clean_classification.csv"
    assert detail["row_count"] > 0


def test_deleting_a_run_removes_its_artifacts(client, fixture_csv, isolated_storage):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    run_key = job["result"]["run_key"]
    model_file = isolated_storage["model_dir"] / f"{run_key}_model.pkl"
    assert model_file.exists()

    response = client.delete(f"/api/runs/{run_key}")

    assert response.status_code == 200
    assert response.json()["artifacts_removed"] >= 1
    assert not model_file.exists()
    assert client.get(f"/api/runs/{run_key}").status_code == 404


def test_registry_is_shared_across_engine_resets(client, fixture_csv):
    """Two workers would see the same rows; a reset engine is the same test."""
    upload_and_wait(client, fixture_csv("clean_regression.csv"))
    db.reset_engine()
    assert client.get("/api/runs").json()["count"] >= 1


def test_cache_hit_still_registers_the_run(client, fixture_csv):
    """Artifacts can outlive their registry row — a fresh database, a restored
    volume, a deleted row. A reused run must still appear in the history."""
    first = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    run_key = first["result"]["run_key"]

    assert db.delete_run(run_key) is True
    assert client.get("/api/runs").json()["count"] == 0

    second = upload_and_wait(client, fixture_csv("clean_classification.csv"))

    assert second["result"]["status"] == "reused", "artifacts should still be on disk"
    assert client.get(f"/api/runs/{run_key}").status_code == 200


def test_registry_row_records_the_trained_shape(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    run = client.get(f"/api/runs/{job['result']['run_key']}").json()
    assert run["row_count"] > 100
    assert run["column_count"] == 4
    assert run["health_verdict"] in {"strong", "fair", "unreliable"}


def test_recovery_runs_on_application_startup(isolated_storage):
    """Not just the helper — the lifespan hook must actually call it.

    Verified end to end against a real killed uvicorn process too; this keeps
    the wiring covered in CI, where killing a process is impractical.
    """
    from fastapi.testclient import TestClient

    import main

    db.create_job("stuck", "Upload received")
    db.append_step("stuck", "Training models", 60)
    assert db.get_job("stuck")["state"] == "running"

    # Entering the context manager runs the app's startup hooks.
    with TestClient(main.app) as fresh_client:
        status = fresh_client.get("/api/upload/status/stuck").json()

    assert status["state"] == "failed"
    assert status["error_kind"] == "interrupted"
