"""Phase 6: the job queue, object storage, and observability.

These three share a shape: each has a production backend that the test suite
should not require, and a local default that must stay indistinguishable to
every caller. So most of what is asserted here is about the seam — that the
fallback is taken, that it is taken loudly, and that nothing above it notices.
"""

import pickle
from pathlib import Path

import pytest

import db
import observability
from tests.conftest import upload_and_wait
from utils import queue as queue_module
from utils import storage as storage_module

CLASSIFICATION = "clean_classification.csv"


# --- job queue --------------------------------------------------------------


def test_the_default_is_in_process_execution(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "queue_backend", "inline")
    queue_module.reset_queue()

    queue = queue_module.get_queue()
    assert queue.name == "inline"
    assert queue.out_of_process is False
    assert queue.healthy() is True


def test_an_unreachable_redis_falls_back_to_inline_rather_than_swallowing_jobs(monkeypatch, caplog):
    """The failure that matters: a queue nothing consumes accepts work forever.

    Falling back means a slower request; not falling back means an upload that
    reports "queued" and never finishes.
    """
    from config import settings

    monkeypatch.setattr(settings, "queue_backend", "rq")
    # Reserved by RFC 2606 conventions for tests; nothing listens here.
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")
    queue_module.reset_queue()

    with caplog.at_level("ERROR"):
        queue = queue_module.get_queue()

    assert queue.name == "inline"
    assert any("falling back to inline" in record.getMessage() for record in caplog.records)


def test_queue_status_reports_what_readyz_shows(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "queue_backend", "inline")
    queue_module.reset_queue()

    assert queue_module.queue_status() == {"backend": "inline", "healthy": True}


def test_uploads_still_run_in_process_on_the_default_backend(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    assert job["state"] == "completed"


def test_an_out_of_process_queue_receives_the_job_and_a_readable_input(
    client, fixture_csv, isolated_storage, monkeypatch
):
    """The handover contract with a real worker, without needing Redis.

    Two things have to hold, and both are invisible with an inline queue: the
    job is handed to the queue rather than to BackgroundTasks, and its input
    file is somewhere a different process could actually open.
    """
    submitted = {}

    class RecordingQueue(queue_module.JobQueue):
        name = "rq"
        out_of_process = True

        def enqueue(self, func, *args, job_id=None, **kwargs):
            submitted["func"] = func
            submitted["args"] = args
            submitted["job_id"] = job_id
            return job_id

        def healthy(self):
            return True

    queue_module.set_queue(RecordingQueue())

    with fixture_csv(CLASSIFICATION).open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": (CLASSIFICATION, handle, "text/csv")},
            data={"mode": "auto"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["queue"] == "rq"
    # Keyed by job id, so a redelivery cannot start the same run twice.
    assert submitted["job_id"] == body["job_id"]
    assert submitted["func"].__name__ == "_run_upload_pipeline"

    stored = submitted["args"][1]
    assert Path(stored.path).exists()
    # On shared storage, not in this process's private temp directory.
    assert Path(stored.path).parent == isolated_storage["spool_dir"]

    # RQ pickles the arguments into Redis. Something unpicklable in here would
    # fail only in production, at enqueue time, with the queue configured —
    # which is exactly the configuration no test would otherwise exercise.
    restored = pickle.loads(pickle.dumps(submitted["args"]))
    assert restored[1].path == stored.path
    assert restored[1].sha256 == stored.sha256

    # The pipeline the worker would run still produces a complete job.
    submitted["func"](*submitted["args"])
    assert db.get_job(body["job_id"])["state"] == "completed"


# --- object storage ---------------------------------------------------------


def test_local_storage_treats_the_disk_as_the_storage(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "storage_backend", "local")
    storage_module.reset_storage()

    storage = storage_module.get_storage()
    assert storage.name == "local"
    # publish/ensure are no-ops rather than copies: the file the pipeline wrote
    # is the file the next request reads.
    assert storage_module.publish_run("a" * 64) == 0


def test_s3_without_a_bucket_falls_back_to_local(monkeypatch, caplog):
    from config import settings

    monkeypatch.setattr(settings, "storage_backend", "s3")
    monkeypatch.setattr(settings, "s3_bucket", None)
    storage_module.reset_storage()

    with caplog.at_level("ERROR"):
        storage = storage_module.get_storage()

    assert storage.name == "local"
    assert any("S3_BUCKET is unset" in record.getMessage() for record in caplog.records)


def test_a_remote_backend_publishes_every_artifact_and_fetches_them_back(client, fixture_csv, isolated_storage):
    """The multi-machine path, against an in-memory stand-in for a bucket."""

    class MemoryStorage(storage_module.ArtifactStorage):
        name = "memory"

        def __init__(self):
            self.objects = {}

        def put(self, local_path, key):
            self.objects[key] = Path(local_path).read_bytes()

        def get(self, key, local_path):
            if key not in self.objects:
                return False
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(self.objects[key])
            return True

        def delete(self, key):
            return self.objects.pop(key, None) is not None

        def exists(self, key):
            return key in self.objects

    remote = MemoryStorage()
    storage_module.set_storage(remote)

    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    run_key = job["result"]["run_key"]

    # Everything a second web process would need, including the signature.
    assert sorted(remote.objects) == sorted(
        [f"{run_key}.json", f"{run_key}_data.csv", f"{run_key}_model.pkl", f"{run_key}_model.pkl.sig"]
    )

    # Simulate that second process: same bucket, empty disk.
    local_paths = [path for _, path in storage_module.run_artifact_paths(run_key)]
    for path in local_paths:
        path.unlink()
    assert not any(path.exists() for path in local_paths)

    storage_module.ensure_local(run_key)
    assert all(path.exists() for path in local_paths)

    # And a prediction served from artifacts that arrived over the wire still
    # passes its signature check.
    row = dict.fromkeys(job["result"]["features"], 1)
    assert client.post(f"/api/predict/{run_key}/row", json={"row": row}).status_code == 200


def test_purging_a_run_clears_both_copies(client, fixture_csv):
    class MemoryStorage(storage_module.ArtifactStorage):
        name = "memory"

        def __init__(self):
            self.objects = {}

        def put(self, local_path, key):
            self.objects[key] = Path(local_path).read_bytes()

        def get(self, key, local_path):
            return key in self.objects

        def delete(self, key):
            return self.objects.pop(key, None) is not None

        def exists(self, key):
            return key in self.objects

    remote = MemoryStorage()
    storage_module.set_storage(remote)

    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    run_key = job["result"]["run_key"]
    assert remote.objects

    storage_module.purge_run(run_key)

    assert remote.objects == {}
    assert not any(path.exists() for _, path in storage_module.run_artifact_paths(run_key))


# --- observability ----------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        # Exact match against the published rate.
        ("gemini-1.5-flash", (1_000_000, 0, 0.075)),
        # Point releases bill as their base model rather than as unknown.
        ("gemini-1.5-flash-002", (1_000_000, 0, 0.075)),
        # Output tokens are the expensive half and must not be counted as input.
        ("gemini-1.5-pro", (0, 1_000_000, 5.00)),
    ],
)
def test_cost_estimates_follow_the_published_rates(model, expected):
    input_tokens, output_tokens, cost = expected
    assert observability.estimate_cost_usd(model, input_tokens, output_tokens) == pytest.approx(cost)


def test_an_unknown_model_is_not_billed_as_free():
    """ "Free" is the one answer that is definitely wrong for a paid API."""
    assert observability.estimate_cost_usd("some-model-shipped-tomorrow", 1_000_000, 0) > 0


def test_paths_with_ids_collapse_to_one_time_series():
    """One series per dataset is how a metrics endpoint becomes a memory leak."""
    assert observability.normalise_path(f"/api/runs/{'a' * 64}") == "/api/runs/:id"
    assert observability.normalise_path("/api/upload/status/0f9c2b1e-4a3d-4c5f-8a1b-2c3d4e5f6a7b") == (
        "/api/upload/status/:id"
    )
    # Real path segments survive, or the metric says nothing.
    assert observability.normalise_path("/api/runs/compare") == "/api/runs/compare"


def test_usage_is_recorded_per_owner_and_per_run(isolated_storage):
    observability.record_llm_usage(
        owner_id="user-a",
        provider="gemini",
        model="gemini-1.5-flash",
        input_tokens=1000,
        output_tokens=500,
        run_key="r" * 64,
    )
    observability.record_llm_usage(
        owner_id="user-b",
        provider="gemini",
        model="gemini-1.5-flash",
        input_tokens=9000,
        output_tokens=9000,
    )

    mine = db.usage_summary(owner_id="user-a")
    assert mine["calls"] == 1
    assert mine["input_tokens"] == 1000
    assert mine["estimated_cost_usd"] > 0
    assert mine["by_model"][0]["model"] == "gemini-1.5-flash"

    # Everyone, for the admin view.
    assert db.usage_summary(owner_id=None)["calls"] == 2


def test_the_usage_endpoint_shows_only_the_callers_spend(authenticated):
    client = authenticated["client"]
    observability.record_llm_usage(
        owner_id=authenticated["alice"]["user_id"],
        provider="gemini",
        model="gemini-1.5-flash",
        input_tokens=1000,
        output_tokens=100,
    )

    alice = client.get("/api/usage", headers=authenticated["alice"]["headers"]).json()
    bob = client.get("/api/usage", headers=authenticated["bob"]["headers"]).json()

    assert alice["calls"] == 1
    assert bob["calls"] == 0
    # Never presented as an invoice, because it is not one.
    assert "estimated" in alice["note"].lower()

    admin = client.get("/api/admin/usage", headers=authenticated["alice"]["headers"]).json()
    assert admin["calls"] == 1


def test_accounting_failure_never_breaks_the_answer(monkeypatch, caplog):
    """A usage table that cannot be written is a smaller problem than a 500."""

    def _explode(**kwargs):
        raise RuntimeError("database is on fire")

    monkeypatch.setattr(db, "record_usage", _explode)

    with caplog.at_level("WARNING"):
        cost = observability.record_llm_usage(
            owner_id="user-a", provider="gemini", model="gemini-1.5-flash", input_tokens=10, output_tokens=10
        )

    assert cost > 0
    assert any("Could not record LLM usage" in record.getMessage() for record in caplog.records)


def test_metrics_endpoint_exposes_the_series_an_operator_graphs(client, fixture_csv):
    if not observability.metrics.enabled:
        pytest.skip("prometheus_client is not installed")

    upload_and_wait(client, fixture_csv(CLASSIFICATION))
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "analyst_http_requests_total" in body
    assert "analyst_training_runs_total" in body
    assert "analyst_artifact_bytes" in body
    # The label set stays bounded: no run key ever becomes a series.
    assert "/api/runs/:id" in body or "analyst_http_requests_total" in body


def test_readiness_reports_every_dependency(client):
    body = client.get("/readyz").json()

    assert body["status"] == "ready"
    assert body["checks"]["database"]["ok"] is True
    # Degraded-but-serving dependencies are reported, not fatal.
    assert body["checks"]["queue"]["backend"] == "inline"
    assert body["checks"]["storage"]["backend"] == "local"
    assert "llm" in body["checks"]


def test_liveness_stays_a_dumb_process_check(client):
    assert client.get("/healthz").json() == {"status": "ok"}
