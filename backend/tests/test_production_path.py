"""Phase 7: the production path, asserted rather than assumed.

Phase 6 wrote the multi-process shape and tested it against substitutes. What
was missing was evidence, and three of the things it substituted for cannot be
checked by reading the code:

* the schema is now Alembic's, and a database written before Phase 7 has to
  survive the change without losing a row;
* an artifact signed by one process has to verify in another, which a
  generated-per-machine key silently breaks;
* a run trained on one machine has to be servable from a second one that has
  never seen the file.

Everything here runs on the default install. The parts that genuinely need
Postgres or MinIO are opt-in through TEST_DATABASE_URL and MINIO_TEST_ENDPOINT
and skip otherwise, because a suite that needs a container is a suite that
stops being run.
"""

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlmodel import SQLModel

import db
from exceptions import ArtifactError
from tests.conftest import upload_and_wait
from utils import artifacts as artifacts_module
from utils import migrations as migrations_module
from utils import queue as queue_module
from utils import storage as storage_module

BACKEND_DIR = Path(__file__).resolve().parents[1]
CLASSIFICATION = "clean_classification.csv"


class _DirectoryStorage(storage_module.ArtifactStorage):
    """A bucket that is a directory: the S3 contract without the S3.

    `MemoryStorage` in test_operations.py proves publish and fetch. This one
    survives a storage handle being reset, because the bytes live on disk
    rather than inside the object — which is what lets one test act as two
    machines sharing a bucket.
    """

    name = "directory"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _object(self, key: str) -> Path:
        return self.root / key

    def put(self, local_path: Path, key: str) -> None:
        self._object(key).write_bytes(Path(local_path).read_bytes())

    def get(self, key: str, local_path: Path) -> bool:
        source = self._object(key)
        if not source.exists():
            return False
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(source.read_bytes())
        return True

    def delete(self, key: str) -> bool:
        target = self._object(key)
        if target.exists():
            target.unlink()
            return True
        return False

    def exists(self, key: str) -> bool:
        return self._object(key).exists()


# --- schema ownership --------------------------------------------------------


def test_startup_leaves_the_schema_at_head(isolated_storage):
    """init_db migrates rather than create_all's, and says which revision it is at."""
    engine = db.get_engine()
    with engine.connect() as connection:
        assert migrations_module.current_revision(connection) == migrations_module.head_revision()

    assert migrations_module.pending_revisions(engine) == []
    assert migrations_module.schema_status(engine)["ok"] is True


def test_a_current_pre_alembic_database_is_stamped_and_keeps_its_rows(tmp_path, monkeypatch):
    """The upgrade *into* Alembic, which every existing install has to survive.

    The easy shape first: a Phase 6 database, which has every table and no
    version row. Upgrading it would try to create tables that are already
    there; it is stamped instead, because the baseline is exactly what
    `create_all` used to produce.
    """
    from config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}")
    db.reset_engine()

    engine = db.get_engine()
    SQLModel.metadata.create_all(engine)
    with engine.connect() as connection:
        assert migrations_module.current_revision(connection) is None
    db.upsert_run(run_key="c" * 64, dataset_hash="d" * 64, filename="legacy.csv", row_count=10, column_count=3)

    db.init_db()

    engine = db.get_engine()
    with engine.connect() as connection:
        assert migrations_module.current_revision(connection) == migrations_module.head_revision()
    assert db.get_run("c" * 64)["filename"] == "legacy.csv"

    db.reset_engine()


def test_a_phase_2_database_is_brought_up_to_the_baseline(tmp_path, monkeypatch):
    """The hard shape, and the one actually sitting in this repo.

    `backend/models/analyst.db` here has two of the six tables and none of Phase
    6's columns, because the old startup only ever created what the phase
    running at the time knew about. Stamping *that* at the baseline would
    declare a schema complete while `users` did not exist and `runs` had no
    `owner_id` — every ownership query would then fail at runtime, which is the
    exact failure the migration framework was adopted to stop.
    """
    from config import settings

    database = tmp_path / "phase2.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{database.as_posix()}")
    db.reset_engine()

    # Phase 2's schema, written by hand because no version of the models can
    # produce it any more: two tables, and no owner anywhere.
    with sqlite3.connect(database) as legacy:
        legacy.execute(
            "CREATE TABLE runs (run_key VARCHAR PRIMARY KEY, dataset_hash VARCHAR NOT NULL, "
            "filename VARCHAR, mode VARCHAR NOT NULL, target VARCHAR, problem_type VARCHAR, "
            "selected_model VARCHAR, health_verdict VARCHAR, row_count INTEGER NOT NULL, "
            "column_count INTEGER NOT NULL, pipeline_version VARCHAR, created_at DATETIME NOT NULL)"
        )
        legacy.execute(
            "CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY, state VARCHAR NOT NULL, "
            "current_step VARCHAR NOT NULL, progress INTEGER NOT NULL, status_log JSON, result JSON, "
            "error VARCHAR, error_kind VARCHAR, run_key VARCHAR, created_at DATETIME NOT NULL, "
            "updated_at DATETIME NOT NULL)"
        )
        legacy.execute(
            "INSERT INTO runs VALUES (?, ?, 'from_phase_2.csv', 'auto', 'churn', 'classification', "
            "'RandomForest', 'good', 120, 5, '1.0', '2026-08-01 10:00:00')",
            ("e" * 64, "f" * 64),
        )

    db.init_db()

    # Adopted, not merely stamped: the missing tables exist and the missing
    # columns are back, with the one owner every pre-auth row can only have had.
    tables = set(inspect(db.get_engine()).get_table_names())
    assert {"users", "conversations", "conversation_messages", "llm_usage"} <= tables

    run = db.get_run("e" * 64)
    assert run["filename"] == "from_phase_2.csv"
    assert run["owner_id"] == db.LOCAL_OWNER_ID
    assert db.list_runs(owner_id=db.LOCAL_OWNER_ID)[0]["run_key"] == "e" * 64

    # And the tables it did not have work for writes, not just for reads.
    db.create_job("adopted-job", "Queued", owner_id=db.LOCAL_OWNER_ID)
    assert db.get_job("adopted-job", owner_id=db.LOCAL_OWNER_ID)["state"] == "queued"

    with db.get_engine().connect() as connection:
        assert migrations_module.current_revision(connection) == migrations_module.head_revision()

    db.reset_engine()


def test_adoption_leaves_post_baseline_migrations_their_work(tmp_path, monkeypatch):
    """Adoption builds the *baseline*, not today's models — and it matters.

    The first version of this built the target from live `SQLModel.metadata`,
    which creates every table the current models declare, and then stamped the
    baseline. Two things broke. The loud one: the next `create_table` migration
    failed on a table that was already there, which is how this was found. The
    quiet one is worse — `0002` copies every existing user's key hash into
    `api_keys`, and a database handed that table pre-made gets an empty one, so
    every credential in circulation stops being listable or revocable while
    still authenticating. That is a data migration silently skipped, and it
    would have shipped.
    """
    import hashlib

    from config import settings

    database = tmp_path / "phase6.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{database.as_posix()}")
    db.reset_engine()

    # A Phase 6 database: the baseline exactly, no version row, one account
    # whose key lives on the user row because that is where Phase 6 put it.
    #
    # The baseline's table set is read from migrations_module._schema_by_revision()
    # rather than hardcoded as "everything except api_keys and audit_log" — that
    # hardcoded version is exactly the kind of list the migrations module's own
    # docstring warns goes stale one phase later, and it did: this test passed
    # under Phase 8 and started creating model_versions and predictions at the
    # "baseline" the moment Phase 9 added them, colliding with migration 0003.
    engine = db.get_engine()
    baseline_table_names = dict(migrations_module._schema_by_revision())[migrations_module.BASELINE_REVISION]
    baseline_tables = [table for table in SQLModel.metadata.sorted_tables if table.name in baseline_table_names]
    SQLModel.metadata.create_all(engine, tables=baseline_tables)
    key_hash = hashlib.sha256(b"a-key-issued-before-phase-8").hexdigest()
    with sqlite3.connect(database) as legacy:
        legacy.execute(
            "INSERT INTO users (user_id, email, api_key_hash, is_admin, is_active, created_at, last_seen_at) "
            "VALUES ('u-legacy', 'before@example.com', ?, 0, 1, '2026-08-01 10:00:00', '2026-08-02 11:00:00')",
            (key_hash,),
        )

    db.init_db()

    with db.get_engine().connect() as connection:
        assert migrations_module.current_revision(connection) == migrations_module.head_revision()

    # The migration ran, rather than finding its table already made for it.
    keys = db.list_api_keys("u-legacy")
    assert len(keys) == 1, "the pre-Phase-8 credential was not carried into the key table"
    assert keys[0]["active"] is True
    assert db.get_api_key_by_hash(key_hash)["user_id"] == "u-legacy"

    db.reset_engine()


def test_a_database_already_holding_a_later_revision_is_stamped_there(tmp_path, monkeypatch):
    """Pre-Alembic does not mean pre-everything.

    A database built by the old `create_all` startup has whatever tables the
    models declared on the day it was made, which for anything built recently
    includes revisions after the baseline. Stamping it at the baseline would
    replay those migrations over tables that are already there.
    """
    from config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{(tmp_path / 'recent.db').as_posix()}")
    db.reset_engine()

    # Every current table, no version row: the shape `create_all` produces today.
    SQLModel.metadata.create_all(db.get_engine())
    with db.get_engine().connect() as connection:
        assert migrations_module.current_revision(connection) is None

    db.init_db()

    with db.get_engine().connect() as connection:
        assert migrations_module.current_revision(connection) == migrations_module.head_revision()

    db.reset_engine()


def test_an_empty_database_gets_the_whole_chain(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}")
    db.reset_engine()

    db.init_db()

    tables = set(inspect(db.get_engine()).get_table_names())
    assert {"runs", "jobs", "users", "conversations", "conversation_messages", "llm_usage"} <= tables
    assert "alembic_version" in tables

    db.reset_engine()


def test_the_migrations_still_describe_the_models(isolated_storage):
    """The guard that keeps the next schema change from arriving without a script.

    `_add_missing_columns` used to paper over exactly this drift, and only for
    added columns — a renamed one would have left the database quietly wrong.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    with db.get_engine().connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        diff = compare_metadata(context, SQLModel.metadata)

    assert diff == [], (
        "The models and backend/migrations/versions/ have drifted apart. Run "
        "`cd backend && alembic revision --autogenerate -m '...'`, read what it wrote, and commit it."
    )


def test_the_baseline_can_be_rolled_back(tmp_path, monkeypatch):
    """A migration with no working downgrade is a deploy with no way back."""
    from alembic import command

    from config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{(tmp_path / 'roundtrip.db').as_posix()}")
    db.reset_engine()
    db.init_db()

    engine = db.get_engine()
    with engine.begin() as connection:
        command.downgrade(migrations_module.alembic_config(connection), "base")
        assert "runs" not in set(inspect(connection).get_table_names())
        command.upgrade(migrations_module.alembic_config(connection), "head")
        assert "runs" in set(inspect(connection).get_table_names())

    db.reset_engine()


def test_readyz_refuses_traffic_when_the_schema_is_behind_the_code(client):
    """Wrong-schema is a 503, not a surprise 500 on whichever request needs the column.

    Every other dependency degrades to something that still serves. This one
    cannot: code that expects a column the database does not have fails at the
    first request that touches it, in a way that looks like a bug in the route.
    """
    assert client.get("/readyz").json()["checks"]["schema"]["ok"] is True

    with db.get_engine().begin() as connection:
        connection.execute(text("DELETE FROM alembic_version"))

    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["schema"]["ok"] is False


# --- Postgres ----------------------------------------------------------------


def test_a_provider_style_url_is_rewritten_to_the_pinned_driver():
    """`postgres://` is what hosting providers hand out and what SQLAlchemy refuses."""
    assert db.normalise_database_url("postgres://u:p@host:5432/analyst").startswith("postgresql+psycopg://")
    # Bare postgresql:// would select psycopg2, which is not what is pinned.
    assert db.normalise_database_url("postgresql://u:p@host:5432/analyst") == (
        "postgresql+psycopg://u:p@host:5432/analyst"
    )
    # Already explicit, and SQLite, are both left exactly as they are.
    assert db.normalise_database_url("postgresql+psycopg://u@h/a") == "postgresql+psycopg://u@h/a"
    assert db.normalise_database_url("sqlite:///./models/analyst.db") == "sqlite:///./models/analyst.db"


def test_pooling_is_configured_for_servers_and_skipped_for_sqlite(monkeypatch):
    """SQLite gets one file handle; a pool size there would be a fiction.

    create_engine does not connect, so this checks the configuration without
    needing a Postgres to point it at.
    """
    from config import settings

    monkeypatch.setattr(settings, "database_url", "postgresql+psycopg://u:p@localhost:5432/analyst")
    monkeypatch.setattr(settings, "db_pool_size", 7)
    monkeypatch.setattr(settings, "db_max_overflow", 3)

    engine = db._create_engine()
    assert engine.pool.size() == 7
    assert engine.dialect.driver == "psycopg"
    engine.dispose()

    monkeypatch.setattr(settings, "database_url", "sqlite://")
    assert not hasattr(db._create_engine().pool, "size") or db.is_sqlite()


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against a real server")
def test_the_suite_is_running_against_the_configured_server(isolated_storage, client, fixture_csv):
    """A marker test, so a Postgres CI run cannot silently be a SQLite one.

    Everything else in the suite is backend-agnostic by construction; this one
    asserts the backend actually changed, and that a full training run survives
    the round trip through it.
    """
    assert not db.is_sqlite()

    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    assert job["state"] == "completed"
    assert db.get_run(job["result"]["run_key"]) is not None


# --- artifact portability ----------------------------------------------------


def _sign_in_another_process(bundle_path: Path, key: str) -> None:
    """Write and sign a bundle in a genuinely separate interpreter."""
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from utils.artifacts import dump_bundle
        dump_bundle({{"phase": 7}}, Path(r"{bundle_path}"))
        """
    )
    environment = {
        **os.environ,
        "ARTIFACT_SIGNING_KEY": key,
        "PYTHONPATH": str(BACKEND_DIR),
        # Nothing here should touch a real key or a real model directory.
        "GEMINI_API_KEY": "",
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=BACKEND_DIR, env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_a_bundle_signed_by_another_process_verifies_here(tmp_path, monkeypatch):
    """The assertion the worker/replica split depends on.

    Signing and verifying in one process proves nothing about two: the same
    in-memory key is used for both halves. This writes the bundle in a separate
    interpreter, which is what the worker actually is.
    """
    from config import settings

    key = "a-shared-key-set-on-every-process"
    monkeypatch.setattr(settings, "artifact_signing_key", key)

    bundle_path = tmp_path / "cross_process_model.pkl"
    _sign_in_another_process(bundle_path, key)

    assert artifacts_module.signature_path(bundle_path).exists()
    assert artifacts_module.load_bundle(bundle_path) == {"phase": 7}


def test_a_bundle_signed_with_a_different_key_is_refused(tmp_path, monkeypatch):
    """The failure a generated-per-machine key produces, and its misleading message.

    This is why an unset ARTIFACT_SIGNING_KEY is reported at startup: the
    symptom is identical to tampering, so nobody looks at the configuration.
    """
    from config import settings

    bundle_path = tmp_path / "other_machine_model.pkl"
    _sign_in_another_process(bundle_path, "the-key-the-worker-generated-for-itself")

    monkeypatch.setattr(settings, "artifact_signing_key", "the-key-the-api-generated-for-itself")
    with pytest.raises(ArtifactError, match="integrity check"):
        artifacts_module.load_bundle(bundle_path)


class _OutOfProcessQueue(queue_module.JobQueue):
    """A worker somewhere else, without needing a Redis to prove it."""

    name = "rq"
    out_of_process = True

    def enqueue(self, func, *args, job_id=None, **kwargs):
        return job_id

    def healthy(self) -> bool:
        return True


def test_an_unset_signing_key_is_only_a_problem_once_something_is_shared(monkeypatch, isolated_storage):
    from config import settings

    monkeypatch.setattr(settings, "artifact_signing_key", None)
    # Pinned, not inherited: this half of the test is *about* the single-machine
    # install, and under TEST_DATABASE_URL the suite runs on a server database —
    # which signing_status() correctly counts as something shared, making the
    # assertion below false for a reason that has nothing to do with the key.
    monkeypatch.setattr(settings, "database_url", "sqlite://")

    # One process, one disk, one SQLite file: a generated key is correct here
    # and nothing should nag about it.
    status = artifacts_module.signing_status()
    assert status["signing_key"] == "generated"
    assert status["ok"] is True

    queue_module.set_queue(_OutOfProcessQueue())

    status = artifacts_module.signing_status()
    assert status["ok"] is False
    assert "ARTIFACT_SIGNING_KEY" in status["detail"]
    assert "another process" in status["detail"]


def test_readyz_reports_the_signing_key_without_refusing_traffic(client, monkeypatch, tmp_path):
    """Reported, not fatal: the machine still serves everything it trained itself."""
    from config import settings

    monkeypatch.setattr(settings, "artifact_signing_key", None)
    storage_module.set_storage(_DirectoryStorage(tmp_path / "bucket"))

    body = client.get("/readyz").json()

    assert body["status"] == "ready"
    assert body["checks"]["artifacts"]["ok"] is False
    assert body["checks"]["artifacts"]["signing_key"] == "generated"
    assert "object storage" in body["checks"]["artifacts"]["detail"]


# --- a replica that never trained the model ----------------------------------


def test_a_replica_with_an_empty_disk_serves_a_prediction(client, fixture_csv, isolated_storage, tmp_path):
    """The only assertion that proves the backend is genuinely shared.

    A second API process differs from the first in exactly two ways: it has an
    empty disk and it never ran the training job. Both are reproduced here —
    every local artifact is deleted, and the only remaining copy is in the
    stand-in bucket. If anything the replica needs were written outside the
    published artifact set, this is where it would show up.
    """
    bucket = _DirectoryStorage(tmp_path / "bucket")
    storage_module.set_storage(bucket)

    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    assert job["state"] == "completed"
    run_key = job["result"]["run_key"]

    # What the trainer published — including the signature, without which the
    # replica would refuse to load the model it just downloaded.
    assert sorted(path.name for path in (tmp_path / "bucket").iterdir()) == sorted(
        [f"{run_key}.json", f"{run_key}_data.csv", f"{run_key}_model.pkl", f"{run_key}_model.pkl.sig"]
    )

    # The replica: same bucket, same database, nothing on disk.
    for _, path in storage_module.run_artifact_paths(run_key):
        path.unlink()
    assert not any(path.exists() for _, path in storage_module.run_artifact_paths(run_key))

    row = dict.fromkeys(job["result"]["features"], 1)
    response = client.post(f"/api/predict/{run_key}/row", json={"row": row})

    assert response.status_code == 200, response.text
    assert response.json()["predictions"], "the replica answered without a prediction in it"
    # And the run itself opens, not just the model behind it.
    assert client.get(f"/api/runs/{run_key}").status_code == 200


def test_a_replica_serves_a_promoted_challenger_it_never_trained(
    client, fixture_csv, isolated_storage, tmp_path, monkeypatch
):
    """Phase 9's champion/challenger versioning has to survive Phase 7's
    replica scenario too: a promoted challenger is written under
    `registry.suffix_for(version)`, a name the default artifact suffix list
    (`ARTIFACT_SUFFIXES`) has never heard of. `_load_bundle` has to resolve
    which version is champion *before* asking object storage to fetch it, or
    a replica with an empty disk asks for v1's file, finds v2 registered,
    and 503s permanently instead of "briefly, until it syncs".
    """
    import registry

    bucket = _DirectoryStorage(tmp_path / "bucket")
    storage_module.set_storage(bucket)

    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    assert job["state"] == "completed"
    run_key = job["result"]["run_key"]

    monkeypatch.setattr(
        registry,
        "better",
        lambda *args, **kwargs: {
            "promote": True,
            "metric": "f1_macro",
            "challenger": 0.9,
            "champion": 0.5,
            "reason": "forced promotion for the test",
        },
    )
    with fixture_csv(CLASSIFICATION).open("rb") as handle:
        retrain_response = client.post(
            f"/api/runs/{run_key}/retrain", files={"file": ("newer.csv", handle, "text/csv")}
        )
    assert retrain_response.status_code == 200, retrain_response.text
    challenger = retrain_response.json()["challenger"]
    assert challenger["version"] == 2

    # What the champion's own suffix is now — the file the replica has to be
    # able to find without ever having trained it.
    v2_suffix = registry.suffix_for(2)
    assert f"{run_key}{v2_suffix}" in {key for key, _ in storage_module.run_artifact_paths(run_key, [v2_suffix])}
    assert (tmp_path / "bucket" / f"{run_key}{v2_suffix}").exists(), "the promoted bundle never reached the bucket"

    # The replica: same bucket, same database, nothing on disk — including v2.
    for _, path in storage_module.run_artifact_paths(run_key, [v2_suffix]):
        if path.exists():
            path.unlink()

    row = dict.fromkeys(job["result"]["features"], 1)
    response = client.post(f"/api/predict/{run_key}/row", json={"row": row})

    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == challenger["version_id"]


@pytest.mark.skipif(
    not os.getenv("MINIO_TEST_ENDPOINT"),
    reason="set MINIO_TEST_ENDPOINT (docker compose --profile storage up) to run against a real bucket",
)
def test_object_storage_works_against_a_real_s3_api(client, fixture_csv, isolated_storage):
    """The same round trip against MinIO, because boto3 is where the surprises are.

    A stand-in cannot fail on a missing bucket, a signature-version mismatch, or
    an endpoint that needs path-style addressing. This one can.
    """
    from config import settings

    bucket = os.getenv("MINIO_TEST_BUCKET", "analyst-artifacts")
    settings_patch = {
        "storage_backend": "s3",
        "s3_bucket": bucket,
        "s3_endpoint_url": os.environ["MINIO_TEST_ENDPOINT"],
        "s3_prefix": "pytest",
    }
    for name, value in settings_patch.items():
        setattr(settings, name, value)
    storage_module.reset_storage()

    storage = storage_module.get_storage()
    assert storage.name == "s3", "S3 storage fell back to local — check the bucket and credentials"

    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    run_key = job["result"]["run_key"]

    try:
        for key, path in storage_module.run_artifact_paths(run_key):
            assert storage.exists(key), f"{key} was never published"
            path.unlink()

        row = dict.fromkeys(job["result"]["features"], 1)
        assert client.post(f"/api/predict/{run_key}/row", json={"row": row}).status_code == 200
    finally:
        storage_module.purge_run(run_key)
        storage_module.reset_storage()


# --- what the drills read ----------------------------------------------------


def test_readyz_reports_everything_the_runbook_checks(client):
    """docs/production.md tells an operator to read these keys; keep them real."""
    body = client.get("/readyz").json()

    assert body["status"] == "ready"
    for key in ("database", "schema", "queue", "storage", "artifacts", "llm"):
        assert key in body["checks"], f"/readyz lost the {key} check that docs/production.md reads"
    assert body["checks"]["schema"]["revision"] == migrations_module.head_revision()


def test_the_startup_log_never_carries_the_database_password(isolated_storage, caplog, monkeypatch):
    """A Postgres URL is a credential. It used to be logged verbatim at startup."""
    from fastapi.testclient import TestClient

    import main

    with caplog.at_level("INFO"), TestClient(main.app):
        pass

    logged = json.dumps([record.__dict__.get("database") for record in caplog.records], default=str)
    assert "://" not in logged
