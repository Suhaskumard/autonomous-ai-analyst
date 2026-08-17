"""Job and run persistence.

`UPLOAD_JOBS` was a module-level dict: lost on restart, unbounded, and invisible
to any other uvicorn worker. Jobs now live in SQLite, so state survives a
restart and every worker sees the same rows. The RunRecord table doubles as the
dataset registry the Phase 3 history UI needs.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import JSON, Column, delete, text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from config import settings
from utils.helpers import now_utc


class JobRecord(SQLModel, table=True):
    """One upload/training job and everything the status endpoint returns."""

    __tablename__ = "jobs"

    job_id: str = Field(primary_key=True)
    state: str = Field(default="queued", index=True)
    current_step: str = "Queued"
    progress: int = 0
    status_log: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error: str | None = None
    error_kind: str | None = None
    run_key: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc)

    def to_status(self) -> dict[str, Any]:
        """The shape the frontend polls for."""
        return {
            "job_id": self.job_id,
            "state": self.state,
            "current_step": self.current_step,
            "progress": self.progress,
            "status_log": self.status_log or [],
            "result": self.result,
            "error": self.error,
            "error_kind": self.error_kind,
            "run_key": self.run_key,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RunRecord(SQLModel, table=True):
    """Registry of completed runs — one row per trained configuration."""

    __tablename__ = "runs"

    run_key: str = Field(primary_key=True)
    dataset_hash: str = Field(index=True)
    filename: str | None = None
    mode: str = "auto"
    target: str | None = None
    problem_type: str | None = None
    selected_model: str | None = None
    health_verdict: str | None = None
    row_count: int = 0
    column_count: int = 0
    pipeline_version: str | None = None
    created_at: datetime = Field(default_factory=now_utc, index=True)


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        _engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
        if settings.database_url.startswith("sqlite"):
            # WAL lets the poller read while a background job writes, which is
            # exactly the access pattern here.
            with _engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()
    return _engine


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())


def reset_engine() -> None:
    """Drop the cached engine so a new database_url takes effect (tests)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def session_scope() -> Iterator[Session]:
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --- job helpers ------------------------------------------------------------


def create_job(job_id: str, first_step: str) -> None:
    with session_scope() as session:
        session.add(
            JobRecord(
                job_id=job_id,
                state="queued",
                current_step="Queued",
                progress=0,
                status_log=[{"step": first_step, "progress": 0, "timestamp": now_utc().isoformat()}],
            )
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        return job.to_status() if job else None


def append_step(job_id: str, step: str, progress: int) -> None:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            return
        job.state = "running"
        job.current_step = step
        job.progress = progress
        # Reassigning is required: SQLAlchemy does not track in-place mutation
        # of a JSON column.
        job.status_log = [
            *(job.status_log or []),
            {"step": step, "progress": progress, "timestamp": now_utc().isoformat()},
        ]
        job.updated_at = now_utc()
        session.add(job)


def complete_job(job_id: str, result: dict[str, Any], run_key: str | None) -> None:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            return
        job.state = "completed"
        job.progress = 100
        job.current_step = "Completed"
        job.status_log = [
            *(job.status_log or []),
            {"step": "Completed", "progress": 100, "timestamp": now_utc().isoformat()},
        ]
        job.result = result
        job.run_key = run_key
        job.updated_at = now_utc()
        session.add(job)


def fail_job(job_id: str, error: str, kind: str, step: str) -> None:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            return
        job.state = "failed"
        job.error = error
        job.error_kind = kind
        job.current_step = step
        job.updated_at = now_utc()
        session.add(job)


def recover_interrupted_jobs() -> int:
    """Mark jobs that were mid-flight when the process died.

    Without this a killed job stays "running" forever and the UI polls a job
    nothing is working on.
    """
    with session_scope() as session:
        stale = session.exec(select(JobRecord).where(JobRecord.state.in_(("queued", "running")))).all()
        for job in stale:
            job.state = "failed"
            job.error = "The server restarted while this job was running. Re-upload the dataset to retry."
            job.error_kind = "interrupted"
            job.current_step = "Interrupted by server restart"
            job.updated_at = now_utc()
            session.add(job)
        return len(stale)


def purge_old_jobs(max_age_hours: int = 72) -> int:
    """Keep the table from growing forever."""
    cutoff = now_utc() - timedelta(hours=max_age_hours)
    with session_scope() as session:
        result = session.exec(delete(JobRecord).where(JobRecord.created_at < cutoff))
        return int(getattr(result, "rowcount", 0) or 0)


# --- run registry -----------------------------------------------------------


def upsert_run(**fields: Any) -> None:
    with session_scope() as session:
        existing = session.get(RunRecord, fields["run_key"])
        if existing:
            for key, value in fields.items():
                setattr(existing, key, value)
            session.add(existing)
        else:
            session.add(RunRecord(**fields))


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.exec(select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit)).all()
        return [
            {**row.model_dump(), "created_at": row.created_at.isoformat() if row.created_at else None} for row in rows
        ]


def get_run(run_key: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(RunRecord, run_key)
        if row is None:
            return None
        return {**row.model_dump(), "created_at": row.created_at.isoformat() if row.created_at else None}


def delete_run(run_key: str) -> bool:
    with session_scope() as session:
        row = session.get(RunRecord, run_key)
        if row is None:
            return False
        session.delete(row)
        return True


__all__ = [
    "JobRecord",
    "RunRecord",
    "append_step",
    "complete_job",
    "create_job",
    "delete_run",
    "fail_job",
    "get_engine",
    "get_job",
    "get_run",
    "init_db",
    "list_runs",
    "purge_old_jobs",
    "recover_interrupted_jobs",
    "reset_engine",
    "session_scope",
    "upsert_run",
]
