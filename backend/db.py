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


class ConversationRecord(SQLModel, table=True):
    """One analyst conversation, scoped to a run.

    Server-side rather than client-side on purpose: the client used to post its
    own history back on every message, so anything it dropped, edited, or
    invented became what the model believed had been said.
    """

    __tablename__ = "conversations"

    conversation_id: str = Field(primary_key=True)
    run_key: str = Field(index=True)
    title: str | None = None
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc)


class MessageRecord(SQLModel, table=True):
    """One turn. `steps` holds the tool calls behind an assistant answer."""

    __tablename__ = "conversation_messages"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(index=True)
    role: str = "user"
    content: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: datetime = Field(default_factory=now_utc, index=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "steps": self.steps or [],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
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


def create_conversation(conversation_id: str, run_key: str, title: str | None = None) -> dict[str, Any]:
    with session_scope() as session:
        record = ConversationRecord(conversation_id=conversation_id, run_key=run_key, title=title)
        session.add(record)
        session.flush()
        return {"conversation_id": record.conversation_id, "run_key": record.run_key, "title": record.title}


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(ConversationRecord, conversation_id)
        if row is None:
            return None
        return {
            "conversation_id": row.conversation_id,
            "run_key": row.run_key,
            "title": row.title,
            "total_tokens": row.total_tokens,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }


def list_conversations(run_key: str, limit: int = 20) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.exec(
            select(ConversationRecord)
            .where(ConversationRecord.run_key == run_key)
            .order_by(ConversationRecord.updated_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "conversation_id": row.conversation_id,
                "run_key": row.run_key,
                "title": row.title,
                "total_tokens": row.total_tokens,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]


def list_messages(conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.exec(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.id)
            .limit(limit)
        ).all()
        return [row.to_dict() for row in rows]


def append_message(
    conversation_id: str,
    role: str,
    content: str,
    steps: list[dict[str, Any]] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, Any]:
    with session_scope() as session:
        record = MessageRecord(
            conversation_id=conversation_id,
            role=role,
            content=content,
            steps=steps or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        session.add(record)

        conversation = session.get(ConversationRecord, conversation_id)
        if conversation is not None:
            conversation.total_tokens += input_tokens + output_tokens
            conversation.updated_at = now_utc()
            # The first thing asked is the most useful label for a session.
            if not conversation.title and role == "user" and content.strip():
                conversation.title = content.strip()[:80]
            session.add(conversation)

        session.flush()
        return record.to_dict()


def count_recent_messages(conversation_id: str, since_minutes: int = 60) -> int:
    """User messages in the recent window — what the rate limit is measured on."""
    cutoff = now_utc() - timedelta(minutes=since_minutes)
    with session_scope() as session:
        rows = session.exec(
            select(MessageRecord).where(
                MessageRecord.conversation_id == conversation_id,
                MessageRecord.role == "user",
                MessageRecord.created_at >= cutoff,
            )
        ).all()
        return len(rows)


def delete_conversations_for_run(run_key: str) -> int:
    """Deleting a dataset takes its conversations with it."""
    with session_scope() as session:
        rows = session.exec(select(ConversationRecord).where(ConversationRecord.run_key == run_key)).all()
        removed = 0
        for row in rows:
            session.exec(delete(MessageRecord).where(MessageRecord.conversation_id == row.conversation_id))
            session.delete(row)
            removed += 1
        return removed


def delete_run(run_key: str) -> bool:
    with session_scope() as session:
        row = session.get(RunRecord, run_key)
        if row is None:
            return False
        session.delete(row)
        return True


__all__ = [
    "ConversationRecord",
    "JobRecord",
    "MessageRecord",
    "RunRecord",
    "append_message",
    "append_step",
    "count_recent_messages",
    "create_conversation",
    "delete_conversations_for_run",
    "get_conversation",
    "list_conversations",
    "list_messages",
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
