"""Data lifecycle: how long uploaded data lives, and how it leaves.

The plan flags this as mattering more than usual here, and it is right —
healthcare-shaped datasets were in scope from the first exhibit. Three things
are needed and none of them existed: an expiry on every artifact, something
that acts on that expiry, and a way for a user to delete everything of theirs
on demand.

Retention is opt-in (`RETENTION_DAYS=0` keeps data forever) because silently
deleting someone's models on an upgrade would be worse than keeping them.
"""

import logging
from datetime import timedelta

import audit
import db
from config import settings
from utils.helpers import now_utc
from utils.storage import purge_run

logger = logging.getLogger(__name__)


def expiry_for_new_run():
    """When a run created now should expire, or None if retention is off."""
    if settings.retention_days <= 0:
        return None
    return now_utc() + timedelta(days=settings.retention_days)


#: The actor a purge is attributed to when nobody asked for it — the one at
#: startup. A destructive action still has to name who did it, and "the policy,
#: on boot" is the honest answer rather than blaming whichever admin restarted
#: the process.
SYSTEM_ACTOR = "system:retention"


def purge_expired(actor=None) -> dict:
    """Delete every run past its expiry, with its artifacts and conversations.

    Run at startup and exposed as an admin endpoint. Deliberately not a
    background timer: a scheduler inside the web process is one more thing that
    dies with it, and an operator with a cron entry is easier to reason about.

    `actor` is the principal who asked, or None for the startup pass. The audit
    entry is written here rather than at the call sites because there are two of
    them and only one used to record anything — which left the purge that runs
    unattended, on every boot, as the single destructive action in the
    application with no record that it happened.
    """
    expired = db.expired_runs()
    if not expired:
        # No entry: nothing was destroyed. A row per restart saying "deleted
        # nothing" is how an audit log becomes something nobody reads.
        return {"runs_purged": 0, "artifacts_removed": 0, "conversations_removed": 0}

    artifacts = conversations = 0
    for entry in expired:
        run_key = entry["run_key"]
        artifacts += purge_run(run_key)
        conversations += db.delete_conversations_for_run(run_key)
        db.delete_run(run_key)
        logger.info("Purged expired run", extra={"run_key": run_key, "owner": entry["owner_id"]})

    summary = {
        "runs_purged": len(expired),
        "artifacts_removed": artifacts,
        "conversations_removed": conversations,
    }
    logger.info("Retention purge complete", extra=summary)
    audit.record(
        audit.RETENTION_PURGED,
        actor if actor is not None else SYSTEM_ACTOR,
        target_type="retention",
        **summary,
    )
    return summary


def delete_everything_for(owner_id: str) -> dict:
    """Erase one owner's data — the answer to "delete my account's data".

    Runs, artifacts, conversations and messages. Usage rows are kept, because
    they are the billing record and carry no dataset content.
    """
    runs = db.list_runs(limit=10_000, owner_id=owner_id)
    artifacts = conversations = 0
    for run in runs:
        run_key = run["run_key"]
        artifacts += purge_run(run_key)
        conversations += db.delete_conversations_for_run(run_key)
        db.delete_run(run_key, owner_id=owner_id)

    summary = {
        "owner_id": owner_id,
        "runs_deleted": len(runs),
        "artifacts_removed": artifacts,
        "conversations_removed": conversations,
    }
    logger.warning("Owner data erased", extra=summary)
    return summary


def retention_policy() -> dict:
    """The machine-readable version of "where does my data go"."""
    return {
        "retention_days": settings.retention_days,
        "expiry_enabled": settings.retention_days > 0,
        "storage_backend": settings.storage_backend,
        "artifacts_stored": [
            "the dataset snapshot (CSV) used for analysis and reproducibility",
            "the fitted model bundle, HMAC-signed",
            "run metadata: metrics, feature typing, model card",
            "analyst conversations and their tool results",
        ],
        "leaves_this_machine": [
            "your question, the column names and dtypes, and compact text summaries of tool "
            "results are sent to the configured LLM when you use the chat panel or generate a report",
        ],
        "never_leaves_this_machine": [
            "the dataset rows themselves",
            "the fitted model",
            "rendered charts",
        ],
        "deletion": "DELETE /api/runs/{run_key} removes a run's artifacts and conversations; "
        "DELETE /api/account/data removes everything you own.",
    }
