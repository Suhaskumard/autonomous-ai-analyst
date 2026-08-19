"""Scheduled reports: the Phase 5 generator on a cron, delivered by email — Phase 10.

Nothing here runs a scheduler in-process. Phase 6's own lifespan comment
already settled that question for retention — "an in-process scheduler is one
more thing that dies with the process, and an operator who wants it hourly can
curl the admin endpoint from cron" — and a report schedule follows the same
shape: `next_run_at` is a due date `db.due_report_schedules` makes visible,
and `POST /api/report-schedules/run-due` is what an operator's cron entry
calls to act on it.

What that endpoint does is hand each due schedule to Phase 6's queue rather
than generate reports inline in the request — "the queue makes this nearly
free" is the plan's own reasoning, and it is also why one slow LLM call cannot
make a cron tick time out. `process_due_schedule` is the function actually
enqueued; it has to be importable by reference for RQ, so it stays at module
level rather than as a closure.

Off by default (`SCHEDULED_REPORTS_ENABLED`) and refuses outright without
`SMTP_HOST` configured — a schedule that could never deliver would fail
silently every time it ran otherwise.
"""

import logging
import secrets
from datetime import datetime, timedelta

from fastapi import HTTPException

import audit
import db
from config import settings
from utils.helpers import now_utc

logger = logging.getLogger(__name__)

INTERVALS = {"daily": timedelta(days=1), "weekly": timedelta(days=7)}


def _require_enabled() -> None:
    if not settings.scheduled_reports_enabled:
        raise HTTPException(status_code=403, detail="Scheduled reports are disabled on this server.")
    if not settings.smtp_host:
        raise HTTPException(status_code=503, detail="SCHEDULED_REPORTS_ENABLED is set but SMTP_HOST is not configured.")


def next_run_after(interval: str, moment: datetime | None = None) -> datetime:
    return (moment or now_utc()) + INTERVALS[interval]


def _parse_recipients(raw) -> list[str]:
    if isinstance(raw, str):
        candidates = [item.strip() for item in raw.split(",")]
    else:
        candidates = [str(item).strip() for item in (raw or [])]
    recipients = [item for item in candidates if item]
    if not recipients:
        raise HTTPException(status_code=400, detail="At least one recipient email address is required.")
    for address in recipients:
        if "@" not in address or address.startswith("@") or address.endswith("@"):
            raise HTTPException(status_code=400, detail=f"'{address}' is not a valid email address.")
    return recipients


def create_schedule(run_key: str, principal, recipients, interval: str) -> dict:
    _require_enabled()
    if interval not in INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval must be one of {sorted(INTERVALS)}.")
    parsed = _parse_recipients(recipients)

    schedule_id = f"sched_{run_key[:12]}_{secrets.token_hex(4)}"
    record = db.add_report_schedule(
        schedule_id=schedule_id,
        run_key=run_key,
        owner_id=principal.user_id,
        recipients=",".join(parsed),
        interval=interval,
        next_run_at=next_run_after(interval),
    )
    logger.info("Report schedule created", extra={"run_key": run_key, "interval": interval})
    audit.record(audit.REPORT_SCHEDULE_CREATED, principal, target_type="run", target_id=run_key, interval=interval)
    return record


def list_schedules(run_key: str, principal) -> list[dict]:
    from auth import owner_scope

    return db.list_report_schedules(run_key=run_key, owner_id=owner_scope(principal))


def delete_schedule(schedule_id: str, principal) -> None:
    from auth import owner_scope

    ok = db.delete_report_schedule(schedule_id, owner_id=owner_scope(principal))
    if not ok:
        raise HTTPException(status_code=404, detail="No report schedule found for that id.")
    logger.info("Report schedule deleted", extra={"schedule_id": schedule_id})
    audit.record(audit.REPORT_SCHEDULE_DELETED, principal, target_type="report_schedule", target_id=schedule_id)


def _send_email(to_addresses: list[str], subject: str, body_text: str) -> None:
    import smtplib
    from email.message import EmailMessage

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from_address or settings.smtp_username or "no-reply@localhost"
    message["To"] = ", ".join(to_addresses)
    message.set_content(body_text)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        if settings.smtp_use_tls:
            server.starttls()
        if settings.smtp_username and settings.smtp_password:
            server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(message)


def process_due_schedule(schedule_id: str) -> None:
    """Build one report and email it. Enqueued by run_due(), run by a worker.

    Every failure mode — quota, missing artifacts, SMTP down — ends the same
    way: `next_run_at` still advances. A schedule that fails once and then
    retries every few seconds forever is worse than one that tries again on
    its normal cadence and is visible as failing via `last_status`.
    """
    from analyst import report as report_module
    from analyst.providers import get_provider
    from auth import Principal
    from observability import record_llm_usage
    from routes.chat import _load_metadata, _load_snapshot
    from utils.helpers import METADATA_DIR
    from utils.security import artifact_path

    schedule = db.get_report_schedule(schedule_id)
    if schedule is None or not schedule["active"]:
        return

    run_key = schedule["run_key"]
    interval = schedule["interval"]
    status = "sent"
    try:
        artifact_path(METADATA_DIR, run_key, ".json")  # validates the key format
        metadata = _load_metadata(run_key)
        snapshot_path, df = _load_snapshot(run_key)
        if df is None:
            raise RuntimeError("No data snapshot for this run.")

        # Read from db directly rather than trusting `schedule["owner_id"]`
        # was ever set correctly: run_owner is the same source of truth every
        # other owner-scoped read in this application uses.
        owner_id = db.run_owner(run_key) or db.LOCAL_OWNER_ID
        principal = Principal(user_id=owner_id, email="", is_admin=False)

        import limits

        try:
            limits.check_llm_quota(principal)
        except HTTPException:
            status = "skipped_over_quota"
            logger.warning("Scheduled report skipped: over quota", extra={"schedule_id": schedule_id})
        else:
            provider = get_provider()
            report = report_module.build_report(df, metadata, snapshot_path, run_key, provider=provider)
            markdown = report_module.render_markdown(report)

            usage = report.pop("usage", None) or {}
            if provider is not None and (usage.get("input_tokens") or usage.get("output_tokens")):
                record_llm_usage(
                    owner_id=owner_id,
                    provider=provider.name,
                    model=provider.model,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    run_key=run_key,
                )

            _send_email(
                schedule["recipients"],
                subject=f"Scheduled report: {metadata.get('target', run_key)} ({run_key[:10]}…)",
                body_text=markdown,
            )
    except Exception:
        status = "failed"
        logger.exception("Scheduled report failed", extra={"schedule_id": schedule_id, "run_key": run_key})
    finally:
        db.record_report_schedule_run(schedule_id, next_run_at=next_run_after(interval), status=status)


def run_due() -> dict:
    """Enqueue every due schedule, up to REPORT_SCHEDULE_MAX_DUE_PER_RUN. What cron calls."""
    from utils.queue import get_queue

    due = db.due_report_schedules()[: settings.report_schedule_max_due_per_run]
    queue = get_queue()
    for schedule in due:
        queue.enqueue(process_due_schedule, schedule["schedule_id"])
    logger.info("Report schedules processed", extra={"due": len(due), "queue": queue.name})
    return {"due": len(due), "queue": queue.name}
