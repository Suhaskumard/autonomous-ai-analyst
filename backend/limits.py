"""Per-principal ceilings.

The guards written in Phase 5 are per *conversation*: a message rate and a
token budget, both reset by starting a new conversation. That slows a single
session down and stops nothing — a client in a loop opens a new conversation
each time and has no ceiling at all. It was the right guard for one operator
protecting themselves from a runaway script, and the wrong one for an account
that might be hostile.

These are per account, and they are measured against tables that already exist:
`llm_usage` records every paid call with its estimated cost, and `runs` records
every stored dataset. Nothing new has to be tracked; what was missing was
somebody consulting it before the next call rather than after.

Every limit defaults to 0, meaning off. A single-operator install should not be
rate-limited by itself, and Phase 6's rule holds here too: capabilities that
change how the app behaves are opt-in.

Two things this deliberately does not do:

* **It is not a distributed rate limiter.** The window is computed from the
  database on each request, so two API processes share one count without
  needing Redis, at the cost of a query per guarded call. That trade is right
  while the guarded endpoints are the ones that take seconds anyway.
* **It does not refund.** The spend ceiling is checked before a call and the
  cost recorded after, so a single call may cross the line rather than being
  stopped exactly at it. Enforcing to the cent would mean pricing a response
  before generating it, which nobody can do.
"""

import logging
from datetime import timedelta

from fastapi import HTTPException

import audit
import db
from config import settings
from utils.helpers import now_utc

logger = logging.getLogger(__name__)


def _refuse(principal, reason: str, detail: str, status_code: int = 429, **context) -> None:
    """Refuse an action, and record that it was refused.

    Quota refusals are audited because a client hammering a ceiling is exactly
    the pattern an operator wants to find later, and because "the system was
    slow that afternoon" is much easier to explain with these rows in hand.
    """
    audit.record(audit.QUOTA_REFUSED, principal, target_type="quota", target_id=reason, **context)
    logger.warning("Quota refused", extra={"principal": principal.user_id, "quota": reason, **context})
    raise HTTPException(status_code=status_code, detail=detail)


def check_llm_quota(principal) -> None:
    """Guard a paid endpoint. Raises 429 when this account is over its ceiling."""
    if not settings.auth_enabled:
        # One implicit owner: a per-account limit would be a per-machine limit
        # wearing a disguise, and the per-conversation guards already cover the
        # runaway-script case this install actually has.
        return

    owner_id = principal.user_id

    if settings.principal_max_llm_calls_per_hour > 0:
        since = now_utc() - timedelta(hours=1)
        calls = db.count_llm_calls_since(owner_id, since)
        if calls >= settings.principal_max_llm_calls_per_hour:
            _refuse(
                principal,
                "llm_calls_per_hour",
                (
                    f"This account has used its hourly limit of "
                    f"{settings.principal_max_llm_calls_per_hour} model calls. Try again later."
                ),
                calls=calls,
                limit=settings.principal_max_llm_calls_per_hour,
            )

    if settings.principal_daily_spend_limit_usd > 0:
        since = now_utc() - timedelta(days=1)
        spent = db.spend_since(owner_id, since)
        if spent >= settings.principal_daily_spend_limit_usd:
            _refuse(
                principal,
                "daily_spend",
                (
                    f"This account has reached its estimated daily spend limit of "
                    f"${settings.principal_daily_spend_limit_usd:.2f}. It resets 24 hours after "
                    "the earliest call in the window."
                ),
                spent_usd=round(spent, 6),
                limit_usd=settings.principal_daily_spend_limit_usd,
            )


def check_storage_quota(principal) -> None:
    """Guard an upload. Raises 413 when this account is at its storage ceiling.

    Checked before the upload is accepted rather than after training, because
    the point is to refuse the bytes, not to discover afterwards that the disk
    is full. The measurement is the account's *stored* runs — the request's own
    size is already capped by MAX_UPLOAD_MB.

    Which means uploads still training do not count toward it yet: an account
    at its limit minus one can start several at once and every one of them is
    checked against a registry none of them has reached. Closing that here
    would mean counting in-flight jobs as stored runs, and then a failed
    training run would occupy quota until somebody noticed. It is
    `check_job_concurrency` that bounds the burst, which is why the two limits
    are documented as a pair rather than as alternatives.
    """
    if not settings.auth_enabled:
        return

    owner_id = principal.user_id

    if settings.principal_max_runs > 0:
        runs = db.count_runs(owner_id)
        if runs >= settings.principal_max_runs:
            _refuse(
                principal,
                "max_runs",
                (
                    f"This account is at its limit of {settings.principal_max_runs} stored datasets. "
                    "Delete one to make room."
                ),
                status_code=413,
                runs=runs,
                limit=settings.principal_max_runs,
            )

    if settings.principal_storage_quota_mb > 0:
        used = owner_storage_bytes(owner_id)
        ceiling = settings.principal_storage_quota_mb * 1024 * 1024
        if used >= ceiling:
            _refuse(
                principal,
                "storage_quota",
                (
                    f"This account is using {used / 1024 / 1024:.1f} MB of its "
                    f"{settings.principal_storage_quota_mb} MB storage quota. Delete a dataset to "
                    "make room."
                ),
                status_code=413,
                used_mb=round(used / 1024 / 1024, 1),
                limit_mb=settings.principal_storage_quota_mb,
            )


def check_job_concurrency(principal) -> None:
    """Guard the workers. Raises 429 when this account is already holding its share.

    The third exhaustible resource, and the one neither of the other two
    catches: an account can sit well under its spend ceiling and its disk quota
    and still queue fifty training runs, which occupies every worker and makes
    the application unavailable for everybody else. It costs the attacker one
    small CSV per run.

    A ceiling rather than a queue-per-account, which would be the better answer
    and is a much larger change: fair scheduling means the queue has to know
    about principals, and RQ's does not. Refusing the fifty-first upload is the
    version that fits behind one function call, and it makes the failure the
    abusive account's rather than everyone's.
    """
    if not settings.auth_enabled:
        return
    if settings.principal_max_concurrent_jobs <= 0:
        return

    active = db.count_active_jobs(principal.user_id)
    if active >= settings.principal_max_concurrent_jobs:
        _refuse(
            principal,
            "concurrent_jobs",
            (
                f"This account already has {active} datasets queued or training, which is its "
                f"limit of {settings.principal_max_concurrent_jobs}. Wait for one to finish."
            ),
            active=active,
            limit=settings.principal_max_concurrent_jobs,
        )


def owner_storage_bytes(owner_id: str) -> int:
    """How much disk this account's stored runs occupy.

    Summed from the artifact files each run owns, so it counts what is actually
    on the disk rather than what the dataset claimed to be.
    """
    from utils.storage import run_artifact_paths

    total = 0
    for run in db.list_runs(limit=10_000, owner_id=owner_id):
        for _, path in run_artifact_paths(run["run_key"]):
            try:
                total += path.stat().st_size
            except OSError:
                # Purged, or held only in object storage on another machine.
                continue
    return total


def quota_status(principal) -> dict:
    """What this account has used and what it is allowed — for /api/usage."""
    if not settings.auth_enabled:
        return {"enforced": False, "reason": "Authentication is off; there is one implicit owner."}

    owner_id = principal.user_id
    hour_ago = now_utc() - timedelta(hours=1)
    day_ago = now_utc() - timedelta(days=1)

    return {
        "enforced": True,
        "llm_calls_last_hour": db.count_llm_calls_since(owner_id, hour_ago),
        "llm_calls_per_hour_limit": settings.principal_max_llm_calls_per_hour or None,
        "estimated_spend_last_day_usd": round(db.spend_since(owner_id, day_ago), 6),
        "daily_spend_limit_usd": settings.principal_daily_spend_limit_usd or None,
        "runs_stored": db.count_runs(owner_id),
        "max_runs": settings.principal_max_runs or None,
        "jobs_active": db.count_active_jobs(owner_id),
        "max_concurrent_jobs": settings.principal_max_concurrent_jobs or None,
        "storage_used_mb": round(owner_storage_bytes(owner_id) / 1024 / 1024, 1),
        "storage_quota_mb": settings.principal_storage_quota_mb or None,
        # Restated here because the number above is an estimate from published
        # list prices, not an invoice, and a limit built on it inherits that.
        "spend_is_estimated": True,
    }
