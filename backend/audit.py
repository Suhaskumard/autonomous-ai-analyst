"""The audit log: who did the irreversible thing, and when.

Phase 6 made every action attributable in principle — each request has a
principal — and recorded none of it. The application logs are not a substitute:
they are unstructured, they rotate, and they are the first thing lost when a
container is replaced. "Whose account deleted that dataset?" needs an answer
that survives a redeploy.

What is recorded is deliberately narrow: privileged and destructive actions
only. Reading a run is not audited, because an audit log that records
everything is a second copy of the access log that nobody reads, and it would
put a row in the database for every poll of a job status endpoint.

Three properties matter and each is a decision rather than an accident:

* **Append-only.** `db` has `append_audit` and `list_audit` and no update or
  delete partner. Nothing in this application edits history.
* **Outside retention.** `lifecycle.purge_expired` deletes runs, artifacts and
  conversations; it does not touch this table. The record that a dataset was
  deleted has to outlive the dataset.
* **No payload.** Actions carry ids and counts, never dataset contents, column
  names, or question text — the log is read by more people than the data is.
"""

import logging

from config import settings
from logging_config import request_id_var

logger = logging.getLogger(__name__)

# --- the vocabulary ---------------------------------------------------------
# Fixed strings rather than free text, so a query for "every deletion" does not
# depend on how a caller phrased it that day.

USER_CREATED = "user.created"
USER_DEACTIVATED = "user.deactivated"
KEY_ISSUED = "key.issued"
KEY_REVOKED = "key.revoked"
RUN_DELETED = "run.deleted"
ACCOUNT_DATA_ERASED = "account.data_erased"
RETENTION_PURGED = "retention.purged"
QUOTA_REFUSED = "quota.refused"
#: Phase 9. Promotion changes what every caller of a run is served from that
#: moment on, which makes it destructive in the way that matters here: the
#: answer someone got yesterday is not the answer they get today, and the
#: reason has to be attributable. A challenger that was *not* promoted is
#: recorded too — "we tried and kept the old one" is the more interesting row
#: six months later, and a log that only holds successes cannot show restraint.
MODEL_PROMOTED = "model.promoted"
MODEL_CHALLENGED = "model.challenged"

#: Actions a non-admin may see about themselves. Everything else is admin-only,
#: because "which admin deactivated whom" is not an account holder's business.
SELF_VISIBLE = frozenset(
    {
        KEY_ISSUED,
        KEY_REVOKED,
        RUN_DELETED,
        ACCOUNT_DATA_ERASED,
        QUOTA_REFUSED,
        MODEL_PROMOTED,
        MODEL_CHALLENGED,
    }
)


def record(
    action: str,
    actor,
    target_type: str | None = None,
    target_id: str | None = None,
    **detail,
) -> None:
    """Append one entry. `actor` is a Principal, or an id for system actions.

    Never raises. An audit write that fails must not turn a successful deletion
    into a 500 — the action already happened, and losing the record of it is
    the lesser of the two bad outcomes. It is logged loudly instead.
    """
    import db

    actor_id = getattr(actor, "user_id", None) or str(actor)
    actor_email = getattr(actor, "email", None)

    try:
        db.append_audit(
            action=action,
            actor_id=actor_id,
            actor_email=actor_email,
            target_type=target_type,
            target_id=target_id,
            detail=detail,
            request_id=request_id_var.get(),
        )
    except Exception:
        logger.exception("Audit write failed", extra={"action": action, "actor": actor_id})
        return

    logger.info(
        "audit %s",
        action,
        extra={"action": action, "actor": actor_id, "target": target_id},
    )


def enabled() -> bool:
    """Whether the log is meaningful.

    With authentication off every action is attributed to the same implicit
    local owner, so the log records a sequence of events but not *who* — which
    is honest, and worth saying out loud rather than implying otherwise.
    """
    return settings.auth_enabled
