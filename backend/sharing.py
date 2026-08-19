"""Read-only, expiring share links — Phase 10.

Ownership exists (Phase 6) and audit exists (Phase 8), which is what makes a
link like this safe to add: every link names who created it and when, and
revoking it is one call. What it grants is narrow and fixed — the same payload
`GET /runs/{run_key}/result` already returns, nothing more. No chat, no fresh
prediction, no report generation: those cost LLM money or compute per call,
and a link is meant to be handed to someone this application has no way to
bill or rate-limit. The dashboard result is pre-computed at training time, so
serving it to a link is free in the way that matters.

Off by default (`SHARE_LINKS_ENABLED`), the same posture as Phase 6/8's
capabilities: it is a new door an unauthenticated caller can walk through, and
that is an operator's decision to make, not this application's to make for
them.
"""

import logging
import secrets
from datetime import timedelta

from fastapi import HTTPException

import audit
import db
from config import settings
from utils.helpers import now_utc

logger = logging.getLogger(__name__)

TOKEN_BYTES = 24


def _require_enabled() -> None:
    if not settings.share_links_enabled:
        raise HTTPException(status_code=403, detail="Sharing is disabled on this server.")


def create_link(run_key: str, principal, ttl_days: int | None = None) -> dict:
    """Mint a link for a run this principal owns. Returns the token once."""
    _require_enabled()
    requested = ttl_days if ttl_days and ttl_days > 0 else settings.share_link_default_ttl_days
    capped = min(requested, settings.share_link_max_ttl_days)
    expires_at = now_utc() + timedelta(days=capped)

    token = secrets.token_urlsafe(TOKEN_BYTES)
    record = db.add_share_link(token=token, run_key=run_key, owner_id=principal.user_id, expires_at=expires_at)
    logger.info("Share link created", extra={"run_key": run_key, "owner": principal.user_id})
    audit.record(audit.SHARE_LINK_CREATED, principal, target_type="run", target_id=run_key)
    return {**record, "token": token}


def list_links(run_key: str, principal) -> list[dict]:
    from auth import owner_scope

    return db.list_share_links(run_key, owner_id=owner_scope(principal))


def revoke_link(token: str, principal) -> None:
    from auth import owner_scope

    ok = db.revoke_share_link(token, owner_id=owner_scope(principal))
    if not ok:
        # Same 404-not-403 reasoning as `require_owned_run`: a token that
        # exists but belongs to someone else should look identical to one
        # that was never issued.
        raise HTTPException(status_code=404, detail="No share link found for that token.")
    logger.info("Share link revoked", extra={"token_prefix": token[:8]})
    audit.record(audit.SHARE_LINK_REVOKED, principal, target_type="share_link", target_id=token[:8])


def resolve_token(token: str) -> str:
    """The run key a live token grants access to, or 404.

    Deliberately one error for "no such token", "revoked", and "expired" —
    the same reasoning `require_owned_run` gives for owner mismatches: telling
    an attacker which of those is true is a free bit they should not get.
    """
    _require_enabled()
    link = db.get_share_link(token)
    if link is None or not link["active"]:
        raise HTTPException(status_code=404, detail="This share link is invalid or has expired.")
    return link["run_key"]
