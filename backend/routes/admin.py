"""Accounts, usage, retention, and the privacy statement.

Registration is gated by a bootstrap token rather than open: an unauthenticated
"create an account" endpoint on an app with no email verification is an open
door. The first account is created with the token from `AUTH_BOOTSTRAP_TOKEN`
and is an admin; after that, admins create the rest.
"""

import logging
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

import audit
import db
import lifecycle
import limits
from auth import Principal, create_user, current_principal, issue_key, owner_scope, require_admin
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


@router.post("/auth/register")
def register(
    request: RegisterRequest,
    x_bootstrap_token: str | None = Header(default=None),
):
    """Create the first account, using the configured bootstrap token.

    Closed entirely when no token is configured, and closed again once an
    account exists — after that, `POST /api/admin/users` is the way in.
    """
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="Authentication is disabled; there are no accounts to create.")

    configured = (settings.auth_bootstrap_token or "").strip()
    if not configured:
        raise HTTPException(
            status_code=403,
            detail="Registration is closed. Set AUTH_BOOTSTRAP_TOKEN to create the first account.",
        )
    if not x_bootstrap_token or not secrets.compare_digest(x_bootstrap_token, configured):
        raise HTTPException(status_code=403, detail="Invalid bootstrap token.")
    if db.count_users() > 0:
        raise HTTPException(
            status_code=409,
            detail="An account already exists. Ask an admin to create further accounts.",
        )

    user, api_key = create_user(request.email, is_admin=True)
    # The bootstrap account has no creator to attribute this to, so it is
    # recorded against itself — the first row in the log is the account that
    # every later row will be attributed to.
    audit.record(
        audit.USER_CREATED,
        Principal(user_id=user["user_id"], email=user["email"], is_admin=True),
        target_type="user",
        target_id=user["user_id"],
        via="bootstrap",
        is_admin=True,
    )
    return {
        **user,
        "api_key": api_key,
        # Said plainly because it is true and the consequence is losing access.
        "notice": "This API key is shown once and cannot be recovered. Store it now.",
    }


@router.get("/auth/me")
def whoami(principal: Principal = Depends(current_principal)):
    return {
        "user_id": principal.user_id,
        "email": principal.email,
        "is_admin": principal.is_admin,
        "auth_enabled": settings.auth_enabled,
    }


@router.post("/admin/users")
def add_user(request: RegisterRequest, principal: Principal = Depends(require_admin)):
    user, api_key = create_user(request.email)
    audit.record(audit.USER_CREATED, principal, target_type="user", target_id=user["user_id"], email=user["email"])
    return {**user, "api_key": api_key, "notice": "This API key is shown once and cannot be recovered."}


@router.get("/admin/users")
def all_users(principal: Principal = Depends(require_admin)):
    return {"users": db.list_users()}


@router.post("/admin/users/{user_id}/deactivate")
def deactivate_user(user_id: str, principal: Principal = Depends(require_admin)):
    if user_id == principal.user_id:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")
    user = db.set_user_active(user_id, is_active=False)
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    audit.record(audit.USER_DEACTIVATED, principal, target_type="user", target_id=user_id, email=user["email"])
    return user


# --- key lifecycle ----------------------------------------------------------


class IssueKeyRequest(BaseModel):
    label: str | None = Field(default=None, max_length=60)
    # None uses API_KEY_DEFAULT_TTL_DAYS; 0 means this key does not expire.
    ttl_days: int | None = Field(default=None, ge=0, le=3650)


@router.get("/account/keys")
def my_keys(principal: Principal = Depends(current_principal)):
    """Every key on this account, and never anything that could authenticate.

    The hash is not returned either: it is not usable as a credential, but it
    is a verifier, and a leaked list of them turns an offline guess into a
    check. There is nothing here a caller needs it for.
    """
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="Authentication is disabled; there are no keys to manage.")
    return {"keys": db.list_api_keys(principal.user_id), "max_active": settings.max_active_api_keys}


@router.post("/account/keys")
def issue_my_key(request: IssueKeyRequest, principal: Principal = Depends(current_principal)):
    """Issue an additional key, so rotation does not need an outage.

    Deliberately available to the account itself rather than admin-only: the
    person who has to update every caller is the one who should be able to mint
    the replacement, and requiring an admin for a routine rotation is how
    rotations stop happening.
    """
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="Authentication is disabled; there are no keys to manage.")

    record, api_key = issue_key(principal.user_id, label=request.label, ttl_days=request.ttl_days)
    audit.record(
        audit.KEY_ISSUED,
        principal,
        target_type="api_key",
        target_id=record["key_id"],
        label=record["label"],
        expires_at=record["expires_at"],
    )
    return {
        **record,
        "api_key": api_key,
        "notice": (
            "This API key is shown once and cannot be recovered. Roll it out to every caller, "
            "then revoke the old one — both work until you do."
        ),
    }


@router.delete("/account/keys/{key_id}")
def revoke_my_key(key_id: str, principal: Principal = Depends(current_principal)):
    """Revoke one key. The blast radius is exactly the callers using it.

    Refuses to revoke the last active key: an account with no working
    credential cannot issue itself a new one, and recovering from that needs an
    admin and a support conversation. Deactivate the account instead if that is
    what you meant.
    """
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="Authentication is disabled; there are no keys to manage.")

    # Ownership first. Checking "is this your last key" before "is this your
    # key" answered 409 to someone revoking a *different account's* key — a
    # confusing reply to a request that should simply not be found.
    keys = db.list_api_keys(principal.user_id)
    target = next((key for key in keys if key["key_id"] == key_id), None)
    if target is None:
        # Same answer for "no such key" and "not yours", for the same reason a
        # run belonging to someone else is a 404.
        raise HTTPException(status_code=404, detail="No such key on this account.")

    if target["active"] and sum(1 for key in keys if key["active"]) <= 1:
        raise HTTPException(
            status_code=409,
            detail=(
                "This is the only active key on the account. Issue a replacement first, or ask an "
                "admin to deactivate the account if you meant to end access entirely."
            ),
        )

    record = db.revoke_api_key(key_id, principal.user_id)
    if record is None:  # pragma: no cover - raced with another revocation
        raise HTTPException(status_code=404, detail="No such key on this account.")
    audit.record(audit.KEY_REVOKED, principal, target_type="api_key", target_id=key_id, label=record["label"])
    return record


@router.get("/admin/audit")
def audit_log(
    limit: int = Query(default=200, ge=1, le=1000),
    action: str | None = Query(default=None),
    principal: Principal = Depends(require_admin),
):
    """The append-only record of privileged and destructive actions."""
    return {
        "entries": db.list_audit(limit=limit, action=action),
        "attributable": audit.enabled(),
        "note": (
            "Append-only, and outside the retention policy it records."
            if audit.enabled()
            else "Authentication is off, so every entry is attributed to the single local owner."
        ),
    }


@router.get("/account/audit")
def my_audit(
    limit: int = Query(default=100, ge=1, le=500),
    principal: Principal = Depends(current_principal),
):
    """What this account did — key issuance, deletions, refused quota."""
    entries = [
        entry
        for entry in db.list_audit(limit=limit * 2, actor_id=principal.user_id)
        if entry["action"] in audit.SELF_VISIBLE
    ]
    return {"entries": entries[:limit]}


@router.get("/usage")
def usage(
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(current_principal),
):
    """What the LLM has cost, for this account. Estimated, and labelled so."""
    summary = db.usage_summary(owner_id=owner_scope(principal) or db.LOCAL_OWNER_ID, since_days=days)
    return {
        **summary,
        # What was spent is only half the question; what is still allowed is the
        # other half, and a client that wants to avoid a 429 needs both.
        "quota": limits.quota_status(principal),
        "note": "Costs are estimated from published list prices and are not an invoice.",
    }


@router.get("/admin/usage")
def all_usage(days: int = Query(default=30, ge=1, le=365), principal: Principal = Depends(require_admin)):
    return {**db.usage_summary(owner_id=None, since_days=days), "note": "Estimated, across all accounts."}


@router.get("/privacy")
def privacy():
    """Where uploaded data goes, as data rather than prose in a README."""
    return lifecycle.retention_policy()


@router.post("/admin/retention/purge")
def purge_now(principal: Principal = Depends(require_admin)):
    """Run the retention policy immediately."""
    # The audit entry is written inside purge_expired, so that the pass which
    # runs unattended at startup is recorded on the same path as this one.
    return lifecycle.purge_expired(actor=principal)


@router.delete("/account/data")
def delete_my_data(
    confirm: bool = Query(default=False, description="Must be true; this cannot be undone."),
    principal: Principal = Depends(current_principal),
):
    """Erase everything this account owns."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This permanently deletes every dataset, model and conversation you own. "
            "Repeat the request with ?confirm=true.",
        )
    owner = owner_scope(principal) or db.LOCAL_OWNER_ID
    summary = lifecycle.delete_everything_for(owner)
    # Recorded after the fact and outside the retention policy: this row is the
    # only thing that will still exist to say the erasure happened.
    audit.record(audit.ACCOUNT_DATA_ERASED, principal, target_type="account", target_id=owner, **summary)
    return summary
