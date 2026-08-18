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

import db
import lifecycle
from auth import Principal, create_user, current_principal, owner_scope, require_admin
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
    return user


@router.get("/usage")
def usage(
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(current_principal),
):
    """What the LLM has cost, for this account. Estimated, and labelled so."""
    summary = db.usage_summary(owner_id=owner_scope(principal) or db.LOCAL_OWNER_ID, since_days=days)
    return {
        **summary,
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
    return lifecycle.purge_expired()


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
    return lifecycle.delete_everything_for(owner)
