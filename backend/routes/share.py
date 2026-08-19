"""Read-only, expiring share links onto a run's result — Phase 10.

Two audiences, two kinds of route: `/runs/{run_key}/share*` is for the owner,
authenticated the same way as every other run-scoped endpoint. `/share/{token}`
is for whoever was handed the link — no principal, no API key, because the
token itself is the credential. See sharing.py for what a link does and does
not grant.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from auth import Principal, current_principal, require_owned_run
from sharing import create_link, list_links, resolve_token, revoke_link

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/runs/{run_key}/share")
def create_share_link(
    run_key: str,
    ttl_days: int | None = Query(default=None, gt=0, description="Defaults to SHARE_LINK_DEFAULT_TTL_DAYS"),
    principal: Principal = Depends(current_principal),
):
    require_owned_run(run_key, principal)
    return create_link(run_key, principal, ttl_days=ttl_days)


@router.get("/runs/{run_key}/share")
def list_share_links(run_key: str, principal: Principal = Depends(current_principal)):
    require_owned_run(run_key, principal)
    return {"links": list_links(run_key, principal)}


@router.delete("/share/{token}")
def delete_share_link(token: str, principal: Principal = Depends(current_principal)):
    revoke_link(token, principal)
    return {"token_prefix": token[:8], "revoked": True}


@router.get("/share/{token}")
def get_shared_run(token: str):
    """The dashboard payload for a shared run. No authentication — the token is it."""
    from routes.runs import _load_metadata
    from routes.upload import _result_payload

    run_key = resolve_token(token)
    try:
        metadata = _load_metadata(run_key)
    except HTTPException as exc:
        # The link is valid but the artifact behind it is gone — a deleted or
        # expired run whose share link outlived it for a moment. Same shape of
        # answer as an invalid token: there is nothing here to serve.
        raise HTTPException(status_code=404, detail="This share link is invalid or has expired.") from exc
    return {**_result_payload(metadata, status="shared"), "created_at": metadata.get("timestamp")}
