"""Authentication and per-user isolation.

Off by default. Every phase before this one assumed a single trusted operator
on localhost, and turning auth on by default would break that install to solve
a problem it does not have. When `AUTH_ENABLED` is false, every request is
attributed to one implicit local owner — so "every artifact has an owner" is
true either way, and switching auth on later does not orphan existing data.

When it is on:

* the credential is a bearer API key, 32 bytes of `secrets.token_urlsafe`;
* only its SHA-256 is stored, so a database dump does not yield working keys;
* the key is shown exactly once, at creation, because it cannot be recovered;
* comparison is constant-time, and a wrong key is indistinguishable from an
  unknown one.

Authorisation is separate and lives in `require_owner`: a valid key proves who
you are, not that a particular run is yours.
"""

import hashlib
import logging
import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException

import db
from config import settings

logger = logging.getLogger(__name__)

API_KEY_BYTES = 32
KEY_PREFIX = "analyst_"


@dataclass(frozen=True)
class Principal:
    """Who a request is acting as."""

    user_id: str
    email: str
    is_admin: bool = False

    @property
    def is_local(self) -> bool:
        return self.user_id == db.LOCAL_OWNER_ID


#: Used for every request when auth is disabled.
LOCAL_PRINCIPAL = Principal(user_id=db.LOCAL_OWNER_ID, email="local@localhost", is_admin=True)


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(API_KEY_BYTES)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_user(email: str, is_admin: bool = False) -> tuple[dict, str]:
    """Create a user and return (public record, the key shown only now)."""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="That email already has an account.")

    api_key = generate_api_key()
    record = db.create_user(
        user_id=secrets.token_hex(12),
        email=email,
        api_key_hash=hash_api_key(api_key),
        is_admin=is_admin,
    )
    logger.info("User created", extra={"user": record["user_id"], "admin": is_admin})
    return record, api_key


def _principal_from_key(api_key: str) -> Principal | None:
    """Resolve a bearer key, in constant time with respect to the secret."""
    row = db.get_user_by_key_hash(hash_api_key(api_key))
    if row is None or not row["is_active"]:
        return None
    # The lookup is by hash, but compare the stored hash again explicitly so
    # the success path does not short-circuit differently from the failure one.
    if not secrets.compare_digest(row["api_key_hash"], hash_api_key(api_key)):
        return None
    db.touch_user(row["user_id"])
    return Principal(user_id=row["user_id"], email=row["email"], is_admin=row["is_admin"])


def current_principal(authorization: str | None = Header(default=None)) -> Principal:
    """FastAPI dependency: who is calling.

    401 when auth is on and the credential is missing or bad; the local
    principal when auth is off.
    """
    if not settings.auth_enabled:
        return LOCAL_PRINCIPAL

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Send 'Authorization: Bearer <api key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    principal = _principal_from_key(authorization.split(" ", 1)[1].strip())
    if principal is None:
        # Deliberately the same message for unknown, wrong, and deactivated:
        # anything more specific tells an attacker which keys exist.
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="This endpoint requires an admin account.")
    return principal


def owner_scope(principal: Principal) -> str | None:
    """The owner filter for a query.

    None means "do not filter", which is only ever right when auth is off and
    there is exactly one owner. With auth on, every query is scoped — including
    an admin's, so an admin cannot read someone's data by accident.
    """
    return None if not settings.auth_enabled else principal.user_id


def require_owned_run(run_key: str, principal: Principal) -> None:
    """Authorise access to one run's artifacts, or 404.

    404 rather than 403 on purpose: a run key is a content hash, and confirming
    that a particular hash exists but belongs to someone else leaks both the
    existence of the dataset and the fact that another user holds it.
    """
    if not settings.auth_enabled:
        return
    owner = db.run_owner(run_key)
    if owner is None:
        # An artifact on disk with no registry row — a restored volume, a fresh
        # database, a deleted row. There is no owner to check against, and the
        # endpoints that read straight from disk (/insights, /runs/{k}/result)
        # would otherwise serve it to whoever asked. Unowned means unreachable
        # while auth is on; the operator's fix is to re-upload, not to leave a
        # dataset readable by every account on the server.
        logger.warning("Refused access to an unowned artifact", extra={"run_key": run_key})
        raise HTTPException(status_code=404, detail="No run found for this key.")
    if owner != principal.user_id:
        logger.warning(
            "Cross-owner artifact access refused",
            extra={"run_key": run_key, "owner": owner, "caller": principal.user_id},
        )
        raise HTTPException(status_code=404, detail="No run found for this key.")
