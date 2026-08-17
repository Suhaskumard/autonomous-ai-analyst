"""Central place for environment-driven settings.

Phase 2 of the upgrade plan replaces this with a pydantic-settings object; for
now it just keeps os.getenv calls from being scattered across the routes.
"""

import os

_MB = 1024 * 1024


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _list_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# --- CORS -------------------------------------------------------------------
# Explicit origins only. allow_credentials stays off until there is an auth
# story that needs it (a wildcard plus credentials is rejected by browsers).
def cors_origins() -> list[str]:
    return _list_env("CORS_ALLOW_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"])


# --- Upload limits ----------------------------------------------------------
def max_upload_bytes() -> int:
    return _int_env("MAX_UPLOAD_MB", 50) * _MB


def max_upload_rows() -> int:
    return _int_env("MAX_UPLOAD_ROWS", 1_000_000)


ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".txt", ".tsv"}

ALLOWED_UPLOAD_CONTENT_TYPES = {
    "text/csv",
    "text/plain",
    "text/tab-separated-values",
    "application/csv",
    "application/vnd.ms-excel",  # what Windows/Excel labels a .csv as
    "application/octet-stream",  # what several browsers send for any file
    "",
}


# --- LLM --------------------------------------------------------------------
def gemini_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key and key.strip() else None


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")
