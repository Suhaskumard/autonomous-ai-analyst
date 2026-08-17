"""Typed application settings.

One object, validated at import, instead of os.getenv calls scattered across
routes. Every knob is documented in backend/.env.example.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
_MB = 1024 * 1024

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---------------------------------------------------------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash-8b"

    # --- CORS --------------------------------------------------------------
    # Kept as a raw string: pydantic-settings would otherwise try to JSON-decode
    # a list-typed field and reject the comma-separated form people actually write.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Upload limits -----------------------------------------------------
    max_upload_mb: int = Field(default=50, gt=0)
    max_upload_rows: int = Field(default=1_000_000, gt=0)

    # --- Persistence -------------------------------------------------------
    # Jobs and run history live here so they survive a restart and are visible
    # to every worker process.
    database_url: str = f"sqlite:///{(BASE_DIR / 'models' / 'analyst.db').as_posix()}"

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * _MB

    @property
    def gemini_key(self) -> str | None:
        key = (self.gemini_api_key or "").strip()
        return key or None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
