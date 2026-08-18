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

    # --- Training cost -----------------------------------------------------
    # Cross-validation multiplies every fit by the fold count, so breadth is no
    # longer free: ten candidates over five folds is fifty fits. This caps how
    # many candidates a run may try (0 = no cap, use the whole registry) for
    # operators who would rather have a fast answer than an exhaustive one.
    max_candidate_models: int = Field(default=0, ge=0)

    # --- Analyst sandbox ---------------------------------------------------
    # Generated code runs in a separate one-shot interpreter. These are the
    # ceilings it runs under; see analyst/runner.py for what they can and
    # cannot enforce on each platform.
    sandbox_timeout_seconds: int = Field(default=20, gt=0, le=300)
    sandbox_memory_mb: int = Field(default=1024, ge=128)
    # How many tool calls one question may take before the agent must answer.
    agent_max_steps: int = Field(default=6, ge=1, le=20)
    # Per-conversation ceilings, so one session cannot bill indefinitely.
    chat_max_messages_per_hour: int = Field(default=40, ge=1)
    chat_token_budget: int = Field(default=120_000, ge=1000)

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
