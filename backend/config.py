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

    # --- Multi-user (Phase 6) ----------------------------------------------
    # Off by default: everything before this phase assumed one trusted operator
    # on localhost, and turning auth on by default would break that install for
    # no benefit. When off, a single implicit local owner owns every artifact,
    # so "every artifact has an owner" holds either way.
    auth_enabled: bool = False
    # Presenting this as `X-Bootstrap-Token` allows creating the first user.
    # Without it set, the registration endpoint is closed entirely.
    auth_bootstrap_token: str | None = None

    # --- Artifact integrity ------------------------------------------------
    # joblib.load() executes pickle opcodes, so loading an artifact whose bytes
    # someone else could influence is remote code execution. Every bundle is
    # HMAC-signed on write and verified on read. Leave unset to have a key
    # generated and persisted on first use.
    artifact_signing_key: str | None = None

    # --- Storage -----------------------------------------------------------
    storage_backend: str = "local"  # "local" | "s3"
    s3_bucket: str | None = None
    s3_prefix: str = "analyst"
    s3_endpoint_url: str | None = None  # set for MinIO and other S3-compatibles
    s3_region: str | None = None

    # --- Job queue ---------------------------------------------------------
    # "inline" runs training in the web process (FastAPI BackgroundTasks);
    # "rq" hands it to a separate worker so a web restart cannot kill a job.
    queue_backend: str = "inline"  # "inline" | "rq"
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "analyst"
    queue_job_timeout_seconds: int = 3600

    # --- Data lifecycle ----------------------------------------------------
    # Artifacts older than this are purged on startup and by the retention
    # endpoint. 0 disables expiry (artifacts are kept until deleted by hand).
    retention_days: int = Field(default=0, ge=0)

    # --- Observability -----------------------------------------------------
    metrics_enabled: bool = True
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Persistence -------------------------------------------------------
    # Jobs and run history live here so they survive a restart and are visible
    # to every worker process. SQLite is the single-process default and stops
    # being the right answer the moment a worker and an API process write at
    # once: set a postgresql:// URL for that shape (see docs/production.md).
    database_url: str = f"sqlite:///{(BASE_DIR / 'models' / 'analyst.db').as_posix()}"
    # Pool settings apply to server-backed databases only — SQLite gets a
    # single file handle and none of this is meaningful there. The defaults
    # suit one API process and one worker; raise pool_size with replicas, and
    # keep (pool_size + max_overflow) * processes under Postgres' max_connections.
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout_seconds: int = Field(default=30, gt=0)
    # Below Postgres' idle timeout and any proxy's, so a connection is recycled
    # by us rather than discovered dead by a request.
    db_pool_recycle_seconds: int = Field(default=1800, gt=0)
    # A worker or API process usually starts before the database accepts
    # connections. Retrying beats crash-looping and beats a healthcheck that
    # reports ready while the schema does not exist yet.
    db_connect_retry_seconds: int = Field(default=30, ge=0)
    # How long a single connection attempt may take before it is abandoned.
    # libpq's default is *no timeout*: a database that accepts the TCP SYN and
    # then goes silent — a wedged proxy, a dropped NAT entry, a host that
    # vanished — blocks the caller forever, with no error and nothing in the
    # logs. Found exactly that way, hanging a test run at connection ~200.
    db_connect_timeout_seconds: int = Field(default=10, gt=0)

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
