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

    # --- Sandbox isolation (Phase 8) ---------------------------------------
    # "subprocess" is the restricted interpreter described in analyst/runner.py:
    # a denylist and an audit hook, which stop generated code that misbehaves
    # and would not stop generated code that attacks. "container" runs the same
    # runner inside `docker run --network none --read-only`, which is a kernel
    # boundary rather than an interpreter convention, and is what this should be
    # set to before anyone untrusted can reach the analyst.
    #
    # Default stays "subprocess": it needs no daemon and no image, which is the
    # right trade for the single-operator install, and switching is one variable.
    sandbox_backend: str = "subprocess"  # "subprocess" | "container"
    # The runtime and the image built from ops/Dockerfile.sandbox. Podman is a
    # drop-in here; it takes the same flags and needs no privileged daemon.
    sandbox_runtime: str = "docker"
    sandbox_image: str = "analyst-sandbox:latest"
    # A seccomp profile narrows the syscall table further than --cap-drop can.
    # Empty means the runtime's default profile, which is already restrictive;
    # ops/seccomp-analyst.json is the tighter one this repo ships.
    sandbox_seccomp_profile: str | None = None
    # A fork bomb is the cheapest denial of service in a language with os.fork,
    # and neither rlimits on Windows nor the audit hook stop one reliably.
    sandbox_pids_limit: int = Field(default=128, ge=16)
    sandbox_cpus: float = Field(default=1.0, gt=0)
    # Starting a container costs a second or so more than starting a process,
    # which matters because the timeout is a wall clock over the whole thing.
    sandbox_startup_grace_seconds: int = Field(default=10, ge=0)
    # Only for the docker-out-of-docker case: when the API is itself a container
    # talking to the host daemon, a bind mount is resolved by that daemon
    # against the *host* filesystem, so the path the API sees is not the path to
    # mount. Set this to the host directory that the API's models/ volume comes
    # from and snapshot paths are rewritten before being mounted. Wrong or unset
    # in that topology, the mount silently produces an empty directory.
    sandbox_host_models_dir: str | None = None
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

    # --- Per-principal limits (Phase 8) ------------------------------------
    # The Phase 5 guards are per *conversation*, which slows one session and
    # stops nothing: a client that opens a new conversation each time has no
    # ceiling at all. These are per account, measured against the usage and run
    # tables that already record every call. 0 disables a limit, which is the
    # default — a single trusted operator should not be rate-limited by their
    # own install.
    principal_max_llm_calls_per_hour: int = Field(default=0, ge=0)
    principal_daily_spend_limit_usd: float = Field(default=0.0, ge=0.0)
    # Disk is the other exhaustible resource, and the one a paid-endpoint limit
    # does not cover: uploading is free and training writes a model bundle.
    principal_max_runs: int = Field(default=0, ge=0)
    principal_storage_quota_mb: int = Field(default=0, ge=0)
    # Workers are the third exhaustible resource and the one neither of the
    # others covers: an account can sit under its spend and its disk quota and
    # still occupy every worker with training runs, which is an outage for
    # everyone else that costs the attacker almost nothing. Counted over jobs
    # this account has queued or running right now.
    principal_max_concurrent_jobs: int = Field(default=0, ge=0)

    # --- Key lifecycle (Phase 8) -------------------------------------------
    # More than one key may be active per account so that rotation is not an
    # outage. A ceiling keeps "rotate" from quietly becoming "accumulate".
    max_active_api_keys: int = Field(default=5, ge=1)
    # 0 means keys do not expire. Set a number of days for issued keys to age
    # out on their own; existing keys are unaffected.
    api_key_default_ttl_days: int = Field(default=0, ge=0)

    # --- Post-deployment monitoring (Phase 9) -------------------------------
    # Every phase before this one judged a model at the moment it was trained
    # and never looked again. These are the settings for looking again.
    #
    # On by default, unlike Phase 6's and Phase 8's capabilities, and the
    # difference is deliberate: those change how the application behaves for a
    # caller, while this only records what it already did. A model nobody is
    # watching is the failure this phase exists to prevent, so the safe default
    # is the one that watches. The data is owner-scoped and retention-governed
    # exactly like the datasets it came from.
    prediction_logging_enabled: bool = True
    # A batch prediction on 50,000 rows should not write 50,000 rows. Above this
    # the request is sampled and every stored row is flagged `sampled`, so a
    # partial log is never mistaken for a complete one.
    prediction_log_max_rows: int = Field(default=200, ge=1)
    # Inputs are as sensitive as the training rows they resemble. Turning this
    # off keeps the volume, latency and outcome record — which is what the
    # monitoring view is built from — and drops the feature values, which costs
    # drift detection entirely. Stated here because that is the whole trade.
    prediction_log_inputs: bool = True

    # How many recent predictions a drift comparison reads. Wide enough to be a
    # distribution, bounded so the query stays cheap on a busy run.
    drift_sample_size: int = Field(default=1000, ge=30)
    # Below this many logged predictions the comparison is reported but never
    # given a verdict. 200 is not a round default: comparing a reference
    # against a fresh sample of the *identical* distribution reads as drift on
    # essentially every trial below this row count, measured by repeated
    # simulation in ml/drift.py and its tests. Lowering this without lowering
    # NUMERIC_BINS to match reintroduces that false-positive rate.
    drift_min_rows: int = Field(default=200, ge=30)

    # Fit a calibrated variant of a classifier and keep it if it is measurably
    # better. Costs a few extra fits per training run. Off leaves calibration
    # *measured* and reported — the number is still in the metadata and still
    # in the API — and only skips acting on it.
    calibrate_probabilities: bool = True

    # Promotion margin for a retrained challenger, as a fraction of the
    # champion's score. A challenger that wins by less than this is not
    # promoted: repeated retraining on a fixed dataset produces a spread of
    # scores around the same true value, and promoting on any improvement at
    # all means promoting noise, forever, and calling it progress.
    challenger_promotion_margin: float = Field(default=0.01, ge=0.0)

    # --- Sharing (Phase 10) -------------------------------------------------
    # Off by default, like every Phase 6/8 capability that changes what an
    # unauthenticated caller may reach: a share link is a bearer credential
    # that needs no account, so turning this on is a decision an operator
    # makes, not a default this application makes for them.
    share_links_enabled: bool = False
    share_link_default_ttl_days: int = Field(default=7, gt=0)
    # A ceiling on what a creator may ask for, independent of the default —
    # "expiring" has to mean something even when a caller requests a year.
    share_link_max_ttl_days: int = Field(default=30, gt=0)

    # --- Scheduled reports (Phase 10) ---------------------------------------
    # Reuses Phase 6's queue to build the report; nothing here runs a
    # scheduler in-process, for the same reason startup retention does not
    # (main.py's lifespan). `POST /api/report-schedules/run-due` is what an
    # operator's cron entry calls.
    scheduled_reports_enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, gt=0)
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    smtp_from_address: str | None = None
    # A ceiling on how many rows delivering all due reports may process at
    # once, so one cron tick cannot be made to run an unbounded amount of
    # LLM-backed report generation by simply creating enough schedules.
    report_schedule_max_due_per_run: int = Field(default=25, ge=1)

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


#: Settings whose value may instead be given as a path, via `<NAME>_FILE`.
#: Every one of them is a credential.
_FILE_BACKED_SECRETS = (
    "gemini_api_key",
    "artifact_signing_key",
    "auth_bootstrap_token",
    "database_url",
    "smtp_password",
)


def _load_file_backed_secrets(settings: "Settings") -> None:
    """Support the `<NAME>_FILE` convention for secrets.

    Environment variables are readable by anything that can list a process, get
    a core dump, or run `docker inspect`, and they end up in shell history and
    CI logs. Docker secrets, Kubernetes secret volumes, and systemd credentials
    all present a secret as a *file* instead — so `GEMINI_API_KEY_FILE=/run/
    secrets/gemini` is what any of them can be pointed at without this
    application learning about a specific secret manager.

    The direct variable still works and still wins, so nothing that exists today
    changes. A `_FILE` that cannot be read is fatal rather than ignored: falling
    back to "no key" would look like a configuration choice.
    """
    import os

    for field in _FILE_BACKED_SECRETS:
        path = os.getenv(f"{field.upper()}_FILE")
        if not path:
            continue
        # `model_fields_set` rather than a truthiness check on the value.
        # `database_url` has a default — the local SQLite file — so a truthy
        # value means "set or defaulted", which made DATABASE_URL_FILE
        # unusable: it was every time reported as a conflict with a
        # DATABASE_URL nobody had set, and then ignored. What matters here is
        # whether the operator gave the variable, and this is the question that
        # asks that.
        if field in settings.model_fields_set:
            # Both given. The explicit variable wins, but say so — the two
            # disagreeing silently is how the wrong credential gets used.
            print(f"warning: both {field.upper()} and {field.upper()}_FILE are set; using {field.upper()}")
            continue
        secret = Path(path).read_text(encoding="utf-8").strip()
        if not secret:
            raise ValueError(f"{field.upper()}_FILE points at {path}, which is empty.")
        object.__setattr__(settings, field, secret)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _load_file_backed_secrets(settings)
    return settings


settings = get_settings()
