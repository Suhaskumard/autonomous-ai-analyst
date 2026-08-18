import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

import db
import lifecycle
from config import settings
from exceptions import PipelineError
from logging_config import configure_logging, request_id_var
from observability import CONTENT_TYPE_LATEST, init_sentry, metrics, normalise_path
from routes.admin import router as admin_router
from routes.chat import router as chat_router
from routes.insights import router as insights_router
from routes.predict import router as predict_router
from routes.profile import router as profile_router
from routes.runs import router as runs_router
from routes.upload import router as upload_router
from utils.artifacts import signing_status
from utils.helpers import ensure_storage_dirs
from utils.migrations import schema_status
from utils.queue import get_queue, queue_status
from utils.storage import get_storage

configure_logging(level=settings.log_level, as_json=settings.log_json)
logger = logging.getLogger(__name__)

# Before the app is built, so an exception raised during startup is reported.
init_sentry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage_dirs()
    db.init_db()
    # A job that was mid-flight when *this* process died would otherwise poll
    # as "running" forever with nothing working on it. That reasoning only
    # holds when the job ran here: with an external queue the run belongs to a
    # worker that is very likely still fitting it, and failing it on an API
    # restart would break the one guarantee the queue exists to provide. RQ's
    # own failed-job registry covers a worker that really did die.
    if not get_queue().out_of_process:
        recovered = db.recover_interrupted_jobs()
        if recovered:
            logger.warning("Marked %d interrupted job(s) as failed after restart", recovered)
    purged = db.purge_old_jobs()
    if purged:
        logger.info("Purged %d job records older than the retention window", purged)

    # Retention runs at startup rather than on a timer: an in-process scheduler
    # is one more thing that dies with the process, and an operator who wants it
    # hourly can curl the admin endpoint from cron.
    if settings.retention_days > 0:
        summary = lifecycle.purge_expired()
        if summary["runs_purged"]:
            logger.info("Retention purge at startup", extra=summary)

    # Said once, loudly, at startup as well as on /readyz: the failure it warns
    # about happens at prediction time, in another process, long after whoever
    # could have set the variable has stopped looking.
    artifacts = signing_status()
    if not artifacts["ok"]:
        logger.error(artifacts["detail"])

    logger.info(
        "Startup complete",
        extra={
            "database": "sqlite" if db.is_sqlite() else "server",
            "schema_revision": schema_status(db.get_engine()).get("revision"),
            "auth": settings.auth_enabled,
            "queue": queue_status()["backend"],
            "storage": get_storage().name,
            "artifact_signing_key": artifacts["signing_key"],
            "retention_days": settings.retention_days,
        },
    )
    yield


app = FastAPI(title="Autonomous AI Analyst", version="2.0.0", lifespan=lifespan)

# Explicit origins from CORS_ALLOW_ORIGINS. allow_credentials stays off even
# now that authentication exists: the credential is a bearer token in a header,
# which the browser does not attach on its own, so nothing here needs cookie
# semantics — and "*" plus credentials is a combination browsers reject anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request, and every log line it produces, with one id."""
    request_id = request.headers.get("x-request-id") or uuid4().hex[:12]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    elapsed = time.perf_counter() - started
    duration_ms = round(elapsed * 1000, 2)
    response.headers["x-request-id"] = request_id
    # Run keys and job ids are collapsed to ":id" first — one time series per
    # dataset is how a metrics endpoint turns into a memory leak.
    metrics.observe_request(request.method, normalise_path(request.url.path), response.status_code, elapsed)
    logger.info(
        "%s %s -> %s",
        request.method,
        request.url.path,
        response.status_code,
        extra={"status_code": response.status_code, "duration_ms": duration_ms},
    )
    return response


@app.exception_handler(PipelineError)
async def pipeline_error_handler(request: Request, exc: PipelineError):
    """Domain errors carry their own status code and kind."""
    logger.info("Pipeline error on %s: %s", request.url.path, exc, extra={"kind": exc.kind})
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc), "kind": exc.kind})


app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(predict_router, prefix="/api", tags=["predict"])
app.include_router(profile_router, prefix="/api", tags=["profile"])
app.include_router(insights_router, prefix="/api", tags=["insights"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(runs_router, prefix="/api", tags=["runs"])
app.include_router(admin_router, prefix="/api", tags=["admin"])


@app.get("/")
def health_check() -> dict:
    return {"message": "AUTONOMOUS AI ANALYST is running", "version": app.version}


@app.get("/healthz")
def liveness() -> dict:
    """Liveness: the process is up. Used by the container healthcheck."""
    return {"status": "ok"}


@app.get("/readyz")
def readiness():
    """Readiness: can this process actually serve?

    The database is the only hard dependency — without it nothing works, so a
    failure here is a 503 and the load balancer should stop sending traffic.
    Everything else is reported but not fatal: a dead Redis degrades to inline
    execution and an unreachable bucket degrades to local disk, both of which
    still serve requests. Reporting them keeps that visible rather than silent.
    """
    checks: dict = {}
    try:
        db.list_runs(limit=1)
        checks["database"] = {"ok": True, "dialect": "sqlite" if db.is_sqlite() else "server"}
    except Exception as exc:
        logger.exception("Readiness check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "checks": {"database": {"ok": False, "detail": str(exc)}}},
        )

    # A process running against a schema older than its code is a hard failure:
    # it will not fail at startup, it will fail on whichever request first
    # touches the column that is missing, which is a much worse way to find out.
    schema = schema_status(db.get_engine())
    checks["schema"] = schema
    if not schema["ok"]:
        logger.error("Schema is not at head", extra=schema)
        return JSONResponse(status_code=503, content={"status": "unavailable", "checks": checks})

    queue = queue_status()
    checks["queue"] = {"ok": queue["healthy"], **queue}
    storage = get_storage()
    checks["storage"] = {"ok": True, "backend": storage.name}
    # Reported, never fatal: a per-machine signing key serves fine until a
    # second process has to verify what this one wrote. See utils/artifacts.py.
    checks["artifacts"] = signing_status()
    checks["llm"] = {"ok": settings.gemini_key is not None, "configured": settings.gemini_key is not None}

    return {
        "status": "ready",
        "auth_enabled": settings.auth_enabled,
        "retention_days": settings.retention_days,
        "checks": checks,
        # Kept for the Phase 2 healthcheck and anything already reading it.
        "llm_configured": settings.gemini_key is not None,
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus exposition.

    Unauthenticated on purpose, and safe to be: every series is a counter or a
    duration labelled by method, normalised path, model, and outcome. No run
    key, filename, column name, or email is ever a label. Operators who still
    want it private should not route it from the internet — that is a job for
    the ingress, not for a bearer token this endpoint's scraper would have to
    hold.
    """
    if not metrics.enabled:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Metrics are disabled. Set METRICS_ENABLED=true and install prometheus-client.",
            },
        )
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)
