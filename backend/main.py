import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

import db
from config import settings
from exceptions import PipelineError
from logging_config import configure_logging, request_id_var
from routes.chat import router as chat_router
from routes.insights import router as insights_router
from routes.predict import router as predict_router
from routes.profile import router as profile_router
from routes.runs import router as runs_router
from routes.upload import router as upload_router

configure_logging(level=settings.log_level, as_json=settings.log_json)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # A job that was mid-flight when the process died would otherwise poll as
    # "running" forever, with nothing working on it.
    recovered = db.recover_interrupted_jobs()
    if recovered:
        logger.warning("Marked %d interrupted job(s) as failed after restart", recovered)
    purged = db.purge_old_jobs()
    if purged:
        logger.info("Purged %d job records older than the retention window", purged)
    logger.info("Startup complete", extra={"database": settings.database_url})
    yield


app = FastAPI(title="Autonomous AI Analyst", version="2.0.0", lifespan=lifespan)

# Explicit origins from CORS_ALLOW_ORIGINS. allow_credentials is deliberately
# off: there is no auth story yet, and "*" plus credentials is a combination
# browsers reject outright.
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
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["x-request-id"] = request_id
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


@app.get("/")
def health_check() -> dict:
    return {"message": "AUTONOMOUS AI ANALYST is running", "version": app.version}


@app.get("/healthz")
def liveness() -> dict:
    """Liveness: the process is up. Used by the container healthcheck."""
    return {"status": "ok"}


@app.get("/readyz")
def readiness():
    """Readiness: the process can actually serve — i.e. the database answers."""
    try:
        db.list_runs(limit=1)
    except Exception as exc:
        logger.exception("Readiness check failed")
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": str(exc)})
    return {"status": "ready", "llm_configured": settings.gemini_key is not None}
