"""Structured logging with request and job correlation.

A failed pipeline used to leave a single string on a dict in memory. Every log
line now carries the request id and, inside a training job, the job id — so a
failure is diagnosable from the log stream alone.
"""

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.job_id = job_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "job_id": getattr(record, "job_id", None),
        }
        # Anything passed via logger.info(..., extra={...}) rides along.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps({k: v for k, v in payload.items() if v is not None}, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ids = " ".join(
            f"[{name}={value}]"
            for name, value in (("req", getattr(record, "request_id", None)), ("job", getattr(record, "job_id", None)))
            if value
        )
        base = f"{self.formatTime(record)} {record.levelname:<8} {record.name} {ids} {record.getMessage()}".replace(
            "  ", " "
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if as_json else TextFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own handlers; route them through ours.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
