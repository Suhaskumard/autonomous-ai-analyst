"""Where training jobs run.

`BackgroundTasks` runs the pipeline inside the web process. That has two
consequences the plan calls out: a CPU-bound training run competes with request
handling, and a restart or deploy kills whatever was mid-flight. Phase 2 made
that survivable — an interrupted job is marked failed on the next startup
rather than polling forever — but survivable is not the same as surviving.

With `QUEUE_BACKEND=rq` the job is handed to Redis and picked up by a separate
worker process, so restarting the API does not touch it. `inline` keeps the
current behaviour, because requiring Redis to analyse a CSV on a laptop would
be a worse default than the problem it solves.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class JobQueue(ABC):
    name: str
    #: Whether a handed-over job survives this process exiting. The upload
    #: route branches on it: an out-of-process job needs its input file moved
    #: to shared storage first, because the worker cannot see our temp dir.
    out_of_process: bool = False

    @abstractmethod
    def enqueue(self, func: Callable, *args: Any, job_id: str | None = None, **kwargs: Any) -> str | None:
        """Schedule `func`. Returns a queue-side id, or None for inline."""

    @abstractmethod
    def healthy(self) -> bool:
        """Whether work handed over now would actually be picked up."""


class InlineQueue(JobQueue):
    """FastAPI BackgroundTasks: same process, same lifetime.

    Normally the task is attached to the request by the caller and `enqueue` is
    not used. It still runs the function — synchronously — because the one path
    that reaches it is the fallback from an unreachable Redis, and blocking the
    request is a great deal better than accepting a job nothing will ever run.
    """

    name = "inline"
    out_of_process = False

    def enqueue(self, func: Callable, *args: Any, job_id: str | None = None, **kwargs: Any) -> str | None:
        func(*args, **kwargs)
        return None

    def healthy(self) -> bool:
        return True


class RedisQueue(JobQueue):
    """RQ over Redis: the job outlives the web process that accepted it."""

    name = "rq"
    out_of_process = True

    def __init__(self, redis_url: str, queue_name: str, job_timeout: int):
        import redis
        from rq import Queue

        self._connection = redis.from_url(redis_url)
        self._queue = Queue(queue_name, connection=self._connection, default_timeout=job_timeout)

    def enqueue(self, func: Callable, *args: Any, job_id: str | None = None, **kwargs: Any) -> str | None:
        job = self._queue.enqueue(func, *args, job_id=job_id, **kwargs)
        logger.info("Job enqueued", extra={"queue_job": job.id, "queue": self._queue.name})
        return job.id

    def healthy(self) -> bool:
        try:
            return bool(self._connection.ping())
        except Exception as exc:
            logger.warning("Redis is not reachable: %s", exc)
            return False

    @property
    def depth(self) -> int:
        try:
            return int(len(self._queue))
        except Exception:  # pragma: no cover - redis down
            return 0


_queue: JobQueue | None = None


def get_queue() -> JobQueue:
    global _queue
    if _queue is not None:
        return _queue

    if settings.queue_backend == "rq":
        try:
            candidate = RedisQueue(settings.redis_url, settings.queue_name, settings.queue_job_timeout_seconds)
            if not candidate.healthy():
                raise RuntimeError(f"Redis at {settings.redis_url} did not answer PING")
            _queue = candidate
        except Exception as exc:
            # Falling back is the right call: a queue that cannot be reached
            # would silently accept jobs nothing ever runs. Inline is slower to
            # scale but it always finishes.
            logger.error("Queue backend 'rq' unavailable (%s); falling back to inline execution.", exc)
            _queue = InlineQueue()
    else:
        _queue = InlineQueue()

    logger.info("Job queue backend: %s", _queue.name)
    return _queue


def reset_queue() -> None:
    global _queue
    _queue = None


def set_queue(queue: JobQueue | None) -> None:
    """Install a queue directly (tests)."""
    global _queue
    _queue = queue


def queue_status() -> dict:
    """What /readyz and the metrics endpoint report about job execution."""
    queue = get_queue()
    status = {"backend": queue.name, "healthy": queue.healthy()}
    if isinstance(queue, RedisQueue):
        status["depth"] = queue.depth
    return status
