"""RQ worker: where training actually runs when the queue is enabled.

`BackgroundTasks` runs the pipeline inside the web process, so a deploy, a
crash, or a `docker compose restart backend` in the middle of a fit takes the
job with it. Phase 2 made that *visible* — an interrupted job is marked failed
on the next startup instead of polling forever — and this makes it survivable:
the API accepts the upload, Redis holds the job, and this process runs it. The
API can restart as often as it likes without the run noticing.

Run it alongside the API, with the same image and the same environment:

    QUEUE_BACKEND=rq python worker.py

Two things must line up between the two processes, and both fail loudly rather
than silently doing the wrong thing:

* `DATABASE_URL` must point at the same database, because the worker reports
  progress by writing the same job rows the API's status endpoint reads;
* the artifact directories must be the same volume, because the worker writes
  the model bundle the API later serves predictions from.

RQ forks per job on POSIX. On Windows it cannot, so use `QUEUE_BACKEND=inline`
for local development there — which is the default anyway.
"""

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

import db
from config import settings
from logging_config import configure_logging
from observability import init_sentry
from utils.helpers import ensure_storage_dirs

configure_logging(level=settings.log_level, as_json=settings.log_json)
logger = logging.getLogger(__name__)


def main() -> int:
    if settings.queue_backend != "rq":
        logger.error(
            "QUEUE_BACKEND is '%s', not 'rq'. This worker would sit idle while the API ran "
            "every job in its own process; start it with QUEUE_BACKEND=rq.",
            settings.queue_backend,
        )
        return 2

    try:
        import redis
        from rq import Queue, Worker
    except ImportError:
        logger.error("QUEUE_BACKEND=rq needs the 'redis' and 'rq' packages. Install backend/requirements.txt.")
        return 2

    ensure_storage_dirs()
    # The API normally creates the schema, but a worker can legitimately start
    # first — and it writes job rows, so it cannot assume the tables exist.
    db.init_db()
    init_sentry()

    connection = redis.from_url(settings.redis_url)
    try:
        connection.ping()
    except Exception as exc:
        # Better to exit and let the supervisor restart us than to run a worker
        # that quietly consumes nothing.
        logger.error("Redis at %s is not reachable: %s", settings.redis_url, exc)
        return 1

    logger.info(
        "Worker starting",
        extra={"queue": settings.queue_name, "redis": settings.redis_url, "database": settings.database_url},
    )
    worker = Worker([Queue(settings.queue_name, connection=connection)], connection=connection)
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
