# Running this in production

Phase 6 wrote the multi-user shape: authentication, a real queue, object
storage, retention, metrics. Every part of it was tested against a substitute —
an in-memory bucket, a recording queue, a temporary SQLite file. This document
is the other half: the drills that replace each substitute with the real thing
and watch it work, and the settings that go with them.

The sequencing rule has not changed. **Turn one thing on at a time**, with
`/readyz` in view. Four subsystems enabled at once is a stack whose next failure
has four candidate causes, and each of these fails differently.

Everything below assumes the repo root as the working directory and Docker
running. On Windows, the shell scripts want Git Bash.

---

## The order

| # | Turn on | Profile | Proven by |
| --- | --- | --- | --- |
| 1 | A real queue | `queue` | `python ops/queue_restart_drill.py` |
| 2 | Postgres | `postgres` | the full suite against it |
| 3 | Object storage | `storage` | a replica serving a model it never trained |
| 4 | — | — | `ops/backup.sh` → restore → `ops/verify_restore.py` |
| 5 | — | — | `python ops/loadtest.py`, numbers written down |

Before any of them, set a signing key:

```bash
export ARTIFACT_SIGNING_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

Leave it unset and each process generates its own, which is correct on one
machine and silently wrong on two: a bundle signed by the worker fails
verification in the API, and the error says the artifact failed its integrity
check — which reads as tampering rather than as a missing environment variable.
`/readyz` reports `checks.artifacts.ok: false` whenever the key is generated and
anything is shared. Store it where the deployment's other secrets live; a
restored backup needs the same key or its models cannot be loaded.

---

## 1. The queue, for real

```bash
export QUEUE_BACKEND=rq
docker compose --profile queue up -d --build
python ops/queue_restart_drill.py
```

The drill uploads a dataset, waits until the worker has visibly started
training, runs `docker compose restart backend`, and polls until the job is
terminal. It passes when the run **completes** — the job outlived the process
that accepted it, and the API that came back reports the result of a job it
never ran.

It fails on purpose if `QUEUE_BACKEND` is `inline`, because there the restart
genuinely does kill the run: training happens in the web process, the job is
marked `interrupted`, and that is correct behaviour rather than a bug. A drill
that passed in both configurations would be proving nothing.

Restart the worker instead — `--service worker` — and RQ requeues the job. That
is the other half of the promise and worth watching once.

What the queue does **not** survive: a job that was in Redis when Redis was
restarted, since it runs with persistence off. A lost queue means a re-upload,
not a lost model.

---

## 2. Postgres

SQLite stops being the right store the moment a worker and an API process write
at once. It does not queue concurrent writers; it fails them with "database is
locked".

```bash
docker compose --profile postgres --profile queue up -d
export DATABASE_URL=postgresql+psycopg://analyst:analyst@postgres:5432/analyst
docker compose up -d --force-recreate backend worker
```

`postgres://` and `postgresql://` are both accepted and rewritten to the pinned
psycopg driver. Pool settings are `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`,
`DB_POOL_TIMEOUT_SECONDS` and `DB_POOL_RECYCLE_SECONDS`; keep
`(pool_size + max_overflow) × processes` under the server's `max_connections`,
which is 100 by default. They do nothing on SQLite, which has one file handle.

Run the suite against it — this is the exit criterion, and it is the only way to
find the places where SQLite was being forgiving:

```bash
docker compose --profile postgres up -d postgres
TEST_DATABASE_URL=postgresql+psycopg://analyst:analyst@localhost:5432/analyst pytest
```

The suite uses one database and empties it between tests. `test_production_path.py`
has a marker test that fails if `TEST_DATABASE_URL` is set but the run is
somehow still on SQLite, so a green Postgres CI job cannot be a SQLite one.

Timestamps are naive columns holding UTC. The engine pins the Postgres session
timezone to UTC so a value reads back as it was written, which is what SQLite
has always done by dropping the offset.

---

## 3. Alembic owns the schema

Before Phase 7, `init_db` called `create_all` and then patched in any column a
later phase had added. That covered exactly one kind of change — an added,
defaulted column — and did nothing at all, silently, for a rename, a type
change, or a backfill.

```bash
cd backend
alembic current                                     # what this database is at
alembic revision --autogenerate -m "what changed"   # a draft; read it
alembic upgrade head
alembic downgrade -1
```

The API and the worker both run `upgrade head` at startup, so a deploy needs no
manual step; on Postgres an advisory lock makes the second one wait rather than
race. A database created before Phase 7 is **adopted**: missing tables created,
missing columns added, then stamped at `0001_baseline`. Both halves matter — the
old startup only created what the phase running at the time knew about, so the
database in this repo has two of six tables and none of Phase 6's columns.
Nothing is dropped and no row is lost; there are tests for both shapes. A
backfilled column ends up nullable where the baseline says NOT NULL, so an
adopted database is compatible rather than identical.

`/readyz` returns **503** when the schema is not at the revision the code
expects. This is the one dependency that does not degrade: a queue can fall back
to inline and a bucket to local disk, but code that expects a column the
database does not have fails at whichever request first touches it, looking like
a bug in a route.

`backend/migrations/README.md` has the conventions. The suite fails if the
models and the migration scripts drift apart.

---

## 4. Object storage, and a replica that proves it

```bash
export STORAGE_BACKEND=s3 S3_BUCKET=analyst-artifacts
export S3_ENDPOINT_URL=http://minio:9000
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
docker compose --profile storage --profile postgres --profile queue up -d
```

`minio-init` creates the bucket and exits. Without it the first run fails on a
missing bucket — and the storage layer degrades past that by writing to local
disk, so the misconfiguration would look like everything working.

Then the assertion that actually proves a shared backend:

```bash
docker compose --profile replica up -d backend-replica
# train on the first API
curl -F file=@backend/tests/fixtures/clean_classification.csv -F mode=auto localhost:8000/api/upload
# …and predict from the second, which has its own empty volume
curl -X POST localhost:8001/api/predict/<run_key>/row \
     -H 'Content-Type: application/json' -d '{"row": {"age": 1, "spend": 1, "region": 1}}'
```

`backend-replica` runs on its own volume on purpose. It has never trained
anything and cannot see the first container's disk, so a prediction served from
it can only have come through the bucket — and it only loads at all if the
signature written by the trainer verifies under this process's key. That single
request exercises shared storage, shared database, and key portability at once.

`tests/test_production_path.py` runs the same scenario against a directory
standing in for the bucket, and against real MinIO when `MINIO_TEST_ENDPOINT` is
set. Credentials come from the boto3 chain, never from settings — they must not
be one `/readyz` bug away from being logged.

---

## 5. Backup and restore

A backup nobody has restored is a hypothesis. The drill is the point; the
scripts exist so it can be repeated.

```bash
ops/backup.sh                       # -> backups/<timestamp>/
docker compose down -v              # a genuinely clean stack, volumes included
docker compose --profile postgres --profile storage up -d
ops/restore.sh backups/<timestamp>
python ops/verify_restore.py
```

`backup.sh` takes the database (`pg_dump --format=custom`, or the SQLite online
backup API — a live SQLite file cannot be copied byte-for-byte) and the artifact
bytes (an `mc mirror` of the bucket, or a tar of the models volume), plus a
manifest recording the schema revision it came from.

`restore.sh` refuses to run against a stack that still has runs in it unless
`FORCE=1`. Restoring on top of the original data cannot tell you what the backup
was missing, which is the entire question.

`verify_restore.py` is the part that matters. It checks `/readyz`, lists the
runs, reopens the newest one, and **predicts from it** — because `pg_restore`
exiting zero says the rows arrived and says nothing about whether the model
still loads. The most likely failure is a stack restored without its original
`ARTIFACT_SIGNING_KEY`: every bundle then fails verification and is refused.

Redis is deliberately not backed up. It holds jobs in flight, not results.

What a restore does not bring back: an API key. Only the SHA-256 is stored, so a
restored account needs a new key issued.

---

## 6. Load, measured

```bash
python ops/loadtest.py --concurrency 8 --requests 24 --json ops/results/inline.json
# then again with the queue profile on, and compare
```

It reports upload accept latency (p50/p95/max) and one training run end to end.
Each virtual user uploads a *different* dataset: uploads are content-addressed,
so sending the same file twice measures the cache rather than the pipeline, and
makes any load test look wonderful.

Queue depth and training duration have been exported since Phase 6. This is what
turns them into numbers an incident can be compared against — record them in
`docs/phase7-results.md`, with the machine they came from. An unrecorded
measurement is a rumour.

---

## Reading /readyz

```jsonc
{
  "status": "ready",
  "checks": {
    "database":  { "ok": true, "dialect": "server" },
    "schema":    { "ok": true, "revision": "0001_baseline", "head": "0001_baseline" },
    "queue":     { "ok": true, "backend": "rq", "depth": 0 },
    "storage":   { "ok": true, "backend": "s3" },
    "artifacts": { "ok": true, "signing_key": "configured", "portable": true },
    "llm":       { "ok": true, "configured": true }
  }
}
```

Only `database` and `schema` are fatal — everything else is reported precisely
because it degrades rather than fails, and a degradation nobody can see is worse
than one that is written down. `queue.ok: false` means Redis is unreachable and
jobs are running inline. `artifacts.ok: false` means this process signs with a
key nothing else has.

## What is still not done

Phase 8 in `Upgrade_Plan.pdf`, and it matters if untrusted people will reach
this: rate limiting is still per conversation rather than per account, keys
cannot be rotated without downtime, there is no audit log, and secrets come from
the environment rather than a secret manager. The analyst sandbox is a
restricted execution environment, not a boundary — with untrusted users it needs
a container with no network namespace, a seccomp profile, and a read-only
filesystem.
