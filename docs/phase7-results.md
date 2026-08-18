# Phase 7 — the evidence

Phase 7 is not a feature. Its deliverable is evidence that the Phase 6 code does
what it says when the substitutes are removed, and evidence has to be written
down somewhere or it decays into "I think we tried that once".

This file is the record. Every row below was filled in by watching the thing
happen on 2026-08-18, against a real stack: RQ worker, PostgreSQL 16.14, MinIO,
and a second API replica. Fill a row in when you have watched it, not when you
believe it would work.

Each drill is in `docs/production.md`. Each writes its own PASS/FAIL.

The drills found **eleven** defects that the unit tests could not: two in the
container images, one live-credential leak, four in the drill scripts
themselves, one in the test harness, and one — a missing database connect
timeout — in the application. That ratio is the argument for this phase.

---

## Exit criteria

| # | Criterion | Drill | Status | Date | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | A training run survives `restart backend` and is seen to finish | `python ops/queue_restart_drill.py` | ☑ **PASS** | 2026-08-18 | 250k-row dataset; restart landed 2.1 s in, at 12% "Reading CSV". Run finished at 100%, key `e341b9a0ff8f…`, and the restarted API reported it. |
| 2 | The full suite passes against Postgres, Alembic-managed | `TEST_DATABASE_URL=… pytest` | ☐ not yet run | | |
| 3 | A prediction is served from a replica that never trained the model | replica curl, `docs/production.md` §4 | ☑ **PASS** | 2026-08-18 | Trained on `:8000` (worker, Postgres, MinIO). `:8001` had **0 files** on disk, pulled all four artifacts from the bucket on demand, verified a signature written by another process, and answered `no` at 0.755. |
| 4 | A restored backup produces a working system | `ops/backup.sh` → `ops/restore.sh` → `python ops/verify_restore.py` | ☑ **PASS** | 2026-08-18 | `down -v` destroyed all four volumes. After restore, the run trained before the wipe opened and predicted on **both** `:8000` and `:8001`. 25 KB dump + 2.22 MiB of bucket. |

What *is* already proven, by `backend/tests/test_production_path.py` on every CI
run, is the part that does not need containers:

- a pre-Alembic database is adopted rather than recreated, and keeps its rows —
  both a Phase 6 one (every table) and a Phase 2 one (two tables, no `owner_id`,
  which is what `backend/models/analyst.db` in this repo actually is);
- the migrations and the models have not drifted apart;
- the baseline rolls back and forward again;
- `/readyz` is 503 when the schema is behind the code;
- a bundle signed in a *separate process* verifies here, and one signed with a
  different key is refused;
- a run whose local artifacts are all deleted is still servable from the bucket.

That is the boundary worth keeping in mind: the tests prove the mechanisms, the
drills prove the deployment.

The scripts themselves have been exercised against a local `uvicorn` (not a
container): `ops/loadtest.py` produced the numbers below, and
`ops/verify_restore.py` was run against both an empty server (FAIL, exit 1, with
the reason) and one with a trained run (PASS, exit 0). What has *not* run is the
half that needs Docker: `queue_restart_drill.py`, `backup.sh`, `restore.sh`.

---

## Numbers

`python ops/loadtest.py --concurrency 8 --requests 24 --json ops/results/<name>.json`

Record each configuration separately. The interesting comparison is inline
versus queued at the same concurrency: it is what the queue bought.

| Configuration | Concurrency | Uploads | Accept p50 | Accept p95 | Accept max | Train wall | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inline + SQLite + local disk | 1 | 1 | 0.028 s | — | 0.028 s | **16.0 s** | none |
| inline + SQLite + local disk | 4 | 8 | 0.054 s | 0.176 s | 0.176 s | **126.9 s** | none |
| rq + Postgres + local disk | | | | | | | |
| rq + Postgres + MinIO | | | | | | | |

Read those two rows together, because the pair is the finding. Accepting an
upload is cheap either way — the request only streams the file to disk, hashes
it, and hands the job over, so all eight were accepted in 0.2 s at 35/s. What
changed is the training: **16 s alone, 127 s with eight runs in flight**, an
eight-fold slowdown from eight-fold concurrency. That is inline execution
working exactly as described — training competes with itself and with request
handling inside the web process — and it is the number the queued row exists to
be compared against.

Recorded 2026-08-18 against a local uvicorn (not the container), inline queue,
SQLite, local disk. Raw: `ops/results/inline-sqlite-local.json`.

Machine, so a later number can be compared to this one rather than to a
different laptop:

- CPU / RAM: Intel Core i5-12500H, 12C/16T, 15.6 GB — Windows 11, Python 3.13.13
- Docker version and platform: *not measured — the daemon was not running*
- Dataset: 1,500 rows × 5 columns, generated per virtual user by `ops/loadtest.py`

### Backup and restore timings

| Step | Duration | Size on disk |
| --- | --- | --- |
| `ops/backup.sh` | | |
| `ops/restore.sh` | | |
| Time from `docker compose up` to `/readyz` ready | | |

---

## What went wrong

The most valuable column in this file, and the one that is always empty in
practice. A drill that passed first time teaches nothing; write down the one
that did not, and what the misleading symptom was.

| Drill | Symptom | Actual cause | Fix |
| --- | --- | --- | --- |
| Load test | A repeat run reported training in **1.0 s** instead of 16 s | The generated datasets were seeded by request index, so the second invocation uploaded bytes the server had already trained on and got the cached run back. The tool was measuring the content-address cache. | `ops/loadtest.py` now picks a random seed base per invocation and prints it; `--seed` reproduces a set deliberately. |
| Restart drill | Crashed with `ConnectionAbortedError` traceback instead of reporting anything | The polling loop caught `urllib.error.URLError`, but a socket dying part-way through a response raises `ConnectionAbortedError`/`RemoteDisconnected` from `http.client` — which is *the event the drill exists to cause*. It crashed on its own success condition. | `get_json` now catches `OSError` and `http.client.HTTPException`. |
| Restart drill | Second attempt: "the job finished before the restart" on a dataset that takes a minute to train | The first (crashed) attempt had already trained that exact file, so the re-upload was a content-address cache hit answered in about a second. Same trap as the load test, one layer up. | The drill now detects `result.status == "reused"` and says so instead of blaming the dataset size. |
| First `docker compose ps` | `worker` and `frontend` both permanently **unhealthy** while working perfectly | Two independent probes that could never pass. The worker inherits the API image's `curl localhost:8000/healthz` and runs no web server. The frontend probes `http://localhost/`, which busybox wget resolves to `::1`, while `nginx.conf` says `listen 80;` — IPv4 only. | Worker gets a compose-level healthcheck that pings Redis (the thing that actually matters for it); the frontend Dockerfile probes `127.0.0.1`. All seven services now report healthy. |
| Replica setup | The "empty" replica volume already held 10 metadata files and 5 model bundles | **The backend build context had no `.dockerignore`.** Docker reads it from the *build context root* — here `./backend` — and only the repo-root file existed, whose patterns (`backend/models`, `backend/.env`, …) could never match from inside `backend/`. So `COPY . .` baked in `backend/.env` (the live Gemini key), every dataset snapshot, every fitted model, `tests/`, and `scratch/`; Docker then seeded the named volume from that on first mount. `frontend/` did have one (`node_modules`, `dist`, `.env`), which is why that image was unaffected. | Added `backend/.dockerignore`, widened `frontend/.dockerignore`, and noted on the root file that it governs nothing that is actually built. Verified after rebuild: no `.env`, no `tests/`, no `scratch/`, 0 baked artifacts. **The Gemini key must be rotated — it was in every backend image built on this machine, on top of being in git history.** |
| Replica setup, again | After rebuilding, the replica volume *still* had the stale files | `docker compose build backend` rebuilds only that service's image. `worker` and `backend-replica` build from the same context but are separate images and were still the old, polluted ones. | Rebuilt all three, removed the seeded volume, and re-verified 0 files before the drill. |
| Backup | Green transfer summary, but the bucket landed in `C:/Program Files/Git/backup/` | Git Bash rewrites arguments that look like absolute POSIX paths into Windows paths, so the *container* path `/backup` was mangled before docker saw it. The backup reported success and was missing every artifact. | `MSYS_NO_PATHCONV=1` / `MSYS2_ARG_CONV_EXCL='*'` at the top of both scripts. The same bug then bit a `pg_restore -l` I typed by hand, which is how obvious it is to miss. |
| Backup | `database.dump` was **816 bytes** and restored an empty system | The script dumped `POSTGRES_DB`, but the application reads `DATABASE_URL`, and the two named different databases. Two sources of truth for "which database", and the wrong one is silently empty rather than an error. | Both scripts now parse the database and user out of `DATABASE_URL` when it is set, falling back to `POSTGRES_DB`. |
| Restore | `mc: \`sh\` is not a recognized command. Did you mean \`share\`?` | The `minio/mc` image's entrypoint is `mc`, so `... minio/mc:latest sh -c "…"` ran `mc sh`. | Pass `--entrypoint sh` for that step. |
| Suite on Postgres | Hung at test ~38 (`test_chat_answers_and_persists_the_conversation`), **forever**, 0.02 CPU-seconds per 10 s wall and no Postgres session to show for it | `py-spy dump` on the stuck process: blocked in `psycopg.connect` → `select`, under `drop_all` in the test fixture. **libpq has no default connect timeout**, so a connection that never completes blocks the caller indefinitely — and it silently defeated `wait_for_database`, which can only retry an exception it never receives. What stopped the connections completing was the fixture itself: it disposed the engine per test, throwing away the pool and opening several hundred short-lived sockets, which wedged Docker Desktop's port proxy. | `DB_CONNECT_TIMEOUT_SECONDS` (default 10) on the Postgres engine, so this fails loudly instead of hanging. The fixture keeps one engine and pool for the session on a server database; SQLite still gets a fresh one per test because its file is recreated. Test throughput went from stuck-at-37 to ~27/min. |
| Migration adoption | — (caught before it shipped) | `backend/models/analyst.db` here is a *Phase 2* database: two of six tables, no `owner_id`. Stamping it at the baseline would have declared the schema complete while `users` did not exist. | Adoption creates missing tables and backfills missing columns before stamping. `test_a_phase_2_database_is_brought_up_to_the_baseline` covers it. |

Known candidates, so they are recognised rather than debugged from scratch:

- **"Artifact failed its integrity check"** after enabling the queue or a
  replica → `ARTIFACT_SIGNING_KEY` is unset somewhere, so that process generated
  its own. The message says tampering; the cause is configuration. `/readyz`
  says `artifacts.ok: false` before this ever happens.
- **Jobs accepted but never started** → the worker is not running, or is running
  with `QUEUE_BACKEND=inline`, in which case it consumes nothing while the API
  quietly runs every job itself.
- **"Database is locked"** → SQLite with two writers. That is the whole reason
  Postgres is in this phase.
- **Everything works, but the bucket is empty** → `S3_BUCKET` is wrong or
  missing and storage degraded to local disk. It is logged as an error at
  startup and reported by `/readyz`; the app deliberately keeps serving.
- **A restored stack lists runs that 404** → the database came back and the
  artifacts did not.
