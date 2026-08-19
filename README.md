# 🚀 ADVANCED AUTONOMOUS AI ANALYST

An enterprise-grade, full-stack autonomous data science system. It automatically ingests tabular data, executes an end-to-end ML pipeline, and provides an **Advanced AI Analyst** powered by Gemini with native **Code Execution** capabilities.

![Aesthetic Dashboard](https://img.shields.io/badge/UI-Premium_Glassmorphism-blueviolet)
![Engine](https://img.shields.io/badge/Engine-Gemini_(configurable)-orange)
![Capability](https://img.shields.io/badge/Chat-Code_Execution_Enabled-success) 
 
## 🌟 Key Features
  
- **An analyst that actually analyses**:
  - **Code runs here, not in someone else's cloud.** Generated pandas executes in a sandboxed subprocess on this machine, against the stored snapshot. The rows never leave your disk, and every number can be recomputed by hand from the CSV.
  - **Tools before code.** `describe_column`, `correlate`, `filter_rows`, `group_stats`, `plot` and `run_model` are deterministic implementations — the same question gives the same answer, with no generation step. `run_python` is the escape hatch for genuinely novel questions.
  - **The work is shown, not summarised.** Each answer carries the tool calls behind it: the code that ran, the table it returned, the chart it drew. Rendered as code, tables and images — not flattened into markdown prose.
  - **History lives on the server.** The client posts a question and a conversation id, never a transcript it could have edited. Conversations persist per dataset and are deleted with it.
  - **Honest about its reach**: the header reports the model actually configured and that execution is local. With no API key the chat says so plainly and everything else keeps working.
- **Autonomous report**: one button runs a full EDA sweep — column profiles, relationships, per-segment consistency, distributions, model result — and writes it up. Every number is computed by the same tools; the model only writes the prose around results it was handed, and the report still builds (findings without narrative) when no model is configured.
- **Premium Design System**: 
  - Modern **Glassmorphic** UI with a sophisticated dark-mode aesthetic.
  - HSL-tailored color palette with vibrant accents and micro-interactions.
  - Responsive layout with high-performance animations.
- **Autonomous ML Pipeline**:  
  - **Leak-free by construction**: the train/test split happens *before* anything is fitted. Preprocessing and estimator live in one sklearn Pipeline fitted on the training split only, so no test-set statistic reaches training.
  - **Target you control**: pick the target column in the UI, or let it auto-detect. Free-text and identifier columns are refused with an explanation instead of being trained on.
  - **Smart Modeling**: Auto-selection, Ensembling, or Manual fine-tuning. String class labels are encoded for XGBoost and decoded again at predict time.
  - **Explainability**: SHAP importances labeled with real column names, with one-hot columns summed back into their source column. The UI states whether SHAP or model-native importances were used.
  - **Health verdict**: every run is scored against a naive baseline (majority class / mean), so a model that does not beat guessing is labeled as such rather than as the "Optimized Model".
- **Scores you can defend**:
  - **Cross-validated, not lucky**: every candidate is scored over stratified folds and reported as **mean ± standard deviation**, so a model that won on one fortunate split is visibly indistinguishable from its neighbours.
  - **A holdout nothing touches**: 20% of the data is withheld from cross-validation, hyperparameter search, and model selection alike, then scored exactly once. That number is the one the dashboard leads with.
  - **Selected on macro F1, not accuracy**: on a 95/5 target a constant predictor scores 95% accuracy. It scores 0.49 macro F1 against a 0.49 baseline, so it can no longer win — and every candidate is compared against a `DummyClassifier` cross-validated on the very same folds.
  - **Real metrics**: balanced accuracy, macro/weighted F1, and ROC-AUC for classification, with a per-class precision/recall/F1/support breakdown; MAE, R², and MAPE alongside RMSE for regression. MAPE is withheld rather than faked when the actuals are mostly zero.
- **Imbalance is named and handled**:
  - A skewed target produces a warning that says how skewed, and class weights are applied to every estimator that supports them (XGBoost gets `scale_pos_weight`).
  - Optional **SMOTE** oversampling runs *inside* the pipeline, so synthetic rows only ever appear in a fold's training half and never in the rows a score is computed on.
- **Tuning with a budget you set**:
  - Randomized hyperparameter search over each model's space, bounded by a wall-clock budget. A long sweep degrades to "tuned the first few, kept defaults for the rest" rather than hanging, and the chosen parameters are recorded per model.
- **A registry worth choosing from**:
  - Logistic/linear regression, ridge, lasso, elastic net, random forest, extra trees, gradient boosting, histogram gradient boosting, decision tree, SVM, XGBoost, LightGBM — and CatBoost when installed. The candidate set adapts to dataset size, because a histogram booster is all fixed cost on 300 rows and a kernel SVM is superlinear on 50,000.
  - High-cardinality categoricals are **target-encoded** (with sklearn's internal cross-fitting) instead of exploding into a capped one-hot block that throws the tail away.
- **Ensembling that earns its name**:
  - A weighted `VotingClassifier`/`VotingRegressor` rather than an unweighted mode/mean vote. Members are weighted by how far they beat the baseline, and anything that fails to beat it is excluded from the vote entirely. Soft voting means an ensemble now reports calibrated confidence like any other model.
- **Model cards**:
  - Every run persists its configuration, feature typing, class distribution, cross-validated and held-out metrics, tuning history, and the exact library versions that produced them — so a result from three months ago is still interpretable when a scikit-learn upgrade moves a score.
- **Performance Caching**:
  - Runs are content-addressed on the whole configuration — dataset hash + mode + manual model + target + pipeline version — so changing the mode retrains instead of silently replaying the previous result.
  - A replayed run says so. The dashboard states whether a result was freshly trained, served from cache, or reopened from the workspace.
- **See the data before you train it**:
  - A per-column profile — type, missing %, cardinality, distribution sparkline — is computed on upload, before anything is fitted. The target picker lives on that same screen, so choosing a target is an informed decision rather than a bare dropdown.
  - Columns that cannot serve as a target are disabled with the reason attached.
- **Every number is drawn, not just computed**:
  - Histograms, category bars, feature importances, and per-model scores render as real charts (Recharts) with axes and scales, plus a confusion matrix for classification or a residual plot for regression, and a correlation heatmap.
  - Every chart has a table view — the numbers are readable without relying on colour, and the palette is validated for contrast and colour-vision deficiency on this surface.
  - A per-column statistics table shows what was actually trained on: the trained shape, classes, dropped columns, duplicates removed, and the mean/median/range/spread of each column.
- **Dataset workspace**:
  - Past runs survive a refresh and a restart. Reopen one into the dashboard, expand a lightweight summary of its health and drivers, compare model scores across up to four runs, or delete a run and every artifact belonging to it.
- **Prediction flow**:
  - Upload a CSV or paste a single row. Results come back as a table with class probabilities and confidence, downloadable as CSV.
- **Honest parsing**:
  - Malformed rows are counted and reported rather than silently dropped, and text is Unicode-normalised instead of stripped to ASCII — a Japanese or Arabic dataset trains without mangling.

---

## 🛠 Project Architecture

```text
backend/
  analyst/        <-- The agent: tools, sandbox, provider, report
    sandbox.py    <-- Spawns the one-shot runner, times it out, parses it
    runner.py     <-- The restricted child interpreter (guards live here)
    tools.py      <-- Deterministic tools + the run_python escape hatch
    providers.py  <-- LLM behind an interface (Gemini, and a fake for tests)
    agent.py      <-- The bounded call-tools-then-answer loop
    report.py     <-- The autonomous EDA sweep
  routes/
    chat.py       <-- Analyst endpoint, conversations, report
    upload.py     <-- Pipeline orchestration & Caching
    predict.py    <-- Inference API (batch + single row)
    profile.py    <-- Profile a CSV without training it
    runs.py       <-- Run registry: list, reopen, compare, delete
    insights.py   <-- Lightweight per-run summary
    monitoring.py <-- Predictions, drift, versions, retrain (Phase 9)
  ml/             <-- Core Analytical Engine
    preprocess.py <-- Target validation, feature typing (fits nothing)
    train.py      <-- Split-then-fit Pipelines, label encoding
    explain.py    <-- Name-aligned feature attribution
    health.py     <-- Baseline comparison / result verdict
    profile.py    <-- Per-column profile served before training
    diagnostics.py<-- Confusion matrix, residuals, correlations
    evaluate.py   <-- Imbalance-aware metrics + per-class breakdown
    imbalance.py  <-- Detection, class weights, in-pipeline SMOTE
    tuning.py     <-- Budgeted randomized search
    model_card.py <-- Config, data profile, metrics, library versions
    drift.py      <-- Population Stability Index, training vs. served inputs
    calibration.py<-- Expected calibration error; fits & judges a fix
    serving_schema.py <-- Renamed/missing/type-broken columns, named clearly
    quality.py, ensemble.py
  monitoring.py   <-- Prediction logging + the monitoring view (Phase 9)
  registry.py     <-- Model versions, champion/challenger promotion
  retraining.py   <-- Fits a challenger, scores it against the champion
  utils/
    security.py   <-- Artifact-key validation and path resolution
    uploads.py    <-- Bounded, streamed upload handling
  models/         <-- Generated artifacts (gitignored)
    metadata/     <-- Per-run blueprints + data snapshot
    saved_models/ <-- Fitted Pipeline bundles

  migrations/     <-- Alembic: the schema, reviewable one change at a time
    versions/     <-- 0001_baseline is phases 2-6, as create_all left them
  db.py           <-- Job + run persistence (SQLModel; SQLite or Postgres)
  worker.py       <-- The RQ worker: training that outlives an API restart
  config.py       <-- Typed settings (pydantic-settings)
  logging_config.py <-- JSON logs with request/job correlation
  exceptions.py   <-- Domain errors the routes map to status codes
  tests/          <-- pytest suite + fixture CSVs

frontend/
  src/
    styles.css      <-- Premium Design System
    components.css  <-- Tables, charts, workspace, profile
  components/
    DataProfile.jsx    <-- Pre-training column profile + target picker
    Dashboard.jsx      <-- Result surface: health, charts, summary
    DatasetSummary.jsx <-- Per-column statistics and run provenance
    ModelCard.jsx      <-- How the score was produced: CV, classes, tuning
    Workspace.jsx      <-- Past runs: reopen, compare, delete
    PredictPanel.jsx   <-- CSV or single-row inference, CSV download
    ChatBox.jsx, ErrorBoundary.jsx
    charts/         <-- Recharts components on one validated palette
  hooks/
    useUploadJob.js <-- Cancellable, backed-off, bounded polling
  pages/            <-- Unified Analytical Workflows
  tests/            <-- Vitest + Testing Library

ops/                       <-- The production drills: restart, backup, restore, load
  queue_restart_drill.py   <-- Restart the API mid-training; the run must finish
  backup.sh / restore.sh   <-- Database + artifacts, together or not at all
  verify_restore.py        <-- Proves a restored run still opens and still predicts
  loadtest.py              <-- Upload latency and training wall time, recorded
docs/
  production.md            <-- Runbook: what to turn on, in what order, proven how
  phase7-results.md        <-- Where the numbers and the drill results are written

.github/workflows/ci.yml   <-- lint, format, tests (SQLite + Postgres), builds, secret scan
docker-compose.yml         <-- one-command startup; profiles for queue/postgres/storage/replica
```

---

## 🚀 Quick Start

### 1. Setup Backend
1. Navigate to `backend/`.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. **Configure Environment Variables**:
   - Copy `backend/.env.example` to `backend/.env` and fill it in. `.env` is gitignored — never commit it.
     ```env
     GEMINI_API_KEY="your_api_key_here"
     GEMINI_MODEL="gemini-1.5-flash-8b"  # Recommended for best availability

     # Optional, shown with their defaults
     CORS_ALLOW_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
     MAX_UPLOAD_MB=50
     MAX_UPLOAD_ROWS=1000000
     MAX_CANDIDATE_MODELS=0   # 0 = try the whole registry

     # Analyst sandbox and endpoint guards
     SANDBOX_TIMEOUT_SECONDS=20
     SANDBOX_MEMORY_MB=1024
     AGENT_MAX_STEPS=6              # tool calls allowed per question
     CHAT_MAX_MESSAGES_PER_HOUR=40  # per conversation
     CHAT_TOKEN_BUDGET=120000       # per conversation
     ```
5. **Verify Connection**:
   ```bash
   python scratch/test_gemini.py
   ```
6. Run the server: `uvicorn main:app --reload`.

### 2. Setup Frontend
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`.
3. Start the dev server: `npm run dev`. It proxies `/api` to `localhost:8000`, so no CORS is involved in development.

### 3. Or just use Docker
```bash
docker compose up --build
```
Frontend on http://localhost:5173, API on http://localhost:8000. nginx serves the
built app and proxies `/api` to the backend, so the browser makes no cross-origin
requests. Trained artifacts and the job database persist in the `analyst-models` volume.

### 4. Developing
```bash
pip install -r backend/requirements-dev.txt
pre-commit install     # gitleaks + ruff run before every commit

pytest                 # backend suite
ruff check . && ruff format --check .
cd frontend && npm test && npm run build
```
Changed a model in `db.py`? The schema is Alembic's, so it needs a script:
```bash
cd backend
alembic revision --autogenerate -m "what changed"   # a draft — read it
alembic upgrade head
```
The suite fails if the models and `backend/migrations/versions/` have drifted apart. To run it against Postgres instead of SQLite:
```bash
docker compose --profile postgres up -d postgres
TEST_DATABASE_URL=postgresql+psycopg://analyst:analyst@localhost:5432/analyst pytest
```
Dependencies are pinned. Edit `backend/requirements.in`, then recompile:
```bash
uv pip compile backend/requirements.in -o backend/requirements.txt
uv pip compile backend/requirements-dev.in -o backend/requirements-dev.txt
```

---

## 🤖 Using the Advanced Analyst

Once a dataset is trained, the analyst is your partner for exploration. It picks tools, runs them
against your rows, and shows you the work:

- **"Which two features correlate most strongly, and does that hold within each segment?"** — one
  `correlate` call with `within`, returning the overall ranking and a per-segment breakdown you can
  check against the CSV.
- **"How many rows have spend above 60 in the north region?"** — a `filter_rows` call with the
  count, the match rate, and a sample.
- **"Plot the distribution of the target"** — a `plot` call returning a rendered PNG.
- **Anything else** — falls through to `run_python`, whose code and output are both shown.

Every claim it makes comes from a tool result. Expand any step to see the code, the table, or the
chart that produced the number.

### Guards on the endpoint

The LLM endpoint is the one that costs money, so it is bounded: `AGENT_MAX_STEPS` tool calls per
question, `CHAT_MAX_MESSAGES_PER_HOUR` messages per conversation, and a `CHAT_TOKEN_BUDGET` after
which a conversation must be restarted. Exceeding a limit returns 429 with an explanation. When the
budget forces a trim, the answer says so rather than quietly losing context.

---

## 🎨 Design Philosophy

The interface follows a **"Depth & Clarity"** approach:
- **Glassmorphism**: Layers of transparency and blur for a hierarchy of information.
- **Electric Accents**: Indigo and Pink accents provide high-contrast visual cues.
- **Micro-interactions**: Hover states, smooth progress transitions, and slide-up animations for a premium feel.

---

## 📋 Operational Notes

- **Clean Slate**: trained artifacts live in `backend/models/` and are gitignored; they are regenerated on demand and never committed.
- **Dataset Snapshot**: the backend stores a sanitized CSV snapshot per run, used for the chatbot and for reproducing results.
- **Durable jobs**: job state and the run registry live in SQLite (`DATABASE_URL`), so a restart does not lose them and multiple workers see the same rows. A job interrupted by a restart is marked failed with an explanation rather than polling forever. Point `DATABASE_URL` at Postgres once a worker and an API process write concurrently — SQLite does not queue concurrent writers, it fails them.
- **Migrations**: the schema is Alembic's (`backend/migrations/`). Both the API and the worker run `upgrade head` at startup; a database written before Phase 7 is stamped at the baseline rather than recreated, so nothing is lost. Autogenerate a script for every model change and read it before committing — the suite fails if the models and the migrations drift apart.
- **Structured logs**: one JSON object per line, each carrying a request id (echoed as the `x-request-id` header) and, inside a training job, the job id.
- **Probes**: `GET /healthz` (liveness), `GET /readyz` (readiness — the database and the schema revision are the hard dependencies; the queue, storage backend, artifact signing key, and LLM key are reported but never fatal, because each degrades to something that still serves), and `GET /metrics` (Prometheus, when `METRICS_ENABLED`). A schema behind the code is a 503: unlike the others it cannot degrade, it just fails at whichever request first needs the missing column.
- **Training cost**: cross-validation multiplies every fit by the fold count, so a run is meaningfully slower than a single-split one — that is the price of a score with a variance attached. `MAX_CANDIDATE_MODELS` caps how many candidates a run may try when a fast answer matters more than an exhaustive one, and the candidate set already adapts to dataset size.
- **Optional model libraries**: LightGBM and imbalanced-learn are pinned in `requirements.txt`. CatBoost is used when it happens to be installed but is not pinned — the wheel is ~100 MB and LightGBM covers similar ground. A missing optional library removes its models from the registry and is recorded as absent in the model card, rather than failing the run.

### Where your data goes

**Your rows stay on this machine.** Earlier versions uploaded the dataset snapshot to the Google Gemini Files API so that Google's sandbox could compute on it. That is gone: analysis code now runs in a local sandboxed subprocess against the snapshot on your disk.

What still leaves the machine is the *conversation* — your question, the schema (column names and dtypes), and the compact text summary of each tool result (for example "mean 48.85, 240 rows, 3 distinct values"). Row-level data is not sent, and neither are the rendered charts. If even column names are sensitive, do not use the chat panel; upload, training, the dashboard, and predictions never contact an LLM at all.

**Retention and deletion.** Every run records an owner and an expiry. `RETENTION_DAYS` sets how long artifacts live; `0` (the default) keeps them until deleted by hand, because silently deleting someone's models on an upgrade would be worse than keeping them. The purge runs at startup and on `POST /api/admin/retention/purge` — deliberately not on an in-process timer, which is one more thing that dies with the process; put it on a cron entry if you want it hourly.

Three ways data leaves for good, all of which remove bytes rather than only rows:

| What | How |
| --- | --- |
| One run | `DELETE /api/runs/{run_key}` — metadata, snapshot, model, signature, and every conversation about it |
| Everything you own | `DELETE /api/account/data?confirm=true` |
| Anything past its expiry | `POST /api/admin/retention/purge`, or a restart |

`GET /api/privacy` returns this policy as JSON, so the UI and the docs cannot drift apart. Usage records (token counts and estimated cost) survive a data deletion: they are the billing record and contain no dataset content.

**Deleting a conversation deletes it with its dataset.** A transcript of questions asked about a dataset is often more revealing than the dataset, so it is never left behind.

### Running it for more than one person

Everything above assumes one trusted operator on localhost, which is the default and stays the default. Going multi-user is opt-in, one setting at a time — nothing here turns on by itself.

```bash
# 1. Authentication. Generate a bootstrap token, then create the first account.
AUTH_ENABLED=true
AUTH_BOOTSTRAP_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

curl -X POST localhost:8000/api/auth/register   -H "X-Bootstrap-Token: $AUTH_BOOTSTRAP_TOKEN"   -H "Content-Type: application/json"   -d '{"email": "you@example.com"}'
# -> {"api_key": "analyst_...", "is_admin": true}
```

The key is shown **once** and cannot be recovered — only its SHA-256 is stored. Registration closes as soon as one account exists; after that an admin creates accounts with `POST /api/admin/users`. Paste the key into the strip at the top of the UI, or send it as `Authorization: Bearer <key>`.

```bash
# 2. A real job queue, so training survives an API restart.
docker compose --profile queue up --build
```

This adds Redis and a worker process. `QUEUE_BACKEND=rq` makes the API hand jobs over instead of running them in its own process; `docker compose restart backend` mid-training then leaves the run untouched. An unreachable Redis falls back to inline execution rather than accepting jobs nothing would ever run — a slower request beats an upload that reports "queued" forever. The worker must share the API's `DATABASE_URL` and artifact volume: it reports progress by writing the same job rows the status endpoint reads.

```bash
# 3. Postgres, once a worker and an API process write at the same time.
docker compose --profile postgres --profile queue up -d
DATABASE_URL=postgresql+psycopg://analyst:analyst@postgres:5432/analyst
```

SQLite has no answer for two writers: the second gets "database is locked" rather than a turn. `postgres://` and `postgresql://` are both rewritten to the pinned psycopg driver, and `DB_POOL_SIZE`/`DB_MAX_OVERFLOW` size the pool (they do nothing on SQLite, which has one file handle). The schema is migrated to head by whichever process starts first. Run the suite against it with `TEST_DATABASE_URL=… pytest`; CI does this on every push.

```bash
# 4. Object storage, so more than one machine can serve the same run.
docker compose --profile storage up -d      # MinIO, and a one-shot bucket creator
STORAGE_BACKEND=s3
S3_BUCKET=your-bucket      # S3_ENDPOINT_URL for MinIO and friends
```

Artifacts are mirrored to the bucket after each run and pulled back on a miss. Credentials come from the usual boto3 chain (environment, profile, instance role) and are deliberately not settings, so they cannot end up in a log line. A misconfigured bucket degrades to local disk rather than taking training down — which is why the bucket gets created for you: a missing one would otherwise look like everything working.

`docker compose --profile replica up -d` adds a second API on port 8001 with **its own empty volume**. Train through the first and predict through the second: the only path the model can have taken is the bucket, and it only loads if the signature the trainer wrote verifies under this process's key. That one request tests shared storage, the shared database, and key portability at once.

```bash
# 5. Observability.
METRICS_ENABLED=true       # GET /metrics, Prometheus exposition
SENTRY_DSN=https://...     # errors only; send_default_pii is off
```

Metrics labels are bounded — method, normalised path, model, outcome — and never include a run key, filename, or email, so `/api/runs/<hash>` collapses to `/api/runs/:id` instead of creating one time series per dataset. `GET /api/usage` reports what the LLM has cost your account and `GET /api/admin/usage` reports it across all of them. **Both are estimates** computed from published list prices, for spotting a runaway session — not an invoice.

Run `python worker.py` directly if you are not using Docker. RQ forks per job on POSIX and cannot on Windows, so use `QUEUE_BACKEND=inline` for local development there.

**Set `ARTIFACT_SIGNING_KEY` before turning any of this on.** Left unset, each process generates its own key — correct on one machine, and silently wrong on two: a bundle signed by the worker fails verification in the API, and the error reads as tampering rather than as a missing variable. `/readyz` reports `checks.artifacts.ok: false` whenever the key is generated and something is shared, and startup logs it once.

### Proving it, rather than assuming it

Every capability above was written and unit-tested against a substitute. **[`docs/production.md`](docs/production.md)** is the runbook that swaps in the real thing and watches it work — in order, one subsystem at a time, because a stack where four things changed at once is a stack whose next failure has four candidate causes.

```bash
python ops/queue_restart_drill.py     # restart the API mid-training; the run must finish
TEST_DATABASE_URL=… pytest            # the whole suite against Postgres
ops/backup.sh && ops/restore.sh …     # then: python ops/verify_restore.py
python ops/loadtest.py --concurrency 8 --requests 24
```

Results go in [`docs/phase7-results.md`](docs/phase7-results.md), which is deliberately unfilled: the drills need a running Docker daemon, and a checkbox ticked without watching the thing happen is worse than an empty one.

### About the sandbox

Generated code runs in a separate one-shot interpreter with a wall-clock timeout, an import denylist, and a PEP 578 audit hook that refuses network access, subprocesses, and file writes. On POSIX it also gets `RLIMIT_AS`/`RLIMIT_CPU`/`RLIMIT_FSIZE`; **Windows has no equivalent**, so there the timeout and output caps are the only ceilings.

That is a restricted execution environment, not a security boundary against a determined attacker — the guards live inside the interpreter they are guarding, and arbitrary Python eventually gets underneath them. It is sound for a single trusted operator, and it is the default.

Set `SANDBOX_BACKEND=container` and it stops being a convention and becomes a boundary: the same runner, in its own container with **no network namespace**, a read-only root, every capability dropped, and exactly one dataset mounted read-only. Build it with `docker build -f ops/Dockerfile.sandbox -t analyst-sandbox:latest ./backend`, then prove it with `python ops/verify_sandbox.py --build` — nine assertions against a real container, two of which are written to fail if the isolation is absent. There is no fallback: with the container backend configured and no runtime reachable, every execution fails rather than silently running unisolated. See [`docs/security.md`](docs/security.md).

---

## 📈 Keeping a Model Honest After It Ships

Every phase before this one judged a model once, at the moment it was trained. Nothing revisited
it. A model that was defensible in August and quietly wrong by November fails exactly the way this
project was written about — confidently, with a number, and nothing saying otherwise. This closes
that gap, and unlike the two sections above it is **on by default**: it only stops throwing away a
record of what a model already did, rather than changing what a caller may do.

- **Every served prediction is logged** — inputs, output, model version, latency — so an answer
  given in August can be reproduced in November. `GET /api/predictions/{id}` returns exactly that.
- **Drift**, measured as Population Stability Index between the training-time distribution and what
  is actually being predicted on, read back from the log. Quantile-binned rather than equal-width,
  which matters: an early version read a same-distribution batch as material drift on nearly every
  trial, because equal-width bins on a skewed feature starve the tail of samples and PSI spends its
  signal there. `GET /api/runs/{run_key}/drift`; `docs/monitoring.md` has the rest, including the
  simulation that set the row-count floor a verdict needs before it means anything.
- **Calibration**: a confidence shown next to a prediction is a claim about how often it is right.
  A calibrated variant is fitted and kept only if it measurably improves the holdout's calibration
  error — never on a marginal change that is as likely to be noise as improvement.
- **Serving-time schema checks** name what actually broke — a rename (`spend_usd → spend`, suggested),
  a genuine absence, or a column that stopped being numeric — instead of silently imputing a training
  median and returning a confident number for a value nobody supplied.
- **Retraining produces a challenger, not a replacement.** It is promoted only if it beats the
  champion on the run's own selection metric, by a margin, on rows neither model was fitted on.
  Every promotion — and every refusal — is audited with both scores, so "the new one is better" is
  never asserted without the numbers that would let someone check it.

See [`docs/monitoring.md`](docs/monitoring.md) for what each setting costs and the endpoints for all
of it.

---

## 🔗 Sharing, Export, and Scheduled Reports

Product work, not remediation — nothing here closes a gap the way Phases 7–9 did. Ownership from
Phase 6 and the audit log from Phase 8 are what make each of these safe to add at all.

- **Read-only share links** (opt-in, `SHARE_LINKS_ENABLED`): a token that grants exactly one thing —
  the same dashboard payload `GET /runs/{run_key}/result` already returns — to whoever holds it, no
  account required. Never chat, prediction, or report generation, because those cost LLM money or
  compute per call and a link is meant to be handed to someone this application has no way to bill or
  rate-limit. Every link expires (`SHARE_LINK_DEFAULT_TTL_DAYS`, capped by `SHARE_LINK_MAX_TTL_DAYS`)
  and can be revoked; `POST /api/runs/{run_key}/share` mints one, `DELETE /api/share/{token}` revokes
  it, `GET /api/share/{token}` is the unauthenticated read.
- **Scheduled reports** (opt-in, `SCHEDULED_REPORTS_ENABLED`, refuses without `SMTP_HOST`): the Phase
  5 report generator, regenerated and emailed on a daily or weekly cadence. No in-process scheduler —
  the same reasoning as retention's purge — `next_run_at` is a due date, and `POST
  /api/report-schedules/run-due` is what an operator's cron entry calls to act on it. Each due
  schedule is handed to the Phase 6 queue rather than run inline, so one slow LLM call cannot make a
  cron tick time out, and a failure of any kind still advances `next_run_at` rather than retrying in a
  tight loop.
- **Export a trained pipeline**, so a model can leave the app it was trained in. A portable bundle
  (`?format=bundle`, always available) zips the signed pickle, a standalone `score.py`, and pinned
  requirements. ONNX (`?format=onnx`) is offered when the run's pipeline allows it — sklearn's
  `TargetEncoder` and `HistGradientBoosting*` have no ONNX converter, and a categorical column trips a
  `SimpleImputer` limitation in `skl2onnx`, so those refuse with a specific reason rather than silently
  producing a graph that scores differently from the model it claims to be.
  `GET /api/runs/{run_key}/export/availability` says which formats apply before a caller downloads one.
- **A split frontend bundle**: `recharts` — most of the single 743 kB chunk the build had been warning
  about — now loads lazily behind the dashboard rather than in the initial page load, which dropped to
  roughly 155 kB. The per-column profile shown before a model exists uses a small hand-drawn SVG
  sparkline instead, the one chart that has to render before that lazy chunk would ever be requested.

Deletion reaches all of it: removing a run or an account also revokes its share links and cancels its
report schedules, the same way it already removes predictions and model versions.

---

## 🔐 Security Posture

- **Path safety**: every artifact key (`run_key`) from a client is validated against `^[a-f0-9]{64}$` and resolved under a known storage root before it is used to load an artifact, so a crafted value cannot reach `joblib.load` on an arbitrary file.
- **Bounded uploads**: uploads stream to a temp file in 1 MB chunks under a byte ceiling (`MAX_UPLOAD_MB`), a row ceiling (`MAX_UPLOAD_ROWS`), and an extension/content-type check. Nothing beyond one chunk is held in memory, and the temp file is removed when the job ends.
- **CORS**: an explicit origin list from `CORS_ALLOW_ORIGINS`; no wildcard, and credentials are disabled until there is an auth story that needs them.
- **Secrets**: `backend/.env` is untracked and gitignored, and `pre-commit` runs `gitleaks` on every commit. Startup logs which *kind* of database is configured rather than the URL, which carries a password once it points at Postgres. Backups (`backups/`) are gitignored: one contains every account and every run.

- **Sandboxed execution**: analysis code runs in a separate one-shot interpreter with a timeout, an import denylist, and an audit hook refusing network, subprocess, and file writes. The child is given a minimal environment, so it cannot read `GEMINI_API_KEY`. With `SANDBOX_BACKEND=container` (opt-in) the same runner instead executes in its own container with no network namespace, a read-only root, no capabilities, and exactly one dataset mounted read-only — an interpreter convention replaced by a kernel boundary.
- **LLM endpoint limits**: per-conversation message rate and token budget, and a bounded agent loop. Per *account* ceilings on model calls, estimated daily spend, stored datasets, disk, and concurrent training jobs (opt-in, `PRINCIPAL_*`) — the per-conversation guards slow one session and stop nothing, because a client that opens a new conversation each time has no ceiling at all.

- **Signed model artifacts**: `joblib.load` executes pickle opcodes, so every model bundle is written with an HMAC-SHA256 sidecar and verified *before* deserialisation. An artifact this application did not write is refused rather than loaded, which closes the path from "someone can drop a file in `models/`" to remote code execution. Path validation was never sufficient on its own: it proves a file is in the right directory, not that it is ours.
- **Authentication and isolation** (opt-in, `AUTH_ENABLED`): a bearer API key per account, stored only as a SHA-256 hash and shown exactly once. Every run, conversation, job, and artifact carries an owner, and every query is scoped to it — including an admin's, because being an admin is authority over accounts, not over their datasets. A run belonging to someone else answers 404, not 403: a run key is a content hash, and confirming that a given hash exists would leak both the dataset and who holds it.
- **No cross-tenant cache sharing**: content-addressed caching means two accounts uploading the same CSV would otherwise resolve to the same run key and share artifacts — a cache hit handing one user the other's dataset snapshot. The owner is part of the cache key, so deduplication stops at the account boundary. This is a deliberate trade: identical files are stored twice.

- **Portable artifact signatures**: `ARTIFACT_SIGNING_KEY` can be set explicitly so every process verifies against the same key, and a bundle signed in one process is checked against another in the suite. Left unset each process generates its own — fine on one machine, and a latent outage on several, so `/readyz` and the startup log both say so rather than waiting for a prediction to fail.

- **Key lifecycle** (opt-in with auth): several keys may be active per account, so rotation is issue → roll out → revoke with no window in which anything is refused. Revocation is a timestamp rather than a delete, because "which key made this request, and when did we stop trusting it" is a question that gets asked after the fact. The last active key cannot be revoked — an account with no working credential cannot issue itself a replacement.
- **Audit log**: an append-only record of privileged and destructive actions — account creation and deactivation, key issuance and revocation, dataset deletion, account erasure, retention purges, refused quota. Outside the retention policy it records, because the entry saying a dataset was deleted has to outlive the dataset, and carrying no dataset content, because the log is read by more people than the data is.
- **Secrets from files**: every credential also accepts a `<NAME>_FILE` variant naming a path, so Docker secrets, Kubernetes secret volumes and systemd credentials can each be pointed at it. An environment variable is readable by anything that can run `docker inspect` or take a core dump; a file is not.

**[`docs/security.md`](docs/security.md)** is the runbook for all of it — what to turn on, in what order, and what each thing costs.

Still open, and honestly: **the Gemini key committed to git history and baked into every backend image built before Phase 7 has not been rotated**, and nothing here fixes a key that has already leaked. There is no password reset or session expiry. No ceiling here is per-IP, so none of them does anything about a flood at an endpoint that needs no key — that belongs at the ingress. And with `AUTH_ENABLED=false` — the default — anyone who can reach the API can read every dataset on it, which is the correct posture for one operator on localhost and the wrong one for anything else.
