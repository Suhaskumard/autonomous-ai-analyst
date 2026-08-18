# 🚀 ADVANCED AUTONOMOUS AI ANALYST

An enterprise-grade, full-stack autonomous data science system. It automatically ingests tabular data, executes an end-to-end ML pipeline, and provides an **Advanced AI Analyst** powered by Gemini with native **Code Execution** capabilities.

![Aesthetic Dashboard](https://img.shields.io/badge/UI-Premium_Glassmorphism-blueviolet)
![Engine](https://img.shields.io/badge/Engine-Gemini_(configurable)-orange)
![Capability](https://img.shields.io/badge/Chat-Code_Execution_Enabled-success) 
 
## 🌟 Key Features
  
- **Advanced AI Chatbot (Gemini-Powered)**:
  - **Native Code Execution**: your dataset snapshot is uploaded to the Gemini Files API and attached to the conversation, so the model's Python runs against the actual rows — not against a path it cannot reach. See *Where your data goes* below.
  - **Contextual Memory**: Remembers past interactions for multi-turn analytical deep-dives.
  - **Dynamic Analysis**: Ask "Find the top 3 correlations" or "Plot the distribution of the target," and the AI handles the logic.
  - **Honest about its reach**: the chat header reports the configured model and whether the answer came from your rows or from summary statistics alone.
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
  routes/
    chat.py       <-- Advanced Gemini Code Execution Engine
    upload.py     <-- Pipeline orchestration & Caching
    predict.py    <-- Inference API (batch + single row)
    profile.py    <-- Profile a CSV without training it
    runs.py       <-- Run registry: list, reopen, compare, delete
    insights.py   <-- Lightweight per-run summary
  ml/             <-- Core Analytical Engine
    preprocess.py <-- Target validation, feature typing (fits nothing)
    train.py      <-- Split-then-fit Pipelines, label encoding
    explain.py    <-- Name-aligned feature attribution
    health.py     <-- Baseline comparison / result verdict
    profile.py    <-- Per-column profile served before training
    diagnostics.py<-- Confusion matrix, residuals, correlations
    quality.py, evaluate.py, ensemble.py
  utils/
    security.py   <-- Artifact-key validation and path resolution
    uploads.py    <-- Bounded, streamed upload handling
  models/         <-- Generated artifacts (gitignored)
    metadata/     <-- Per-run blueprints + data snapshot
    saved_models/ <-- Fitted Pipeline bundles

  db.py           <-- Job + run persistence (SQLite via SQLModel)
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
    Workspace.jsx      <-- Past runs: reopen, compare, delete
    PredictPanel.jsx   <-- CSV or single-row inference, CSV download
    ChatBox.jsx, ErrorBoundary.jsx
    charts/         <-- Recharts components on one validated palette
  hooks/
    useUploadJob.js <-- Cancellable, backed-off, bounded polling
  pages/            <-- Unified Analytical Workflows
  tests/            <-- Vitest + Testing Library

.github/workflows/ci.yml   <-- lint, format, tests, build, image build, secret scan
docker-compose.yml         <-- one-command startup
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
Dependencies are pinned. Edit `backend/requirements.in`, then recompile:
```bash
uv pip compile backend/requirements.in -o backend/requirements.txt
uv pip compile backend/requirements-dev.in -o backend/requirements-dev.txt
```

---

## 🤖 Using the Advanced Analyst

Once a dataset is uploaded, the **Advanced AI Analyst** is your primary partner for exploration:

- **Mathematical Proof**: "What is the standard deviation of the target relative to feature X?"
- **Complex Logic**: "Filter the dataset for rows where X > 50 and then calculate the mean of Y."
- **Visual Description**: "Run a correlation analysis and tell me which features have a high coefficient."

The Analyst doesn't just "guess"—it writes **Python code** using `pandas` to provide ground-truth answers from your specific CSV snapshot.

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
- **Durable jobs**: job state and the run registry live in SQLite (`DATABASE_URL`), so a restart does not lose them and multiple workers see the same rows. A job interrupted by a restart is marked failed with an explanation rather than polling forever.
- **Structured logs**: one JSON object per line, each carrying a request id (echoed as the `x-request-id` header) and, inside a training job, the job id.
- **Probes**: `GET /healthz` (liveness) and `GET /readyz` (readiness — checks the database and reports whether an LLM key is configured).

### Where your data goes

When you use the chatbot, the dataset snapshot for that run is **uploaded to the Google Gemini Files API** and referenced by the model's code-execution sandbox. That is what makes its arithmetic real and verifiable — and it means the rows leave your machine. Google retains uploaded files for roughly 48 hours. If that is not acceptable for a given dataset, do not use the chat panel for it; upload and training are entirely local. A fully server-side executor is planned for a later phase.

---

## 🔐 Security Posture

- **Path safety**: every artifact key (`run_key`) from a client is validated against `^[a-f0-9]{64}$` and resolved under a known storage root before it is used to load an artifact, so a crafted value cannot reach `joblib.load` on an arbitrary file.
- **Bounded uploads**: uploads stream to a temp file in 1 MB chunks under a byte ceiling (`MAX_UPLOAD_MB`), a row ceiling (`MAX_UPLOAD_ROWS`), and an extension/content-type check. Nothing beyond one chunk is held in memory, and the temp file is removed when the job ends.
- **CORS**: an explicit origin list from `CORS_ALLOW_ORIGINS`; no wildcard, and credentials are disabled until there is an auth story that needs them.
- **Secrets**: `backend/.env` is untracked and gitignored, and `pre-commit` runs `gitleaks` on every commit.

Still open (later phases): there is no authentication or rate limiting on any route, including the paid LLM endpoint. Run this on a trusted network only.
