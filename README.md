<div align="center">

# 🚀 Autonomous AI Analyst

**An enterprise-grade, full-stack autonomous data science platform.**

Automatically ingest tabular data, execute an end-to-end ML pipeline, and interact with an Advanced AI Analyst powered by Gemini API with native sandboxed code execution capabilities.

![Platform](https://img.shields.io/badge/Platform-Full_Stack-lightblue)
![Engine](https://img.shields.io/badge/Engine-Gemini_API-orange)
![Execution](https://img.shields.io/badge/Execution-Sandboxed-green)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

[📖 Features](#-key-features) • [🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#-project-architecture) • [⚙️ Configuration](#-configuration) • [📚 Documentation](#-documentation)

</div>

---

## 🌟 What is the Autonomous AI Analyst?

The **Autonomous AI Analyst** bridges the gap between raw data and actionable insights. Unlike traditional chatbots that merely summarize information, this platform empowers you with an AI partner that **actively writes, validates, and executes code** to analyze your data.

**Key Philosophy:** Your data never leaves your machine. All analysis happens locally in a sandboxed environment. Only schema information and query summaries are sent to the Gemini API—sensitive rows remain completely private.

### Perfect For:
- **Data Scientists:** Accelerate exploratory data analysis (EDA) and model development
- **Business Analysts:** Generate insights without writing code
- **Compliance Teams:** Deploy AI with complete data privacy and audit trails
- **Enterprises:** Production-ready ML with monitoring, versioning, and multi-user support

---

## ✨ Key Features

### 🤖 Intelligent Autonomous Analyst
- **Sandboxed Code Execution:** All generated pandas/numpy code runs in an isolated subprocess on your machine
- **Data Privacy by Default:** Tabular rows never transmitted to cloud APIs—only schema and query context
- **Show Your Work:** Every analysis includes executed code, resulting data, and visualizations
- **Automatic EDA:** Single-click generation of comprehensive profile reports covering columns, distributions, correlations, and relationships
- **Interactive Refinement:** Ask follow-up questions with full conversation context preserved

### 🧠 Enterprise-Grade ML Pipeline
- **Data Leak Prevention:** Train/test splits enforced *before* model fitting; preprocessing and modeling unified in sklearn Pipelines
- **Intelligent Model Selection:** 
  - Automatic algorithm selection (Linear, Tree-based, Neural Net)
  - Ensemble methods combining multiple models
  - Hyperparameter tuning via Randomized Search with configurable computational budget
- **Imbalanced Data Handling:** Automatic detection of skewed targets with class weighting or in-pipeline SMOTE
- **Rigorous Evaluation:**
  - Cross-validated metrics on training data
  - Strict 20% holdout set for final evaluation
  - Baseline comparison (naive model as reference)
- **Feature Explainability:** Built-in SHAP feature importances and detailed model cards for every training run

### 🛡️ Security & Privacy-First Architecture
- **Local-First Processing:** Dataset rows and generated charts remain on user's machine
- **Sandboxed Execution:** AI-generated code runs in restricted subprocess with network/file access controls
- **Multi-User Support:** Optional authentication with per-account rate limits and spend controls
- **Audit Logging:** Comprehensive audit trail for compliance (when AUTH_ENABLED)
- **Data Retention Control:** Configurable automatic purging of old runs and artifacts
- **Artifact Signing:** Cryptographic verification ensures model integrity across distributed deployments

### 📈 Production-Ready Monitoring & MLOps
- **Continuous Prediction Monitoring:**
  - Population Stability Index (PSI) for drift detection
  - Calibration error tracking for model reliability
  - Comprehensive prediction logging
- **Model Versioning:** Champion/challenger workflows with safe promotion pipelines
- **Flexible Export:**
  - ONNX format for portable model deployment
  - Self-contained bundles for air-gapped environments
  - Read-only share links for stakeholder review
- **Scheduled Reports:** Generate and email periodic analysis reports
- **Metrics & Observability:** Prometheus metrics export, Sentry error tracking, structured JSON logging

### 🏗️ Scalable Architecture
- **Multiple Deployment Profiles:** Single-machine Docker, distributed queue (Redis), object storage (MinIO/S3), PostgreSQL
- **Async Processing:** Optional job queue for long-running training tasks (Redis + RQ)
- **Production Databases:** SQLite for dev, PostgreSQL for production
- **Object Storage:** Local filesystem, MinIO, or AWS S3
- **Multi-API Support:** Gemini (default), extensible to other LLM providers

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose (recommended) OR Python 3.11+ with Node.js 18+
- Gemini API key (get free at [Google AI Studio](https://aistudio.google.com/))

### Option 1: Docker (Recommended) — 2 Minutes

**1. Set your API key:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**2. Start the stack:**
```bash
docker compose up --build
```

**3. Open in browser:**
- 🎨 Frontend: http://localhost:5173
- ⚙️ API: http://localhost:8000
- 📊 Health Check: http://localhost:8000/readyz

The entire stack (FastAPI backend, React frontend, SQLite database, and model storage) runs in containers with persistent volumes.

### Option 2: Local Development Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key-here"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend** (in another terminal):
```bash
cd frontend
npm install
npm run dev
```

Access at http://localhost:5173

### 🔧 Docker Compose Profiles

Enable additional features with profiles:

```bash
# Development (default) — single machine, SQLite, in-process training
docker compose up

# With async job queue (Redis + worker)
docker compose --profile queue up

# With PostgreSQL instead of SQLite
docker compose --profile postgres up

# With MinIO object storage
docker compose --profile storage up

# Run multiple API replicas (requires queue + storage)
docker compose --profile replica up

# Full hardened stack (all profiles + security hardening)
docker compose -f docker-compose.yml -f docker-compose.hardened.yml --profile queue --profile postgres --profile storage up
```

---

## 🏗️ Project Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ - Chat Interface | Dataset Upload | Report Generation   │    │
│  │ - Model Monitoring | Share Links | Scheduled Reports    │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│              Backend API (FastAPI - Python 3.11)                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Routes: Chat | Upload | Predict | Profile | Monitoring │    │
│  │ Admin: Auth | Reports | Audit Logs | Health Checks     │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌──────────────────┬──────────────────┬────────────────────┐   │
│  │ Analyst Agent    │ ML Pipeline      │ Sandbox Executor   │   │
│  │ (LLM Interface)  │ (scikit-learn)   │ (subprocess)       │   │
│  └──────────────────┴──────────────────┴────────────────────┘   │
└──────┬──────────────────┬─────────────────────┬──────────────────┘
       │                  │                     │
       ▼                  ▼                     ▼
   ┌────────────┐   ┌──────────────┐    ┌────────────────┐
   │ Database   │   │ Artifact     │    │ Job Queue      │
   │ SQLite/PG  │   │ Storage      │    │ Redis/inline   │
   │            │   │ Local/S3     │    │ (async jobs)   │
   └────────────┘   └──────────────┘    └────────────────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
   Trained Models    Exports (ONNX)    Share Bundles
```

### Directory Structure

```
autonomous-ai-analyst/
│
├── backend/                          # Python FastAPI Application
│   ├── analyst/                      # Core AI Analyst Engine
│   │   ├── agent.py                 # LLM agent orchestration
│   │   ├── container.py             # Dependency injection container
│   │   ├── providers.py             # LLM provider abstraction
│   │   ├── sandbox.py               # Sandboxed execution environment
│   │   ├── runner.py                # Async job execution
│   │   ├── tools.py                 # Available tools for LLM
│   │   └── report.py                # Automated report generation
│   │
│   ├── routes/                       # API Endpoints
│   │   ├── chat.py                  # LLM chat interface
│   │   ├── upload.py                # Dataset upload & validation
│   │   ├── predict.py               # Prediction endpoint
│   │   ├── profile.py               # Data profiling
│   │   ├── runs.py                  # ML run management
│   │   ├── monitoring.py            # Prediction monitoring
│   │   ├── export.py                # Model export
│   │   ├── share.py                 # Share link management
│   │   ├── report_schedules.py      # Scheduled reports
│   │   ├── admin.py                 # Admin/auth endpoints
│   │   └── insights.py              # Automated insights
│   │
│   ├── ml/                           # Machine Learning Pipeline
│   │   ├── train.py                 # Model training orchestration
│   │   ├── evaluate.py              # Cross-validation & metrics
│   │   ├── tuning.py                # Hyperparameter optimization
│   │   ├── ensemble.py              # Ensemble model creation
│   │   ├── preprocess.py            # Feature preprocessing
│   │   ├── imbalance.py             # Imbalanced data handling (SMOTE)
│   │   ├── explain.py               # SHAP feature importances
│   │   ├── quality.py               # Data quality checks
│   │   ├── drift.py                 # Data drift detection (PSI)
│   │   ├── calibration.py           # Model calibration metrics
│   │   ├── diagnostics.py           # Model diagnostics & analysis
│   │   ├── profile.py               # Data profiling
│   │   ├── export.py                # ONNX export
│   │   ├── model_card.py            # Model cards & documentation
│   │   ├── health.py                # Model health checks
│   │   └── serving_schema.py        # Prediction input validation
│   │
│   ├── models/                       # Trained Artifacts Storage
│   │   ├── saved_models/            # Serialized sklearn models
│   │   └── metadata/                # Model metadata JSON
│   │
│   ├── migrations/                   # Alembic Database Migrations
│   │   └── versions/                # Individual migration files
│   │
│   ├── tests/                        # Comprehensive Test Suite
│   │   ├── test_analyst.py          # Agent behavior tests
│   │   ├── test_api_flow.py         # End-to-end API tests
│   │   ├── test_calibration.py      # Calibration logic tests
│   │   ├── test_drift.py            # Drift detection tests
│   │   ├── test_leakage.py          # Data leak prevention tests
│   │   ├── test_production_path.py  # Production workflow tests
│   │   ├── test_security.py         # Security/auth tests
│   │   └── ...                      # 20+ additional test files
│   │
│   ├── main.py                      # FastAPI app initialization
│   ├── db.py                        # Database connection & queries
│   ├── config.py                    # Settings management
│   ├── auth.py                      # Authentication & authorization
│   ├── exceptions.py                # Custom exceptions
│   ├── logging_config.py            # Structured logging setup
│   ├── monitoring.py                # Prometheus metrics
│   ├── observability.py             # Sentry & tracing
│   ├── lifecycle.py                 # Job/data lifecycle management
│   ├── worker.py                    # Async job worker (RQ)
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Container image
│   └── alembic.ini                  # Migration configuration
│
├── frontend/                         # React + Vite Application
│   ├── src/                         # Source code
│   │   ├── components/              # Reusable React components
│   │   ├── pages/                   # Page components
│   │   ├── hooks/                   # Custom React hooks
│   │   ├── services/                # API client services
│   │   └── App.jsx                  # Root component
│   ├── index.html                   # HTML entry point
│   ├── vite.config.js               # Vite build config
│   ├── package.json                 # Dependencies & scripts
│   ├── Dockerfile                   # Container image
│   └── nginx.conf                   # Production web server config
│
├── ops/                              # Operations & Deployment
│   ├── backup.sh                    # Database backup script
│   ├── restore.sh                   # Database restore script
│   ├── loadtest.py                  # Load testing utility
│   ├── queue_restart_drill.py       # Job queue resilience test
│   ├── verify_sandbox.py            # Sandbox isolation verification
│   ├── verify_restore.py            # Backup/restore verification
│   ├── Dockerfile.sandbox           # Hardened container image
│   └── seccomp-analyst.json         # System call whitelist
│
├── docs/                             # Documentation
│   ├── production.md                # Production deployment guide
│   ├── security.md                  # Security hardening guide
│   ├── monitoring.md                # Monitoring & alerting setup
│   └── phase7-results.md            # Feature release notes
│
├── docker-compose.yml               # Default dev/single-machine stack
├── docker-compose.hardened.yml      # Hardened production stack
├── pyproject.toml                   # Python tooling configuration
└── README.md                        # This file
```

---

## ⚙️ Configuration

### Environment Variables

**Core Configuration:**
```bash
# LLM Provider
GEMINI_API_KEY=sk-...                      # Required: Gemini API key
GEMINI_MODEL=gemini-1.5-flash-8b          # Default: flash model (cost-effective)

# Database
DATABASE_URL=sqlite:////app/models/db      # SQLite (dev) or postgres:// (prod)
DB_POOL_SIZE=5                             # Connection pool size
DB_MAX_OVERFLOW=10                         # Pool overflow connections

# Logging
LOG_LEVEL=INFO                             # DEBUG, INFO, WARNING, ERROR
LOG_JSON=true                              # Structured JSON logging
```

**Authentication & Security:**
```bash
AUTH_ENABLED=false                         # Multi-user auth (off by default)
AUTH_BOOTSTRAP_TOKEN=secret-token          # Initial admin token
ARTIFACT_SIGNING_KEY=...                   # 32-byte hex for bundle verification
PRINCIPAL_MAX_LLM_CALLS_PER_HOUR=60       # Rate limit per account
PRINCIPAL_DAILY_SPEND_LIMIT_USD=5.00       # Daily spend limit per account
PRINCIPAL_MAX_RUNS=25                      # Max stored datasets per account
```

**Queue & Storage:**
```bash
QUEUE_BACKEND=inline                       # inline (single machine) or rq (Redis)
REDIS_URL=redis://localhost:6379/0        # Redis connection (if using queue)
STORAGE_BACKEND=local                      # local, s3, or minio
S3_BUCKET=my-bucket                        # S3 bucket for artifacts
S3_REGION=us-east-1                        # AWS region
```

**Features:**
```bash
MAX_UPLOAD_MB=50                           # Max file upload size
MAX_UPLOAD_ROWS=1000000                    # Max rows per dataset
RETENTION_DAYS=0                           # Auto-purge age (0=never)
SHARE_LINKS_ENABLED=true                   # Enable shareable reports
SCHEDULED_REPORTS_ENABLED=true             # Enable email scheduling
METRICS_ENABLED=true                       # Prometheus metrics export
SENTRY_DSN=https://...                     # Error tracking (optional)
```

**CORS:**
```bash
CORS_ALLOW_ORIGINS=http://localhost:5173  # Frontend origin
```

See `.env.example` in backend for complete reference. For production deployments, use `.env.production.example`.

---

## 📦 Dependencies

### Backend (Python 3.11+)
- **Web:** FastAPI, Uvicorn, SQLModel, Alembic
- **ML:** scikit-learn, LightGBM, imbalanced-learn, SHAP
- **Data:** pandas, numpy, scipy
- **LLM:** google-genai (Gemini API)
- **Storage:** boto3 (AWS S3), minio
- **Queue:** RQ (Redis Queue)
- **Database:** SQLAlchemy, psycopg2 (PostgreSQL)
- **Monitoring:** Prometheus, Sentry
- **Visualization:** matplotlib, Recharts (frontend)

### Frontend (Node.js 18+)
- React 18, Vite 5
- Tailwind CSS, Lucide Icons
- Recharts (data visualization)
- Testing: Vitest, React Testing Library

See [backend/requirements.txt](backend/requirements.txt) and [frontend/package.json](frontend/package.json) for complete dependency lists.

---

## 🧪 Testing

### Run All Tests
```bash
# Backend tests
cd backend
pip install -r requirements-dev.txt
pytest -v

# Frontend tests
cd frontend
npm test
```

### Key Test Suites
- **Data Leak Prevention:** `test_leakage.py` — Verifies train/test split integrity
- **Security:** `test_security.py` — Auth, audit logs, rate limiting
- **API Integration:** `test_api_flow.py` — End-to-end workflows
- **ML Pipeline:** `test_calibration.py`, `test_drift.py` — Model evaluation
- **Sandbox:** `test_sandbox_isolation.py` — Code execution safety

---

## 📊 API Endpoints

### Chat & Analysis
- `POST /api/chat` — Send message to analyst (code execution enabled)
- `GET /api/chat/{conversation_id}` — Retrieve conversation history

### Data Management
- `POST /api/upload` — Upload CSV/Parquet dataset
- `GET /api/runs` — List trained models
- `GET /api/runs/{run_id}` — Get model details

### Predictions
- `POST /api/predict` — Make predictions with trained model
- `GET /api/monitoring/predictions` — Prediction history & drift

### Reports
- `POST /api/report` — Generate automated analysis report
- `POST /api/report_schedules` — Schedule recurring reports
- `POST /api/share` — Create read-only share link

### Admin
- `GET /api/readyz` — Health check & dependency status
- `POST /api/admin/audit` — Audit log (if AUTH_ENABLED)
- `POST /api/account/keys` — Manage API keys (if AUTH_ENABLED)

See [OpenAPI docs](http://localhost:8000/docs) for interactive exploration.

---

## 🚨 Production Deployment

For production, follow the hardening checklist in [docs/production.md](docs/production.md):

1. **Use PostgreSQL** instead of SQLite for concurrency
2. **Enable Redis queue** for async job processing
3. **Add object storage** (MinIO/S3) for artifact durability
4. **Set artifact signing key** for multi-process deployments
5. **Enable authentication** for multi-user access
6. **Configure rate limits** per account
7. **Enable audit logging** for compliance
8. **Use hardened containers** with seccomp profiles

Start with a single feature at a time and verify `/readyz` endpoint.

---

## 🔒 Security Hardening

For detailed security guidance, see [docs/security.md](docs/security.md):

- **Rate Limiting:** Per-account API call budgets
- **Sandboxing:** Restricted subprocess execution
- **Audit Logs:** Complete trail of user actions (when AUTH enabled)
- **Data Retention:** Automatic purging of old artifacts
- **Artifact Signing:** Cryptographic bundle verification
- **Container Hardening:** Seccomp profiles, read-only filesystems
- **Secrets Management:** Environment variable substitution via `*_FILE`

---

## 🛠 Development

### Running Locally

**Terminal 1 - Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
export GEMINI_API_KEY="your-key"
uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Terminal 3 - Optional Worker:**
```bash
export QUEUE_BACKEND=rq
export REDIS_URL=redis://localhost:6379/0
python backend/worker.py
```

### Code Quality
- **Linting:** Ruff (configured in pyproject.toml)
- **Testing:** pytest with strict markers
- **Formatting:** Black (via Ruff)

### Database Migrations
```bash
cd backend
alembic upgrade head         # Apply migrations
alembic revision --autogenerate -m "Description"  # Create migration
```

---

## 📚 Documentation

Comprehensive documentation is available in the [docs/](docs/) folder:

- **[production.md](docs/production.md)** — Multi-machine deployment with queue, storage, and Postgres
- **[security.md](docs/security.md)** — Authentication, rate limits, sandboxing, audit logs
- **[monitoring.md](docs/monitoring.md)** — Prometheus metrics, Sentry errors, performance tuning
- **[phase7-results.md](docs/phase7-results.md)** — Latest feature releases and improvements

---

## 🤝 Contributing

Contributions welcome! Please ensure:
1. Tests pass: `pytest` (backend), `npm test` (frontend)
2. Code passes linting: `ruff check backend/`
3. Clear commit messages explaining changes
4. Updated tests for new features

---

## 📄 License

[MIT License](LICENSE) — See LICENSE file for details

---

## 🆘 Support & Issues

- **Bug Reports:** Open an issue with reproduction steps
- **Feature Requests:** Describe use case and expected behavior
- **Questions:** Check [docs/](docs/) first, then open a discussion

---

## 🎯 Roadmap & Feature Status

### ✅ Completed (Phase 7)
- Sandboxed code execution
- Multi-user authentication
- Production monitoring & drift detection
- Scheduled reports & email delivery
- Model versioning & registry
- Share links & export (ONNX)
- Complete test coverage

### 🔄 In Development
- Additional LLM providers (Claude, GPT-4, Llama)
- Advanced visualization templates
- Real-time collaboration features
- Custom model deployment pipelines

### 📋 Planned
- Auto-ML for image/NLP tasks
- Causal inference framework
- Advanced ensemble techniques
- Kubernetes-native deployment

---

<div align="center">

**Built with ❤️ for data science teams that care about privacy, reproducibility, and enterprise readiness.**

[GitHub](https://github.com/) • [Documentation](docs/) • [Issues](https://github.com/issues)

</div>
│   ├── ml/                 # Scikit-learn pipelines, tuning, diagnostics
│   ├── routes/             # FastAPI endpoints
│   ├── models/             # Generated artifacts (gitignored)
│   ├── db.py               # SQLModel Database
│   └── worker.py           # RQ Background Worker
│
├── frontend/               # React UI
│   ├── src/components/     # Dashboard, Data Profile, ChatBox
│   ├── src/styles.css      # Premium Glassmorphism Design System
│   └── src/pages/          # Analytical Workflows
│
├── ops/                    # Production drills, backups, and load tests
└── docs/                   # Detailed documentation
```

---

## 📖 Documentation

For deep dives into operational readiness, security, and monitoring, please consult the docs:

- **[Security Posture](docs/security.md)**: Details on the sandbox, data privacy, and access controls.
- **[Production Runbook](docs/production.md)**: Guides for backups, restores, scaling, and load testing.
- **[Monitoring & Drift](docs/monitoring.md)**: How the system evaluates model decay and data drift over time.
- **[Phase 7 Results](docs/phase7-results.md)**: Verifiable outcomes from our production drills.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
