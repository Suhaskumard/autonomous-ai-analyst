# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

# libgomp is the OpenMP runtime xgboost links against; curl is for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is cached until the pinned set changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Artifacts and the SQLite database live here; compose mounts a volume over it.
RUN mkdir -p models/metadata models/saved_models \
    && useradd --create-home --uid 10001 analyst \
    && chown -R analyst:analyst /app
USER analyst

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
