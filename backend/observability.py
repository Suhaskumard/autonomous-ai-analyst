"""Metrics, error reporting, and what the LLM actually cost.

Three things an operator needs that logs alone do not give: a number they can
graph, an alert when something throws, and an answer to "why is the bill that
size". All three degrade to nothing rather than failing — a missing Sentry DSN
or prometheus_client should never stop the app serving.

Cost is *estimated*, and labelled as such everywhere it surfaces. Providers
bill on their own token accounting, not ours, and published prices move. The
figure is for spotting a runaway session, not for reconciling an invoice.
"""

import logging

from config import settings

logger = logging.getLogger(__name__)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# --- LLM cost ---------------------------------------------------------------

#: USD per 1M tokens, (input, output). Published list prices, which move — the
#: point is an order of magnitude for spotting a runaway session, not billing.
#: An unknown model falls back to DEFAULT_RATE rather than reporting zero,
#: because "free" is the one answer that is definitely wrong.
MODEL_RATES: dict[str, tuple[float, float]] = {
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}
DEFAULT_RATE = (0.30, 2.50)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD for one call. See MODEL_RATES on why this is an estimate."""
    name = (model or "").lower()
    rate = MODEL_RATES.get(name)
    if rate is None:
        # Match on prefix so "gemini-1.5-flash-002" bills as "gemini-1.5-flash".
        for known, known_rate in MODEL_RATES.items():
            if name.startswith(known):
                rate = known_rate
                break
    rate = rate or DEFAULT_RATE
    return round((input_tokens / 1_000_000) * rate[0] + (output_tokens / 1_000_000) * rate[1], 6)


# --- metrics ----------------------------------------------------------------


class _Metrics:
    """Prometheus collectors, or inert stand-ins when the library is absent."""

    def __init__(self):
        self.enabled = PROMETHEUS_AVAILABLE and settings.metrics_enabled
        if not self.enabled:
            return

        self.registry = CollectorRegistry()
        self.requests = Counter(
            "analyst_http_requests_total",
            "HTTP requests handled.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "analyst_http_request_seconds",
            "HTTP request duration.",
            ["method", "path"],
            registry=self.registry,
        )
        self.trainings = Counter(
            "analyst_training_runs_total",
            "Training runs, by outcome.",
            ["outcome"],
            registry=self.registry,
        )
        self.training_seconds = Histogram(
            "analyst_training_seconds",
            "Training run duration.",
            registry=self.registry,
            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
        )
        self.llm_calls = Counter(
            "analyst_llm_calls_total",
            "LLM calls, by provider and model.",
            ["provider", "model"],
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "analyst_llm_tokens_total",
            "LLM tokens, by direction.",
            ["provider", "model", "direction"],
            registry=self.registry,
        )
        self.llm_cost = Counter(
            "analyst_llm_estimated_cost_usd_total",
            "Estimated LLM spend in USD. An estimate, not an invoice.",
            ["provider", "model"],
            registry=self.registry,
        )
        self.sandbox_runs = Counter(
            "analyst_sandbox_executions_total",
            "Sandboxed code executions, by outcome.",
            ["outcome"],
            registry=self.registry,
        )
        self.artifact_bytes = Gauge(
            "analyst_artifact_bytes",
            "Bytes of local artifact storage in use.",
            registry=self.registry,
        )
        self.disk_free_bytes = Gauge(
            "analyst_disk_free_bytes",
            "Free bytes on the artifact volume.",
            registry=self.registry,
        )

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        if not self.enabled:
            return
        self.requests.labels(method=method, path=path, status=str(status)).inc()
        self.request_seconds.labels(method=method, path=path).observe(seconds)

    def observe_training(self, outcome: str, seconds: float | None = None) -> None:
        if not self.enabled:
            return
        self.trainings.labels(outcome=outcome).inc()
        if seconds is not None:
            self.training_seconds.observe(seconds)

    def observe_llm(self, provider: str, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        if not self.enabled:
            return
        self.llm_calls.labels(provider=provider, model=model).inc()
        self.llm_tokens.labels(provider=provider, model=model, direction="input").inc(input_tokens)
        self.llm_tokens.labels(provider=provider, model=model, direction="output").inc(output_tokens)
        self.llm_cost.labels(provider=provider, model=model).inc(cost)

    def observe_sandbox(self, outcome: str) -> None:
        if not self.enabled:
            return
        self.sandbox_runs.labels(outcome=outcome).inc()

    def render(self) -> bytes:
        if not self.enabled:
            return b""
        from utils.storage import disk_usage_bytes, free_disk_bytes

        # Sampled at scrape time rather than tracked incrementally: cheap here,
        # and it cannot drift away from what is actually on disk.
        self.artifact_bytes.set(disk_usage_bytes())
        self.disk_free_bytes.set(free_disk_bytes())
        return generate_latest(self.registry)


metrics = _Metrics()


def normalise_path(path: str) -> str:
    """Collapse ids out of a path so the label set stays bounded.

    `/api/runs/<64 hex>` and `/api/upload/status/<uuid>` would otherwise create
    one time series per dataset, which is how a metrics endpoint becomes a
    memory leak.
    """
    parts = []
    for part in path.strip("/").split("/"):
        if len(part) >= 12 and all(c in "0123456789abcdef-" for c in part.lower()):
            parts.append(":id")
        else:
            parts.append(part)
    return "/" + "/".join(parts)


def record_llm_usage(
    owner_id: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    run_key: str | None = None,
    conversation_id: str | None = None,
) -> float:
    """Persist and count one LLM call. Returns the estimated cost."""
    import db

    cost = estimate_cost_usd(model, input_tokens, output_tokens)
    try:
        db.record_usage(
            owner_id=owner_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            run_key=run_key,
            conversation_id=conversation_id,
        )
    except Exception as exc:  # accounting must never break the answer
        logger.warning("Could not record LLM usage: %s", exc)
    metrics.observe_llm(provider, model, input_tokens, output_tokens, cost)
    return cost


# --- error reporting --------------------------------------------------------


def init_sentry() -> bool:
    """Wire up Sentry when a DSN is configured. Absence is not an error."""
    dsn = (settings.sentry_dsn or "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            # Datasets are the product here; sending them to an error tracker
            # would move user data somewhere the privacy note does not cover.
            send_default_pii=False,
        )
        logger.info("Sentry error reporting enabled")
        return True
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed; error reporting is off.")
    except Exception as exc:  # pragma: no cover - bad DSN
        logger.warning("Could not initialise Sentry: %s", exc)
    return False
