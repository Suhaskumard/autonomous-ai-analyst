"""What happened to this model after training day.

Every phase before this one judged a model at the moment it was fitted. The
holdout score, the health verdict, the model card — all of them describe a
model on the day it was born, and until now nothing revisited any of it. A model
that was defensible in August and quietly wrong by November fails exactly the
way this project was written about: confidently, with a number, and with nothing
saying otherwise.

This module is the "afterwards". It holds three things:

* **The prediction log.** Every served answer, with the inputs it was made from
  and the model version that made it. That is what turns "the model said 0.83
  in August" into something anyone can go and check.
* **Drift.** The training-time distribution against the inputs actually being
  sent now, read back out of that log. See `ml/drift.py` for the measure and its
  limits.
* **The monitoring view.** Volume, latency, drift status, calibration, the
  version history, and the model card — the card already existed and had
  nowhere to live once training finished.

Two rules run through all of it.

**Monitoring never breaks serving.** Every write here is best-effort and every
failure is swallowed and logged. A prediction that succeeded and was not
recorded is a gap in a chart; a prediction that failed because the log was full
is an outage. The application already makes this trade for the audit log, and
makes it the same way here.

**Logged inputs are as sensitive as the rows they came from.** They are
owner-scoped, they are covered by the retention policy, they are disclosed at
`/api/privacy`, and they are deleted with the run rather than kept after it. The
audit log is deliberately outside retention because it must outlive what it
records; this is deliberately inside it, because it describes a model that will
not.
"""

import logging
import uuid
from datetime import timedelta
from typing import Any

import pandas as pd

import db
from config import settings
from logging_config import request_id_var
from ml import drift
from utils.helpers import now_utc

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(settings.prediction_logging_enabled)


# --- writing -----------------------------------------------------------------


def log_prediction_batch(
    run_key: str,
    owner_id: str,
    model_version: str | None,
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
    latency_ms: int,
) -> int:
    """Record what was just served. Returns how many rows were stored.

    `frame` is the aligned feature frame the model actually saw — not the file
    the caller uploaded — so what is logged is what was predicted from, columns
    the model ignores included nowhere. `rows` is the response's per-row output.

    Never raises. See the module docstring: a prediction that succeeded and was
    not recorded is a gap in a chart, and that is the cheaper failure.
    """
    if not enabled():
        return 0

    try:
        return db.log_predictions(_build_rows(run_key, owner_id, model_version, frame, rows, latency_ms))
    except Exception:
        logger.exception("Could not assemble prediction log rows", extra={"run_key": run_key})
        return 0


def _build_rows(
    run_key: str,
    owner_id: str,
    model_version: str | None,
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
    latency_ms: int,
) -> list[dict[str, Any]]:
    total = min(len(frame), len(rows))
    cap = settings.prediction_log_max_rows
    sampled = total > cap

    if sampled:
        # Evenly spaced rather than the first N. A batch is often sorted —
        # by date, by customer, by whatever the extract was ordered on — and
        # taking its head would give a drift comparison a systematically
        # unrepresentative sample and no way to know it.
        step = total / cap
        positions = [int(index * step) for index in range(cap)]
    else:
        positions = list(range(total))

    request_id = request_id_var.get() or str(uuid.uuid4())
    moment = now_utc()
    records: list[dict[str, Any]] = []

    for position in positions:
        row = rows[position]
        prediction = row.get("prediction")
        records.append(
            {
                "prediction_id": uuid.uuid4().hex,
                "request_id": request_id,
                "run_key": run_key,
                "owner_id": owner_id,
                "model_version": model_version,
                "created_at": moment,
                "latency_ms": int(latency_ms),
                "inputs": _row_inputs(frame, position),
                "prediction": None if prediction is None else str(prediction),
                "prediction_value": _as_float(prediction),
                "confidence": row.get("confidence"),
                "sampled": sampled,
            }
        )
    return records


def _row_inputs(frame: pd.DataFrame, position: int) -> dict[str, Any]:
    """One row of features as JSON-safe values, or nothing when logging is off.

    `PREDICTION_LOG_INPUTS=false` keeps the volume, latency and outcome record
    and drops the feature values. That is a real privacy choice and it costs
    drift detection entirely, which is why the setting says so where it is
    defined.
    """
    if not settings.prediction_log_inputs:
        return {}
    record = frame.iloc[position].to_dict()
    return {str(column): _jsonable(value) for column, value in record.items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:  # pragma: no cover - exotic dtype
            pass
    if pd.isna(value):
        return None
    return str(value)


def _as_float(value: Any) -> float | None:
    """The numeric form of a prediction, when there is one.

    A regression output and a numerically-labelled class both land here, which
    is fine: `prediction` keeps the authoritative text and this is for charting.
    """
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# --- reading -----------------------------------------------------------------


def drift_report(run_key: str, metadata: dict, owner_id: str | None = None) -> dict:
    """Compare the inputs recently served against the training distribution.

    Two different "nothing to compare" cases are kept apart on purpose. No
    predictions served yet will resolve itself the moment one is — that is
    `drift.compare`'s own `insufficient_data` path, reached by handing it an
    empty frame and letting the row count speak. Predictions served *without*
    their inputs never resolves on its own no matter how many arrive, because
    `PREDICTION_LOG_INPUTS=false` is a standing decision, not a backlog — that
    case is reported as `unavailable` explicitly rather than left to look like
    "just wait", which is what handing it to `compare()` the same way as the
    first case would imply.
    """
    reference = drift.reference_from_metadata(metadata)
    logged = db.list_predictions(run_key, limit=settings.drift_sample_size, owner_id=owner_id)

    frames = [entry["inputs"] for entry in logged if entry.get("inputs")]
    if not frames:
        if not logged:
            return drift.compare(reference, pd.DataFrame(), min_rows=settings.drift_min_rows)
        return {
            "status": drift.UNAVAILABLE,
            "reason": _no_inputs_reason(),
            "rows_compared": len(logged),
            "min_rows": settings.drift_min_rows,
            "columns": [],
        }

    return drift.compare(reference, pd.DataFrame(frames), min_rows=settings.drift_min_rows)


def _no_inputs_reason() -> str:
    if not settings.prediction_log_inputs:
        return (
            "Predictions are being logged without their inputs (PREDICTION_LOG_INPUTS=false), so there is "
            "no live distribution to compare. Volume and latency are still recorded."
        )
    return "The logged predictions carry no feature values, so there is nothing to compare."


def overview(run_key: str, metadata: dict, owner_id: str | None = None) -> dict:
    """Everything the monitoring view shows for one model.

    Assembled here rather than in the route so the same answer is available to
    a retraining trigger, which needs the drift status for exactly the same
    reason a person looking at a dashboard does.
    """
    window_days = max(1, settings.retention_days) if settings.retention_days else 90
    since = now_utc() - timedelta(days=window_days)

    stats = db.prediction_stats(run_key, owner_id=owner_id, since=since)
    drift_status = drift_report(run_key, metadata, owner_id=owner_id)
    versions = db.list_model_versions(run_key, owner_id=owner_id)
    champion = next((version for version in versions if version["is_champion"]), None)

    return {
        "run_key": run_key,
        "logging_enabled": enabled(),
        "inputs_logged": bool(settings.prediction_log_inputs),
        "window_days": window_days,
        "predictions": stats,
        "drift": drift_status,
        "calibration": metadata.get("calibration") or {"verdict": "unmeasured"},
        "champion": champion,
        "versions": versions,
        "model_card": metadata.get("model_card"),
        "trained_at": metadata.get("timestamp"),
        "selection_metric": metadata.get("selection_metric"),
        "health": metadata.get("health"),
        "retraining": retraining_advice(drift_status, stats),
    }


def retraining_advice(drift_status: dict, stats: dict) -> dict:
    """Whether this model is asking to be retrained, and on what evidence.

    Advice, never an action. Retraining automatically on a drift signal sounds
    like the obvious next step and is a good way to promote a model fitted on
    whatever anomaly caused the signal. A person decides; this hands them the
    reason and the endpoint.
    """
    status = drift_status.get("status")
    moved = [
        entry["column"]
        for entry in drift_status.get("columns", [])
        if entry.get("verdict") in {drift.MODERATE, drift.MATERIAL}
    ]

    if status == drift.MATERIAL:
        return {
            "recommended": True,
            "trigger": "drift",
            "reason": (
                f"{len(moved)} feature(s) have moved materially since this model was trained "
                f"({', '.join(moved[:5])}). Retrain on data that includes the new distribution, then let the "
                "champion/challenger comparison decide whether the new model is actually better."
            ),
            "how": "POST /api/runs/{run_key}/retrain with a CSV of newer labelled data.",
        }
    if status == drift.MODERATE:
        return {
            "recommended": False,
            "trigger": "drift",
            "reason": (
                f"{len(moved)} feature(s) have moved moderately ({', '.join(moved[:5])}). Worth watching; not "
                "yet worth refitting."
            ),
            "how": None,
        }
    if status == drift.INSUFFICIENT:
        return {
            "recommended": False,
            "trigger": None,
            "reason": drift_status.get("reason", ""),
            "how": None,
        }
    return {
        "recommended": False,
        "trigger": None,
        "reason": (
            "The inputs this model is being sent still look like the ones it was trained on."
            if stats.get("count")
            else "Nothing has been predicted yet, so there is nothing to judge."
        ),
        "how": None,
    }
