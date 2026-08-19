"""The endpoints for a model after training day.

Everything a run's page could show while it was being trained already had a
route. None of it was ever revisited: there was no way to ask what this model
has been predicting, whether the questions it is being asked still look like the
ones it learned from, or which fitted artifact answered a given request.

Six endpoints, each owner-scoped exactly like the run they belong to:

    GET  /api/runs/{key}/monitoring    everything below, in one call
    GET  /api/runs/{key}/drift         the distribution comparison on its own
    GET  /api/runs/{key}/predictions   what has been served
    GET  /api/predictions/{id}         one served answer, in full
    GET  /api/runs/{key}/versions      the champion and everything it beat
    POST /api/runs/{key}/retrain       fit a challenger on newer labelled data

`require_owned_run` on every one of them, so a run belonging to somebody else
answers 404 for the same reason it does everywhere else: a run key is a content
hash, and confirming one exists would leak both the dataset and who holds it.
"""

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

import audit
import db
import limits
import monitoring
import registry
import retraining
import utils.helpers
from auth import Principal, current_principal, owner_scope, require_owned_run
from config import settings
from exceptions import PipelineError
from utils.helpers import read_csv_with_report
from utils.security import artifact_path, validate_dataset_hash
from utils.storage import ensure_local

logger = logging.getLogger(__name__)

router = APIRouter()


def _metadata(run_key: str) -> dict:
    """The run's metadata artifact, or a 404 that says which half is missing.

    `utils.helpers.METADATA_DIR` is read as a module attribute rather than
    imported by name — several modules imported the name directly and the test
    suite has to rebind it in each of them by hand (`_DIR_HOLDERS` in
    tests/conftest.py). Reading it here at call time means this module never
    needs adding to that list, and it is not a hypothetical: this route read
    from the wrong directory and 404'd for the run's own owner the first time
    it was tested, for exactly this reason.
    """
    validate_dataset_hash(run_key)
    ensure_local(run_key)
    path = artifact_path(utils.helpers.METADATA_DIR, run_key, ".json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No trained artifacts found for this run key.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@router.get("/runs/{run_key}/monitoring")
def monitoring_view(run_key: str, principal: Principal = Depends(current_principal)):
    """Volume, latency, drift, calibration, versions, and the model card.

    One call rather than five, because the question a person actually has is
    "is this model still all right?", and answering it from five endpoints means
    five chances to look at four of them.
    """
    require_owned_run(run_key, principal)
    return monitoring.overview(run_key, _metadata(run_key), owner_id=owner_scope(principal))


@router.get("/runs/{run_key}/drift")
def drift_view(run_key: str, principal: Principal = Depends(current_principal)):
    """Training-time distributions against what is actually being sent now."""
    require_owned_run(run_key, principal)
    return monitoring.drift_report(run_key, _metadata(run_key), owner_id=owner_scope(principal))


@router.get("/runs/{run_key}/predictions")
def prediction_log(
    run_key: str,
    limit: int = Query(default=100, ge=1, le=1000),
    principal: Principal = Depends(current_principal),
):
    """What this model has been asked, and what it answered.

    Most recent first. The inputs are here because reproducing an answer needs
    them; they are as sensitive as the training rows, which is why this is
    owner-scoped and why the retention policy covers it.
    """
    require_owned_run(run_key, principal)
    return {
        "run_key": run_key,
        "logging_enabled": monitoring.enabled(),
        "inputs_logged": bool(settings.prediction_log_inputs),
        "entries": db.list_predictions(run_key, limit=limit, owner_id=owner_scope(principal)),
        "retention_days": settings.retention_days,
        "note": (
            "Prediction inputs are covered by the retention policy and are deleted with the run."
            if settings.retention_days
            else "Retention is off, so these are kept until the run is deleted."
        ),
    }


@router.get("/predictions/{prediction_id}")
def one_prediction(prediction_id: str, principal: Principal = Depends(current_principal)):
    """One served answer, in enough detail to reproduce it.

    The exit criterion this exists for: inputs, model version, output, and when.
    Owner-scoped by the row's own `owner_id` rather than through the run,
    because a prediction outliving its run's registry row should still be
    unreadable to everyone else.
    """
    entry = db.get_prediction(prediction_id, owner_id=owner_scope(principal))
    if entry is None:
        raise HTTPException(status_code=404, detail="No such prediction.")

    version = next(
        (
            candidate
            for candidate in db.list_model_versions(entry["run_key"], owner_id=owner_scope(principal))
            if candidate["version_id"] == entry["model_version"]
        ),
        None,
    )
    return {
        **entry,
        "model": version,
        "how_to_reproduce": (
            f"POST the inputs below to /api/predict/{entry['run_key']}/row. That serves the current "
            "champion, which is the same model only if `model` below is still the champion — the version "
            "is recorded here precisely because it may not be."
        ),
    }


@router.get("/runs/{run_key}/versions")
def versions(run_key: str, principal: Principal = Depends(current_principal)):
    """Every model fitted for this run, and what each had to beat."""
    require_owned_run(run_key, principal)
    return registry.history(run_key, owner_id=owner_scope(principal))


@router.post("/runs/{run_key}/retrain")
async def retrain_run(
    run_key: str,
    file: UploadFile = File(...),
    trigger: str = Form("manual"),
    principal: Principal = Depends(current_principal),
):
    """Fit a challenger on newer labelled data; promote it only if it wins.

    Synchronous rather than queued, deliberately. The response *is* the
    comparison — both scores and the decision — and a job id that has to be
    polled to find out whether a model was replaced turns a decision into a
    notification.

    It does not create a row in the `jobs` table, because there is no polling
    for a synchronous call to service — but it is fitting a model on this
    process just as expensively as an upload's training does, so it is checked
    against the same Phase 8 concurrency ceiling before it starts. Skipping
    that check here would have been a hole: an account at its upload
    concurrency limit could still occupy every worker by retraining instead.
    """
    require_owned_run(run_key, principal)
    owner_id = owner_scope(principal) or db.LOCAL_OWNER_ID
    limits.check_job_concurrency(principal)
    metadata = _metadata(run_key)

    from utils.uploads import read_upload_to_temp

    stored = await read_upload_to_temp(file)
    try:
        frame, report = read_csv_with_report(stored.read_bytes())
    finally:
        stored.cleanup()

    if frame.empty:
        raise HTTPException(status_code=400, detail="The retraining file is empty after parsing malformed rows.")

    try:
        result = retraining.retrain(run_key, owner_id, frame, metadata, trigger=trigger)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Promotion changes what every caller of this run is served, which makes it
    # exactly the kind of act Phase 8's audit log exists for: attributable, to a
    # principal, at a time.
    audit.record(
        audit.MODEL_PROMOTED if result["promoted"] else audit.MODEL_CHALLENGED,
        principal,
        target_type="run",
        target_id=run_key,
        version=result["challenger"]["version"],
        metric=result["comparison"].get("metric"),
        challenger=result["comparison"].get("challenger"),
        champion=result["comparison"].get("champion"),
        trigger=trigger,
    )

    return {**result, "parse_warnings": report.warnings}


@router.get("/monitoring/due")
def due_for_retraining(
    min_age_days: int = Query(default=30, ge=1, le=3650),
    principal: Principal = Depends(current_principal),
):
    """Runs whose champion is old enough that a scheduled retrain is reasonable.

    The "on a schedule" trigger without a scheduler in the web process — an
    in-process timer is one more thing that dies with the process, which is the
    same reasoning that keeps the retention purge on cron. This reports what is
    due; cron reads it and calls the retrain endpoint. `docs/monitoring.md` has
    the line.
    """
    return {
        "min_age_days": min_age_days,
        "due": retraining.scheduled_candidates(owner_id=owner_scope(principal), min_age_days=min_age_days),
        "note": "Reported, not acted on. Retraining needs newer labelled data, which only you can supply.",
    }
