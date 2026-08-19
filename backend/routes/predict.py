import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import db
import monitoring
import registry
from auth import Principal, current_principal, owner_scope, require_owned_run
from exceptions import ArtifactError
from ml import serving_schema
from ml.ensemble import ensemble_predict
from observability import metrics
from utils.artifacts import load_bundle
from utils.helpers import METADATA_DIR, read_csv_with_report
from utils.security import artifact_path, validate_dataset_hash
from utils.storage import ensure_local
from utils.uploads import read_upload_to_temp

logger = logging.getLogger(__name__)

router = APIRouter()


class SingleRowRequest(BaseModel):
    """One row typed or pasted into the UI, instead of a whole CSV."""

    row: dict[str, Any] = Field(default_factory=dict)


def _decode(preds, label_encoder):
    """Turn label-encoded integers back into the original class names."""
    if label_encoder is None:
        return preds
    try:
        return label_encoder.inverse_transform(np.asarray(preds).astype(int))
    except Exception as exc:
        logger.warning("Could not inverse-transform predicted labels: %s", exc)
        return preds


def _load_bundle(run_key: str, owner_id: str | None = None) -> tuple[dict, dict, dict]:
    """Validate the key, then load metadata and the champion model bundle.

    Returns (metadata, bundle, version). The version is what says *which* model
    answered — a run can now hold several, and "the model said 0.83" stops
    being reproducible the moment a retrain lands if nothing recorded which one
    it was.
    """
    validate_dataset_hash(run_key)
    metadata_path = artifact_path(METADATA_DIR, run_key, ".json")

    version = registry.resolve(run_key, owner_id=owner_id)

    # With object storage the run may have been trained by a different web
    # process that this one shares no disk with, so "missing" is only true
    # after the bucket has been asked. A no-op on the local backend. Resolved
    # *before* this call and passed through: a promoted challenger's bundle
    # lives under `version["artifact_suffix"]`, a name the default suffix list
    # does not know, so asking before resolving the version would only ever
    # fetch v1 and leave a replica serving a champion v2 stuck on the 503
    # below forever, not just until it "syncs".
    ensure_local(run_key, [version["artifact_suffix"]])

    model_path = registry.model_path_for(run_key, version["artifact_suffix"])
    if version["versioned"] and not model_path.exists():
        # The registry names a bundle that is not on this disk. Falling back to
        # the original artifact would serve a model other than the champion and
        # log it under the champion's version, which is worse than refusing:
        # the record would say something untrue and nothing would look wrong.
        logger.error(
            "Champion artifact missing",
            extra={"run_key": run_key, "version": version["version"], "path": str(model_path)},
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"The model version currently promoted for this run (v{version['version']}) is not "
                "available on this server. It may still be publishing; try again shortly."
            ),
        )

    if not (metadata_path.exists() and model_path.exists()):
        raise HTTPException(status_code=404, detail="No trained artifacts found for this run key.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Signature first: joblib.load() executes pickle opcodes, so an artifact
    # this application did not write must never reach it.
    bundle = load_bundle(model_path)
    if not isinstance(bundle, dict) or "pipelines" not in bundle:
        raise ArtifactError("Stored artifact predates the current pipeline. Re-upload the dataset to retrain.")
    return metadata, bundle, version


def _predict_frame(bundle: dict, X: pd.DataFrame, run_key: str):
    """Run the stored pipeline(s). Returns (predictions, probabilities|None)."""
    pipelines = bundle["pipelines"]
    problem_type = bundle.get("problem_type")
    try:
        if bundle.get("kind") == "ensemble":
            # A majority vote has no calibrated probability to report.
            return ensemble_predict(pipelines, X, problem_type), None
        pipeline = next(iter(pipelines.values()))
        predictions = pipeline.predict(X)
        probabilities = None
        if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            try:
                probabilities = np.asarray(pipeline.predict_proba(X))
            except Exception as exc:
                logger.info("predict_proba unavailable for this model: %s", exc)
        return predictions, probabilities
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for run %s", run_key)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


def _build_response(
    run_key: str,
    metadata: dict,
    bundle: dict,
    X: pd.DataFrame,
    raw_preds,
    probabilities,
    parse_warnings: list[str],
    version: dict | None = None,
) -> dict:
    """One response shape for both the CSV and single-row entry points."""
    label_encoder = bundle.get("label_encoder")
    problem_type = bundle.get("problem_type", metadata.get("problem_type"))
    decoded = _decode(raw_preds, label_encoder)
    class_names = [str(c) for c in getattr(label_encoder, "classes_", [])]

    rows: list[dict[str, Any]] = []
    for position, prediction in enumerate(decoded):
        entry: dict[str, Any] = {
            "index": position,
            "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
        }
        if probabilities is not None and position < len(probabilities):
            row_probabilities = probabilities[position]
            entry["confidence"] = float(row_probabilities.max())
            # Per-class probabilities, so the UI can show what came second.
            entry["class_probabilities"] = {
                (class_names[i] if i < len(class_names) else str(i)): float(value)
                for i, value in enumerate(row_probabilities)
            }
        rows.append(entry)

    calibration = metadata.get("calibration") or {}
    return {
        "run_key": run_key,
        "dataset_hash": metadata.get("dataset_hash"),
        "selected_model": metadata.get("selected_model"),
        "problem_type": problem_type,
        "target": metadata.get("target"),
        "features": list(X.columns),
        "class_names": class_names,
        "parse_warnings": parse_warnings,
        # Which model answered. A run can hold several versions once it has
        # been retrained, and a confidence with no model behind it cannot be
        # checked later.
        "model_version": (version or {}).get("version_id"),
        "model_version_number": (version or {}).get("version"),
        # A displayed confidence is a claim about how often the prediction is
        # right. Saying whether that claim has been calibrated — and by how far
        # it was off when measured — is the difference between a number and an
        # honest number.
        "confidence_is_calibrated": bool(calibration.get("confidence_is_calibrated")),
        "calibration": {
            "verdict": calibration.get("verdict", "unmeasured"),
            "expected_calibration_error": ((calibration.get("after") or calibration.get("before")) or {}).get("ece"),
        }
        if probabilities is not None
        else None,
        "rows": rows,
        # Kept flat for callers that only want the labels.
        "predictions": [row["prediction"] for row in rows],
        "confidences": [row.get("confidence") for row in rows] if probabilities is not None else None,
        "input_rows": X.head(200).to_dict(orient="records"),
    }


def _align_features(df: pd.DataFrame, bundle: dict, metadata: dict) -> pd.DataFrame:
    """Check the frame against what the model expects, then narrow it.

    The old version of this said `Missing required features: [\'spend\']` and
    stopped, which is correct and nearly useless to someone looking at a file
    that plainly has a spend column in it, spelled differently. It also said
    nothing at all about a column that was numeric at training time and is
    arriving as text — that one went into the pipeline, got imputed to the
    training median, and produced a confident answer for a value nobody
    supplied. See ml/serving_schema.py.
    """
    expected = bundle.get("feature_columns") or metadata.get("features", [])
    result = serving_schema.check(df, expected, serving_schema.numeric_columns_from_metadata(metadata))
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.message())
    # The stored Pipelines carry their own preprocessing, so raw columns go
    # straight in — there is no separate transform step to keep in sync.
    return df[expected]


def predict_rows(
    run_key: str,
    df: pd.DataFrame,
    parse_warnings: list[str] | None = None,
    owner_id: str | None = None,
) -> dict:
    """Load the run's champion model and predict on `df`.

    The one path from a frame to a response, shared by the upload route and by
    the analyst's `run_model` tool — so the agent cannot end up with a second,
    subtly different notion of what this model predicts. It is also the one
    place predictions are logged, for the same reason.
    """
    started = time.perf_counter()
    metadata, bundle, version = _load_bundle(run_key, owner_id=owner_id)
    X = _align_features(df, bundle, metadata)
    raw_preds, probabilities = _predict_frame(bundle, X, run_key)
    response = _build_response(run_key, metadata, bundle, X, raw_preds, probabilities, parse_warnings or [], version)
    elapsed = time.perf_counter() - started

    # After the response is built, so a monitoring problem cannot cost the
    # caller their answer. `X` rather than `df`: what is recorded is what the
    # model was actually given.
    monitoring.log_prediction_batch(
        run_key=run_key,
        owner_id=owner_id or db.LOCAL_OWNER_ID,
        model_version=version.get("version_id"),
        frame=X,
        rows=response["rows"],
        latency_ms=int(elapsed * 1000),
    )
    metrics.observe_prediction("ok", elapsed)
    return response


@router.post("/predict/{run_key}")
async def predict(
    run_key: str,
    file: UploadFile = File(...),
    principal: Principal = Depends(current_principal),
):
    validate_dataset_hash(run_key)
    require_owned_run(run_key, principal)
    stored = await read_upload_to_temp(file)
    try:
        df, report = read_csv_with_report(stored.read_bytes())
    finally:
        stored.cleanup()
    if df.empty:
        raise HTTPException(status_code=400, detail="Prediction CSV is empty after parsing malformed rows.")

    return predict_rows(run_key, df, report.warnings, owner_id=owner_scope(principal) or db.LOCAL_OWNER_ID)


@router.post("/predict/{run_key}/row")
def predict_single_row(
    run_key: str,
    request: SingleRowRequest,
    principal: Principal = Depends(current_principal),
):
    """Predict one row supplied as JSON — no file upload required."""
    validate_dataset_hash(run_key)
    require_owned_run(run_key, principal)
    owner_id = owner_scope(principal) or db.LOCAL_OWNER_ID
    started = time.perf_counter()
    metadata, bundle, version = _load_bundle(run_key, owner_id=owner_id)

    if not request.row:
        raise HTTPException(status_code=400, detail="No values supplied.")

    expected = bundle.get("feature_columns") or metadata.get("features", [])
    supplied = pd.DataFrame([request.row])
    check = serving_schema.check(supplied, expected, serving_schema.numeric_columns_from_metadata(metadata))
    if check.missing:
        # Type breakage is not checked here: a single row typed into a form
        # arrives as strings by definition, and the conversion below is what
        # turns it back into numbers. The CSV path, where a whole column can be
        # the wrong kind, is where that check belongs.
        raise HTTPException(status_code=400, detail=check.message())

    # Values arrive as strings from a form; let pandas infer real dtypes so the
    # numeric columns are not handed to the scaler as text.
    frame = pd.DataFrame([{column: request.row[column] for column in expected}])
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted

    raw_preds, probabilities = _predict_frame(bundle, frame, run_key)
    response = _build_response(run_key, metadata, bundle, frame, raw_preds, probabilities, [], version)
    elapsed = time.perf_counter() - started
    monitoring.log_prediction_batch(
        run_key=run_key,
        owner_id=owner_id,
        model_version=version.get("version_id"),
        frame=frame,
        rows=response["rows"],
        latency_ms=int(elapsed * 1000),
    )
    metrics.observe_prediction("ok", elapsed)
    return response
