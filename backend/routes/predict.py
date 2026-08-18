import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from exceptions import ArtifactError
from ml.ensemble import ensemble_predict
from utils.helpers import METADATA_DIR, MODEL_DIR, read_csv_with_report
from utils.security import artifact_path, validate_dataset_hash
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


def _load_bundle(run_key: str) -> tuple[dict, dict]:
    """Validate the key, then load metadata and the fitted model bundle."""
    validate_dataset_hash(run_key)
    metadata_path = artifact_path(METADATA_DIR, run_key, ".json")
    model_path = artifact_path(MODEL_DIR, run_key, "_model.pkl")

    if not (metadata_path.exists() and model_path.exists()):
        raise HTTPException(status_code=404, detail="No trained artifacts found for this run key.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipelines" not in bundle:
        raise ArtifactError("Stored artifact predates the current pipeline. Re-upload the dataset to retrain.")
    return metadata, bundle


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

    return {
        "run_key": run_key,
        "dataset_hash": metadata.get("dataset_hash"),
        "selected_model": metadata.get("selected_model"),
        "problem_type": problem_type,
        "target": metadata.get("target"),
        "features": list(X.columns),
        "class_names": class_names,
        "parse_warnings": parse_warnings,
        "rows": rows,
        # Kept flat for callers that only want the labels.
        "predictions": [row["prediction"] for row in rows],
        "confidences": [row.get("confidence") for row in rows] if probabilities is not None else None,
        "input_rows": X.head(200).to_dict(orient="records"),
    }


def _align_features(df: pd.DataFrame, bundle: dict, metadata: dict) -> pd.DataFrame:
    expected = bundle.get("feature_columns") or metadata.get("features", [])
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")
    # The stored Pipelines carry their own preprocessing, so raw columns go
    # straight in — there is no separate transform step to keep in sync.
    return df[expected]


@router.post("/predict/{run_key}")
async def predict(run_key: str, file: UploadFile = File(...)):
    metadata, bundle = _load_bundle(run_key)

    stored = await read_upload_to_temp(file)
    try:
        df, report = read_csv_with_report(stored.read_bytes())
    finally:
        stored.cleanup()
    if df.empty:
        raise HTTPException(status_code=400, detail="Prediction CSV is empty after parsing malformed rows.")

    X = _align_features(df, bundle, metadata)
    raw_preds, probabilities = _predict_frame(bundle, X, run_key)
    return _build_response(run_key, metadata, bundle, X, raw_preds, probabilities, report.warnings)


@router.post("/predict/{run_key}/row")
def predict_single_row(run_key: str, request: SingleRowRequest):
    """Predict one row supplied as JSON — no file upload required."""
    metadata, bundle = _load_bundle(run_key)

    if not request.row:
        raise HTTPException(status_code=400, detail="No values supplied.")

    expected = bundle.get("feature_columns") or metadata.get("features", [])
    missing = [column for column in expected if column not in request.row]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    # Values arrive as strings from a form; let pandas infer real dtypes so the
    # numeric columns are not handed to the scaler as text.
    frame = pd.DataFrame([{column: request.row[column] for column in expected}])
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted

    raw_preds, probabilities = _predict_frame(bundle, frame, run_key)
    return _build_response(run_key, metadata, bundle, frame, raw_preds, probabilities, [])
