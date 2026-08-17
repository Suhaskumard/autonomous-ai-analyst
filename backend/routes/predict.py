import json
import logging

import joblib
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from exceptions import ArtifactError
from ml.ensemble import ensemble_predict
from utils.helpers import METADATA_DIR, MODEL_DIR, read_csv_flexible
from utils.security import artifact_path, validate_dataset_hash
from utils.uploads import read_upload_to_temp

logger = logging.getLogger(__name__)

router = APIRouter()


def _decode(preds, label_encoder):
    """Turn label-encoded integers back into the original class names."""
    if label_encoder is None:
        return preds
    try:
        return label_encoder.inverse_transform(np.asarray(preds).astype(int))
    except Exception as exc:
        logger.warning("Could not inverse-transform predicted labels: %s", exc)
        return preds


@router.post("/predict/{run_key}")
async def predict(run_key: str, file: UploadFile = File(...)):
    # Validate before the key is ever interpolated into a path that joblib
    # will deserialize.
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

    stored = await read_upload_to_temp(file)
    try:
        df, parse_warnings = read_csv_flexible(stored.read_bytes())
    finally:
        stored.cleanup()
    if df.empty:
        raise HTTPException(status_code=400, detail="Prediction CSV is empty after parsing malformed rows.")

    expected_features = bundle.get("feature_columns") or metadata.get("features", [])
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    # The stored Pipelines carry their own preprocessing, so raw columns go
    # straight in — there is no separate transform step to keep in sync.
    X = df[expected_features]
    pipelines = bundle["pipelines"]
    problem_type = bundle.get("problem_type", metadata.get("problem_type"))
    label_encoder = bundle.get("label_encoder")

    try:
        if bundle.get("kind") == "ensemble":
            raw_preds = ensemble_predict(pipelines, X, problem_type)
            probabilities = None
        else:
            pipeline = next(iter(pipelines.values()))
            raw_preds = pipeline.predict(X)
            probabilities = None
            if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
                try:
                    probabilities = pipeline.predict_proba(X)
                except Exception as exc:
                    logger.info("predict_proba unavailable for this model: %s", exc)
    except Exception as exc:
        logger.exception("Prediction failed for run %s", run_key)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    decoded = _decode(raw_preds, label_encoder)
    class_names = [str(c) for c in getattr(label_encoder, "classes_", [])]

    confidences = None
    if probabilities is not None:
        confidences = [float(row.max()) for row in np.asarray(probabilities)]

    return {
        "run_key": run_key,
        "dataset_hash": metadata.get("dataset_hash"),
        "selected_model": metadata.get("selected_model"),
        "problem_type": problem_type,
        "target": metadata.get("target"),
        "class_names": class_names,
        "parse_warnings": parse_warnings,
        "predictions": [p.item() if hasattr(p, "item") else p for p in decoded],
        "confidences": confidences,
    }
