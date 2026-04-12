import json

import joblib
from fastapi import APIRouter, File, HTTPException, UploadFile

from ml.ensemble import ensemble_predict
from utils.helpers import METADATA_DIR, MODEL_DIR, read_csv_flexible

router = APIRouter()


@router.post("/predict/{dataset_hash}")
async def predict(dataset_hash: str, file: UploadFile = File(...)):
    metadata_path = METADATA_DIR / f"{dataset_hash}.json"
    model_path = MODEL_DIR / f"{dataset_hash}_model.pkl"
    pipeline_path = MODEL_DIR / f"{dataset_hash}_pipeline.pkl"

    if not (metadata_path.exists() and model_path.exists() and pipeline_path.exists()):
        raise HTTPException(status_code=404, detail="No trained artifacts found for this dataset hash.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_obj = joblib.load(model_path)
    preprocessor = joblib.load(pipeline_path)

    payload = await file.read()
    df, parse_warnings = read_csv_flexible(payload)
    if df.empty:
        raise HTTPException(status_code=400, detail="Prediction CSV is empty after parsing malformed rows.")
    expected_features = metadata.get("features", [])
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    X_processed = preprocessor.transform(df[expected_features])
    if isinstance(model_obj, dict) and model_obj.get("mode") == "ensemble":
        preds = ensemble_predict(model_obj["models"], X_processed, model_obj["problem_type"])
    else:
        preds = model_obj.predict(X_processed)

    return {
        "dataset_hash": dataset_hash,
        "selected_model": metadata.get("selected_model"),
        "parse_warnings": parse_warnings,
        "predictions": [p.item() if hasattr(p, "item") else p for p in preds],
    }
