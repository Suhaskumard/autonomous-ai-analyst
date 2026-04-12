import json

from fastapi import APIRouter, HTTPException

from utils.helpers import METADATA_DIR

router = APIRouter()


@router.get("/insights/{dataset_hash}")
def get_insights(dataset_hash: str):
    metadata_path = METADATA_DIR / f"{dataset_hash}.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="No metadata found for this dataset hash.")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "dataset_hash": dataset_hash,
        "problem_type": metadata.get("problem_type"),
        "model": metadata.get("selected_model"),
        "scores": metadata.get("all_model_scores", {}),
        "feature_importance": metadata.get("feature_importance", []),
        "insights": metadata.get("insights", []),
        "quality_report": metadata.get("quality_report", {}),
        "charts": metadata.get("charts", {}),
    }
