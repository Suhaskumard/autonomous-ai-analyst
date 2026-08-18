import json

from fastapi import APIRouter, Depends, HTTPException

from auth import Principal, current_principal, require_owned_run
from utils.helpers import METADATA_DIR
from utils.security import artifact_path
from utils.storage import ensure_local

router = APIRouter()


@router.get("/insights/{run_key}")
def get_insights(run_key: str, principal: Principal = Depends(current_principal)):
    metadata_path = artifact_path(METADATA_DIR, run_key, ".json")
    require_owned_run(run_key, principal)
    ensure_local(run_key)
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="No metadata found for this run key.")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "run_key": run_key,
        "dataset_hash": metadata.get("dataset_hash"),
        "problem_type": metadata.get("problem_type"),
        "model": metadata.get("selected_model"),
        "target": metadata.get("target"),
        "scores": metadata.get("all_model_scores", {}),
        "health": metadata.get("health", {}),
        "failed_models": metadata.get("failed_models", {}),
        "feature_importance": metadata.get("feature_importance", []),
        "explanation_method": metadata.get("explanation_method"),
        "insights": metadata.get("insights", []),
        "quality_report": metadata.get("quality_report", {}),
        "charts": metadata.get("charts", {}),
        "summary_stats": metadata.get("summary_stats", {}),
    }
