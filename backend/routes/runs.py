"""Dataset/run registry — the workspace of past runs.

Backed by the SQLite table the pipeline writes on every completed run, plus the
stored metadata artifact for the full dashboard payload. This is what lets a
returning user find yesterday's dataset instead of losing it on refresh.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Query

import db
from utils.helpers import METADATA_DIR, MODEL_DIR
from utils.security import artifact_path

logger = logging.getLogger(__name__)

router = APIRouter()

# Four series is the ceiling the categorical palette validates for grouped bars.
MAX_COMPARE_RUNS = 4


def _load_metadata(run_key: str) -> dict:
    metadata_path = artifact_path(METADATA_DIR, run_key, ".json")
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="No stored artifacts for this run key.")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/runs")
def list_runs(limit: int = Query(default=50, ge=1, le=200)):
    runs = db.list_runs(limit=limit)
    return {"count": len(runs), "runs": runs}


# NOTE: declared before /runs/{run_key} on purpose. FastAPI matches routes in
# declaration order, so a dynamic path listed first would swallow "compare".
@router.get("/runs/compare")
def compare_runs(keys: str = Query(..., description="Comma-separated run keys")):
    """Model scores side by side across runs, for the comparison chart."""
    requested = [key.strip() for key in keys.split(",") if key.strip()]
    if not requested:
        raise HTTPException(status_code=400, detail="Provide at least one run key.")
    if len(requested) > MAX_COMPARE_RUNS:
        raise HTTPException(status_code=400, detail=f"Compare at most {MAX_COMPARE_RUNS} runs at once.")

    comparisons = []
    for run_key in requested:
        artifact_path(METADATA_DIR, run_key, ".json")  # validates the key format
        try:
            metadata = _load_metadata(run_key)
        except HTTPException:
            logger.info("Skipping missing run in comparison", extra={"run_key": run_key})
            continue
        comparisons.append(
            {
                "run_key": run_key,
                "mode": metadata.get("mode"),
                "target": metadata.get("target"),
                "problem_type": metadata.get("problem_type"),
                "selected_model": metadata.get("selected_model"),
                "health_verdict": (metadata.get("health") or {}).get("verdict"),
                "scores": metadata.get("all_model_scores", {}),
                "created_at": metadata.get("timestamp"),
            }
        )

    if not comparisons:
        raise HTTPException(status_code=404, detail="None of those runs have stored artifacts.")

    problem_types = {item["problem_type"] for item in comparisons}
    comparable = len(problem_types) == 1
    return {
        "runs": comparisons,
        # Plotting an accuracy against an RMSE on one axis would be meaningless;
        # the UI uses this to explain rather than draw a misleading chart.
        "comparable": comparable,
        "problem_type": comparisons[0]["problem_type"] if comparable else None,
        "metric": ("accuracy" if comparisons[0]["problem_type"] == "classification" else "rmse")
        if comparable
        else None,
    }


@router.get("/runs/{run_key}")
def get_run(run_key: str):
    artifact_path(METADATA_DIR, run_key, ".json")  # validates the key format
    run = db.get_run(run_key)
    if run is None:
        raise HTTPException(status_code=404, detail="No run found for this key.")
    return run


@router.get("/runs/{run_key}/result")
def get_run_result(run_key: str):
    """The full dashboard payload for a past run, so it can be reopened.

    Same shape the upload job returns, rebuilt from stored metadata — the
    dashboard does not need to know whether a result is fresh or reopened.
    """
    from routes.upload import _result_payload

    metadata = _load_metadata(run_key)
    return {**_result_payload(metadata, status="reopened"), "created_at": metadata.get("timestamp")}


@router.delete("/runs/{run_key}")
def delete_run(run_key: str):
    """Remove the registry row and every artifact belonging to the run."""
    paths = [
        artifact_path(METADATA_DIR, run_key, ".json"),
        artifact_path(METADATA_DIR, run_key, "_data.csv"),
        artifact_path(MODEL_DIR, run_key, "_model.pkl"),
    ]
    existed = db.delete_run(run_key)
    # Conversations are about this dataset and have no meaning without it —
    # deleting the dataset but keeping the transcript of questions asked about
    # it would leave the most sensitive part behind.
    conversations_removed = db.delete_conversations_for_run(run_key)
    removed = 0
    for path in paths:
        if path.exists():
            path.unlink()
            removed += 1
    if not existed and removed == 0:
        raise HTTPException(status_code=404, detail="No run found for this key.")
    logger.info(
        "Run deleted",
        extra={"run_key": run_key, "artifacts_removed": removed, "conversations_removed": conversations_removed},
    )
    return {
        "run_key": run_key,
        "deleted": True,
        "artifacts_removed": removed,
        "conversations_removed": conversations_removed,
    }
