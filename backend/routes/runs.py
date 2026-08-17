"""Dataset/run registry.

Backed by the SQLite table the pipeline writes on every completed run. The
Phase 3 history UI is built on these endpoints; they exist now because the
registry is what makes job state durable in the first place.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

import db
from utils.helpers import METADATA_DIR, MODEL_DIR
from utils.security import artifact_path

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/runs")
def list_runs(limit: int = Query(default=50, ge=1, le=200)):
    runs = db.list_runs(limit=limit)
    return {"count": len(runs), "runs": runs}


@router.get("/runs/{run_key}")
def get_run(run_key: str):
    artifact_path(METADATA_DIR, run_key, ".json")  # validates the key format
    run = db.get_run(run_key)
    if run is None:
        raise HTTPException(status_code=404, detail="No run found for this key.")
    return run


@router.delete("/runs/{run_key}")
def delete_run(run_key: str):
    """Remove the registry row and every artifact belonging to the run."""
    paths = [
        artifact_path(METADATA_DIR, run_key, ".json"),
        artifact_path(METADATA_DIR, run_key, "_data.csv"),
        artifact_path(MODEL_DIR, run_key, "_model.pkl"),
    ]
    existed = db.delete_run(run_key)
    removed = 0
    for path in paths:
        if path.exists():
            path.unlink()
            removed += 1
    if not existed and removed == 0:
        raise HTTPException(status_code=404, detail="No run found for this key.")
    logger.info("Run deleted", extra={"run_key": run_key, "artifacts_removed": removed})
    return {"run_key": run_key, "deleted": True, "artifacts_removed": removed}
