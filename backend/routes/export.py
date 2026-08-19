"""Exporting a trained pipeline — Phase 10. See ml/export.py for the formats."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from auth import Principal, current_principal, owner_scope, require_owned_run
from ml.export import export_bundle, export_onnx, onnx_availability
from registry import model_path_for

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/runs/{run_key}/export/availability")
def export_availability(run_key: str, principal: Principal = Depends(current_principal)):
    """Which export formats this run supports, before a caller downloads one."""
    from routes.predict import _load_bundle

    require_owned_run(run_key, principal)
    _, bundle, _ = _load_bundle(run_key, owner_id=owner_scope(principal))
    return {"onnx": onnx_availability(bundle), "bundle": {"available": True, "reason": None}}


@router.get("/runs/{run_key}/export")
def export_run(
    run_key: str,
    format: str = Query(default="bundle", pattern="^(onnx|bundle)$"),
    principal: Principal = Depends(current_principal),
):
    from routes.predict import _load_bundle

    require_owned_run(run_key, principal)
    metadata, bundle, version = _load_bundle(run_key, owner_id=owner_scope(principal))

    if format == "onnx":
        try:
            payload = export_onnx(bundle)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info("ONNX export", extra={"run_key": run_key})
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{run_key}_model.onnx"'},
        )

    model_path = model_path_for(run_key, version["artifact_suffix"])
    payload = export_bundle(run_key, bundle, metadata, model_path)
    logger.info("Bundle export", extra={"run_key": run_key})
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_key}_export.zip"'},
    )
