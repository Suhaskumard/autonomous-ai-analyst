"""Pre-training data profile.

Answers "what is in this file, and which column can I predict?" before any
training starts, so the target picker is an informed choice rather than a
guess. Cheap: parse, profile, discard — nothing is fitted or stored.
"""

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from auth import Principal, current_principal
from exceptions import DatasetTooSmallError
from ml.profile import profile_dataframe
from utils.helpers import read_csv_with_report
from utils.uploads import read_upload_to_temp

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/profile")
async def profile_dataset(
    file: UploadFile = File(...),
    principal: Principal = Depends(current_principal),
):
    # Same streamed ceilings as /upload — this endpoint is just as exposed.
    stored = await read_upload_to_temp(file)
    try:
        df, report = read_csv_with_report(stored.read_bytes())
    finally:
        stored.cleanup()

    if df.shape[0] < 1 or df.shape[1] < 2:
        raise DatasetTooSmallError("Need at least 1 row and 2 columns to profile this file.")

    payload = profile_dataframe(df)
    logger.info(
        "Profiled dataset",
        extra={"rows": payload["rows"], "columns": payload["columns"], "skipped": report.rows_skipped},
    )
    return {
        "filename": file.filename,
        "dataset_hash": stored.sha256,
        **payload,
        "parse_report": report.as_dict(),
    }
