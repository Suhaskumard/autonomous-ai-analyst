"""Bounded upload handling.

`await file.read()` pulls an entire upload into RAM with no size, type, or row
ceiling, so a single large POST takes the process down. Everything here streams
the upload to a temp file in fixed-size chunks and refuses the request the
moment a limit is crossed, so nothing beyond one chunk is ever held in memory.
"""

import os
import tempfile
from hashlib import sha256
from pathlib import Path

from fastapi import HTTPException, UploadFile

from config import ALLOWED_UPLOAD_CONTENT_TYPES, ALLOWED_UPLOAD_EXTENSIONS, settings

CHUNK_SIZE = 1024 * 1024


class StoredUpload:
    """A capped upload that has landed on disk.

    `path` is a temp file the caller owns and must `cleanup()` when finished.
    `sha256` is computed during streaming, so the bytes are never re-read just
    to hash them.
    """

    def __init__(self, path: Path, digest: str, size: int, line_count: int) -> None:
        self.path = path
        self.sha256 = digest
        self.size = size
        self.line_count = line_count

    def read_bytes(self) -> bytes:
        # Bounded by MAX_UPLOAD_MB, which is what makes this safe to do at all.
        return self.path.read_bytes()

    def cleanup(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _check_name_and_type(file: UploadFile) -> None:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or filename}'. Allowed extensions: {allowed}.",
        )
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{content_type}'. Upload a plain-text CSV.",
        )


async def read_upload_to_temp(file: UploadFile) -> StoredUpload:
    """Stream `file` to a temp file, enforcing type, byte, and row ceilings."""
    _check_name_and_type(file)

    byte_cap = settings.max_upload_bytes
    row_cap = settings.max_upload_rows

    digest = sha256()
    total = 0
    newlines = 0

    # Deliberately not a context manager: the file outlives this function and
    # is owned by the caller, which closes it via StoredUpload.cleanup().
    handle = tempfile.NamedTemporaryFile(prefix="aia_upload_", suffix=".csv", delete=False)  # noqa: SIM115
    temp_path = Path(handle.name)

    def _abort(status_code: int, detail: str) -> HTTPException:
        handle.close()
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        return HTTPException(status_code=status_code, detail=detail)

    try:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break

            total += len(chunk)
            if total > byte_cap:
                raise _abort(
                    413,
                    f"Upload exceeds the {byte_cap // (1024 * 1024)} MB limit. "
                    "Reduce the file size or raise MAX_UPLOAD_MB.",
                )

            newlines += chunk.count(b"\n")
            # +1 for the header row; a file without a trailing newline is one
            # row longer than its newline count, hence the generous +2 slack.
            if newlines > row_cap + 2:
                raise _abort(
                    413,
                    f"Upload exceeds the {row_cap:,}-row limit. Sample the dataset or raise MAX_UPLOAD_ROWS.",
                )

            digest.update(chunk)
            handle.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise _abort(400, f"Could not read the uploaded file: {exc}") from exc

    handle.close()

    if total == 0:
        stored = StoredUpload(temp_path, digest.hexdigest(), 0, 0)
        stored.cleanup()
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return StoredUpload(temp_path, digest.hexdigest(), total, newlines)
