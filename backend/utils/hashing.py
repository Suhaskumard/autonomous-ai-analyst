import hashlib


def dataset_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def run_cache_key(
    dataset_hash: str,
    mode: str,
    manual_model: str | None,
    target_column: str | None,
    pipeline_version: str,
) -> str:
    """Content-address the whole run configuration, not just the file.

    Keying on the dataset hash alone meant re-uploading the same CSV in
    Ensemble mode after Auto silently returned the Auto result, so the mode
    selector appeared to do nothing. Including the pipeline version also
    invalidates artifacts automatically when preprocessing changes.
    """
    parts = [
        dataset_hash,
        (mode or "auto").lower(),
        (manual_model or "").strip(),
        (target_column or "").strip(),
        pipeline_version,
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
