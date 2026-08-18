import hashlib


def dataset_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def run_cache_key(
    dataset_hash: str,
    mode: str,
    manual_model: str | None,
    target_column: str | None,
    pipeline_version: str,
    tuning_budget_seconds: float = 0.0,
    use_smote: bool = False,
    owner_id: str = "local",
) -> str:
    """Content-address the whole run configuration, not just the file.

    Keying on the dataset hash alone meant re-uploading the same CSV in
    Ensemble mode after Auto silently returned the Auto result, so the mode
    selector appeared to do nothing. Every knob that changes the result belongs
    here for the same reason — asking for a tuning budget or for SMOTE must
    retrain rather than replay the untuned, unbalanced artifact. Including the
    pipeline version also invalidates artifacts automatically when
    preprocessing changes.
    """
    parts = [
        dataset_hash,
        (mode or "auto").lower(),
        (manual_model or "").strip(),
        (target_column or "").strip(),
        pipeline_version,
        f"tune={float(tuning_budget_seconds or 0):g}",
        f"smote={bool(use_smote)}",
        # Two accounts uploading the same file must not collide onto one run
        # key: a cache hit would serve one user the other's stored snapshot.
        f"owner={owner_id or 'local'}",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
