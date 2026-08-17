"""Input validation for anything that becomes a filesystem path.

A dataset hash arrives from the client and is interpolated straight into a
model path that joblib.load() then deserializes. Unvalidated, that is path
traversal into arbitrary pickle loading, i.e. remote code execution. Every
route that accepts a dataset_hash must run it through validate_dataset_hash
before it touches a path, and build the path with artifact_path so the result
is provably under a known root.
"""

from pathlib import Path
import re

from fastapi import HTTPException

DATASET_HASH_RE = re.compile(r"^[a-f0-9]{64}$")


def validate_dataset_hash(dataset_hash: str) -> str:
    """Return the hash if it is a lowercase hex sha256, else raise HTTP 400."""
    if not isinstance(dataset_hash, str) or not DATASET_HASH_RE.match(dataset_hash):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset_hash. Expected a 64-character lowercase hex SHA-256 digest.",
        )
    return dataset_hash


def artifact_path(root: Path, dataset_hash: str, suffix: str) -> Path:
    """Build root/<hash><suffix> and refuse anything that escapes root.

    The regex already makes traversal impossible; the resolve check is the
    belt-and-braces half, so a future change to the regex cannot silently
    reopen the hole.
    """
    validate_dataset_hash(dataset_hash)
    root_resolved = root.resolve()
    candidate = (root_resolved / f"{dataset_hash}{suffix}").resolve()
    if root_resolved != candidate.parent:
        raise HTTPException(status_code=400, detail="Resolved artifact path escapes its storage root.")
    return candidate
