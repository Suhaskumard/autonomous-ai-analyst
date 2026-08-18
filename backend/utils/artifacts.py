"""Signed model artifacts.

`joblib.load` runs pickle opcodes, and pickle opcodes can call anything. Every
phase so far has kept *paths* safe — the run key is validated as a hex digest
and resolved under a known root — but that only proves the file is in the right
directory, not that this application is what wrote it. Anyone who can drop a
file into `models/saved_models/` gets code execution the next time a prediction
is served.

Replacing pickle outright is not on the table: scikit-learn Pipelines have no
portable non-pickle format, and ONNX conversion loses the custom preprocessing.
So the plan's fallback applies — sign the artifacts. Each bundle is written with
an HMAC-SHA256 sidecar over its bytes, and `load_bundle` refuses to hand
anything to joblib until that signature verifies.

The key lives in settings, or is generated once and persisted alongside the
artifacts. It is not a defence against someone who can read the key file; it is
a defence against a file arriving from anywhere other than this application.
"""

import hashlib
import hmac
import logging
import secrets
from pathlib import Path

import joblib

from config import settings
from exceptions import ArtifactError

logger = logging.getLogger(__name__)

SIGNATURE_SUFFIX = ".sig"
_CHUNK = 1024 * 1024


def _key_file() -> Path:
    from utils.helpers import MODEL_DIR

    return Path(MODEL_DIR).parent / ".artifact_signing_key"


def signing_key() -> bytes:
    """The HMAC key, from settings or from a generated-once local file."""
    configured = (settings.artifact_signing_key or "").strip()
    if configured:
        return configured.encode("utf-8")

    path = _key_file()
    if path.exists():
        return path.read_bytes().strip()

    key = secrets.token_hex(32).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:  # best effort: owner-only where the platform supports it
        path.chmod(0o600)
    except OSError:  # pragma: no cover - Windows and odd filesystems
        pass
    logger.info("Generated a new artifact signing key at %s", path)
    return key


def digest_file(path: Path) -> str:
    """HMAC-SHA256 of a file's bytes, streamed rather than read whole."""
    mac = hmac.new(signing_key(), digestmod=hashlib.sha256)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            mac.update(chunk)
    return mac.hexdigest()


def signature_path(artifact_path: Path) -> Path:
    return Path(str(artifact_path) + SIGNATURE_SUFFIX)


def sign_artifact(artifact_path: Path) -> str:
    """Write the sidecar signature for an artifact already on disk."""
    signature = digest_file(artifact_path)
    signature_path(artifact_path).write_text(signature, encoding="ascii")
    return signature


def verify_artifact(artifact_path: Path) -> None:
    """Raise ArtifactError unless this application wrote these exact bytes."""
    artifact_path = Path(artifact_path)
    sidecar = signature_path(artifact_path)

    if not sidecar.exists():
        raise ArtifactError(
            "This model artifact is unsigned, so it cannot be verified as one this application "
            "produced. Re-upload the dataset to retrain it. (Artifacts written before signing was "
            "introduced are unsigned and must be regenerated.)"
        )

    expected = sidecar.read_text(encoding="ascii").strip()
    actual = digest_file(artifact_path)
    if not hmac.compare_digest(expected, actual):
        logger.error("Artifact signature mismatch", extra={"artifact": artifact_path.name})
        raise ArtifactError(
            "This model artifact failed its integrity check — its contents do not match the "
            "signature written when it was trained. It will not be loaded. Re-upload the dataset "
            "to retrain."
        )


def dump_bundle(bundle: dict, artifact_path: Path) -> None:
    """Serialise and sign, in that order."""
    artifact_path = Path(artifact_path)
    joblib.dump(bundle, artifact_path)
    sign_artifact(artifact_path)


def load_bundle(artifact_path: Path):
    """Verify, then deserialise. Never the other way round."""
    verify_artifact(artifact_path)
    return joblib.load(artifact_path)


def remove_artifact(artifact_path: Path) -> int:
    """Delete an artifact and its signature. Returns how many files went."""
    removed = 0
    for path in (Path(artifact_path), signature_path(artifact_path)):
        if path.exists():
            path.unlink()
            removed += 1
    return removed
