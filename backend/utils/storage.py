"""Where artifacts live.

The local filesystem is fine for one machine and wrong for more than one: two
web processes behind a load balancer do not share `models/`, so a run trained
by one is invisible to the other. This puts a small interface in front of it —
`LocalStorage` keeps today's behaviour exactly, `S3Storage` puts the same bytes
in a bucket.

The interface is deliberately narrow. Training and prediction still work
against real paths on a local disk, because scikit-learn and pandas want files;
remote storage mirrors those files up after a run and pulls them down on a miss.
That is a cache, not a filesystem abstraction, and it is honest about being one.
"""

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class ArtifactStorage(ABC):
    """Push local artifacts somewhere durable; pull them back on demand."""

    name: str

    @abstractmethod
    def put(self, local_path: Path, key: str) -> None: ...

    @abstractmethod
    def get(self, key: str, local_path: Path) -> bool:
        """Fetch into `local_path`. False when the key is not there."""

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalStorage(ArtifactStorage):
    """The single-machine default: artifacts already are the storage.

    put/get are no-ops rather than copies — the file the caller wrote is the
    file the next request reads. Keeping the same interface means the routes
    do not branch on which backend is configured.
    """

    name = "local"

    def put(self, local_path: Path, key: str) -> None:
        return None

    def get(self, key: str, local_path: Path) -> bool:
        return Path(local_path).exists()

    def delete(self, key: str) -> bool:
        return False

    def exists(self, key: str) -> bool:
        return False


class S3Storage(ArtifactStorage):
    """Artifacts in a bucket, so the app can run on more than one machine.

    Credentials come from the usual boto3 chain (environment, profile, instance
    role) — never from settings, so they are not one `/readyz` bug away from
    being logged.
    """

    name = "s3"

    def __init__(
        self, bucket: str, prefix: str = "analyst", endpoint_url: str | None = None, region: str | None = None
    ):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, local_path: Path, key: str) -> None:
        self._client.upload_file(str(local_path), self.bucket, self._key(key))
        logger.info("Artifact uploaded", extra={"key": key, "bucket": self.bucket})

    def get(self, key: str, local_path: Path) -> bool:
        local_path = Path(local_path)
        if local_path.exists():
            return True
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, self._key(key), str(local_path))
            return True
        except Exception as exc:
            logger.info("Artifact not in object storage: %s (%s)", key, exc)
            return False

    def delete(self, key: str) -> bool:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as exc:
            logger.warning("Could not delete %s from object storage: %s", key, exc)
            return False

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False


_storage: ArtifactStorage | None = None


def get_storage() -> ArtifactStorage:
    global _storage
    if _storage is not None:
        return _storage

    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            logger.error("storage_backend is 's3' but S3_BUCKET is unset; falling back to local storage.")
            _storage = LocalStorage()
        else:
            try:
                _storage = S3Storage(
                    bucket=settings.s3_bucket,
                    prefix=settings.s3_prefix,
                    endpoint_url=settings.s3_endpoint_url,
                    region=settings.s3_region,
                )
            except Exception as exc:
                # A misconfigured bucket must not take training down; the local
                # disk still works, it just does not scale past this machine.
                logger.error("Could not initialise S3 storage (%s); falling back to local.", exc)
                _storage = LocalStorage()
    else:
        _storage = LocalStorage()

    logger.info("Artifact storage backend: %s", _storage.name)
    return _storage


def reset_storage() -> None:
    """Drop the cached backend so a new configuration takes effect (tests)."""
    global _storage
    _storage = None


def set_storage(storage: ArtifactStorage | None) -> None:
    """Install a backend directly (tests)."""
    global _storage
    _storage = storage


# --- the artifact set for one run -------------------------------------------

#: (suffix, directory-selector) for every file a run produces.
ARTIFACT_SUFFIXES = (".json", "_data.csv", "_model.pkl", "_model.pkl.sig")


def run_artifact_paths(run_key: str) -> list[tuple[str, Path]]:
    """Every (storage key, local path) belonging to a run."""
    from utils.helpers import METADATA_DIR, MODEL_DIR

    paths = []
    for suffix in ARTIFACT_SUFFIXES:
        root = Path(MODEL_DIR) if "_model.pkl" in suffix else Path(METADATA_DIR)
        paths.append((f"{run_key}{suffix}", root / f"{run_key}{suffix}"))
    return paths


def publish_run(run_key: str) -> int:
    """Mirror a finished run's artifacts to durable storage."""
    storage = get_storage()
    if isinstance(storage, LocalStorage):
        return 0
    published = 0
    for key, path in run_artifact_paths(run_key):
        if path.exists():
            try:
                storage.put(path, key)
                published += 1
            except Exception as exc:
                logger.warning("Could not publish %s: %s", key, exc)
    return published


def ensure_local(run_key: str) -> None:
    """Pull a run's artifacts down if this machine does not have them."""
    storage = get_storage()
    if isinstance(storage, LocalStorage):
        return
    for key, path in run_artifact_paths(run_key):
        if not path.exists():
            storage.get(key, path)


def purge_run(run_key: str) -> int:
    """Delete a run's artifacts locally and remotely. Returns files removed."""
    storage = get_storage()
    removed = 0
    for key, path in run_artifact_paths(run_key):
        if path.exists():
            try:
                path.unlink()
                removed += 1
            except OSError as exc:  # pragma: no cover - permissions, open handles
                logger.warning("Could not delete %s: %s", path, exc)
        if not isinstance(storage, LocalStorage):
            storage.delete(key)
    return removed


def disk_usage_bytes() -> int:
    """How much local disk the artifacts occupy, for the metrics endpoint."""
    from utils.helpers import METADATA_DIR, MODEL_DIR

    total = 0
    for root in (Path(METADATA_DIR), Path(MODEL_DIR)):
        if root.exists():
            total += sum(f.stat().st_size for f in root.glob("*") if f.is_file())
    return total


def free_disk_bytes() -> int:
    from utils.helpers import MODEL_DIR

    try:
        return shutil.disk_usage(Path(MODEL_DIR)).free
    except OSError:  # pragma: no cover
        return 0
