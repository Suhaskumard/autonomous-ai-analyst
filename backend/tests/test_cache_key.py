"""The cache key must cover the whole run configuration.

Regression guard: the cache was keyed on the dataset hash alone, so
re-uploading the same CSV in Ensemble mode after Auto silently returned the
Auto result and the mode selector appeared to do nothing.
"""

import pytest

from ml.train import PIPELINE_VERSION
from utils.hashing import dataset_sha256, run_cache_key

HASH = "0123456789abcdef" * 4


def key(**overrides):
    args = {
        "dataset_hash": HASH,
        "mode": "auto",
        "manual_model": None,
        "target_column": None,
        "pipeline_version": PIPELINE_VERSION,
    }
    args.update(overrides)
    return run_cache_key(**args)


def test_identical_configuration_is_stable():
    assert key() == key()


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "ensemble"},
        {"manual_model": "RandomForest"},
        {"target_column": "price"},
        {"pipeline_version": "999"},
        {"dataset_hash": "f" * 64},
    ],
)
def test_every_configuration_field_changes_the_key(override):
    assert key(**override) != key()


def test_mode_is_case_insensitive():
    assert key(mode="AUTO") == key(mode="auto")


def test_key_is_path_safe():
    """The key becomes a filename, so it must survive validate_dataset_hash."""
    from utils.security import validate_dataset_hash

    assert validate_dataset_hash(key()) == key()


def test_dataset_hash_is_content_addressed():
    assert dataset_sha256(b"a,b\n1,2\n") == dataset_sha256(b"a,b\n1,2\n")
    assert dataset_sha256(b"a,b\n1,2\n") != dataset_sha256(b"a,b\n1,3\n")
