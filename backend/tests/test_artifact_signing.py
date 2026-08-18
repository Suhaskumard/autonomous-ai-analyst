"""Phase 6: signed model artifacts.

`joblib.load` runs pickle opcodes, so the question "is this file one we wrote?"
has to be answered before deserialisation, not after. These tests are about the
order of those two operations as much as about the signature itself.
"""

import joblib
import pytest

from exceptions import ArtifactError
from tests.conftest import upload_and_wait
from utils.artifacts import dump_bundle, load_bundle, sign_artifact, signature_path, verify_artifact


@pytest.fixture
def bundle_path(tmp_path):
    path = tmp_path / "bundle.pkl"
    dump_bundle({"kind": "single", "pipelines": {"fake": [1, 2, 3]}}, path)
    return path


def test_writing_a_bundle_also_writes_its_signature(bundle_path):
    sidecar = signature_path(bundle_path)

    assert sidecar.exists()
    assert len(sidecar.read_text(encoding="ascii").strip()) == 64  # hmac-sha256 hex


def test_a_signed_bundle_round_trips(bundle_path):
    assert load_bundle(bundle_path)["pipelines"]["fake"] == [1, 2, 3]


def test_a_tampered_bundle_is_refused(bundle_path):
    # The realistic attack: replace the artifact, keep the signature.
    joblib.dump({"kind": "single", "pipelines": {"evil": "payload"}}, bundle_path)

    with pytest.raises(ArtifactError, match="integrity check"):
        load_bundle(bundle_path)


def test_an_unsigned_bundle_is_refused(tmp_path):
    """The pre-Phase-6 case: an artifact on disk with no sidecar at all."""
    path = tmp_path / "legacy.pkl"
    joblib.dump({"kind": "single", "pipelines": {}}, path)

    with pytest.raises(ArtifactError, match="unsigned"):
        load_bundle(path)


def test_a_signature_from_a_different_key_is_refused(bundle_path, monkeypatch):
    """Someone else's signing key does not make an artifact ours."""
    from config import settings

    monkeypatch.setattr(settings, "artifact_signing_key", "a-different-key")

    with pytest.raises(ArtifactError, match="integrity check"):
        verify_artifact(bundle_path)


def test_verification_happens_before_deserialisation(bundle_path, monkeypatch):
    """The property that actually matters: joblib is never reached on a bad file.

    A signature checked *after* loading would still fail the test above while
    having already executed whatever the pickle contained.
    """
    bundle_path.write_bytes(b"not a pickle at all")
    called = False

    def _tripwire(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("joblib.load was reached for an unverified artifact")

    monkeypatch.setattr(joblib, "load", _tripwire)

    with pytest.raises(ArtifactError):
        load_bundle(bundle_path)
    assert called is False


def test_signing_is_wired_into_training(client, fixture_csv, isolated_storage):
    """End to end: the pipeline signs what it writes, and predict verifies it."""
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"))
    run_key = job["result"]["run_key"]
    model_path = isolated_storage["model_dir"] / f"{run_key}_model.pkl"

    assert signature_path(model_path).exists()

    row = dict.fromkeys(job["result"]["features"], 1)
    assert client.post(f"/api/predict/{run_key}/row", json={"row": row}).status_code == 200

    # Swap the model for one this application did not write; prediction must
    # stop rather than load it.
    joblib.dump({"kind": "single", "pipelines": {"evil": None}}, model_path)
    response = client.post(f"/api/predict/{run_key}/row", json={"row": row})

    assert response.status_code == 409  # ArtifactError
    assert "integrity check" in response.json()["detail"]


def test_resigning_makes_a_replaced_artifact_loadable_again(bundle_path):
    """The operator's escape hatch, and proof the signature tracks the bytes."""
    joblib.dump({"kind": "single", "pipelines": {"replacement": 1}}, bundle_path)
    with pytest.raises(ArtifactError):
        verify_artifact(bundle_path)

    sign_artifact(bundle_path)
    assert load_bundle(bundle_path)["pipelines"] == {"replacement": 1}
