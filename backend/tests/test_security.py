"""Path traversal into pickle deserialization must be impossible.

Regression guard: dataset_hash was interpolated into a file path with no format
check and handed to joblib.load, which is remote code execution.
"""

import pytest
from fastapi import HTTPException

from utils.helpers import METADATA_DIR
from utils.security import artifact_path, validate_dataset_hash

TRAVERSAL_ATTEMPTS = [
    "../../../../etc/passwd",
    "..\\..\\windows\\system32\\config",
    "a" * 63 + "/",
    "a" * 63 + "\\",
    "../" + "a" * 61,
    "%2e%2e%2f" + "a" * 55,
    "a" * 64 + "/../../evil",
    "\x00" + "a" * 63,
]

MALFORMED = [
    "A" * 64,  # uppercase hex
    "a" * 63,  # too short
    "a" * 65,  # too long
    "g" * 64,  # not hex
    "",
    "null",
]


@pytest.mark.parametrize("candidate", TRAVERSAL_ATTEMPTS + MALFORMED)
def test_invalid_keys_are_rejected(candidate):
    with pytest.raises(HTTPException) as excinfo:
        validate_dataset_hash(candidate)
    assert excinfo.value.status_code == 400


def test_valid_key_is_accepted():
    key = "0123456789abcdef" * 4
    assert validate_dataset_hash(key) == key


def test_artifact_path_stays_under_its_root():
    key = "0123456789abcdef" * 4
    path = artifact_path(METADATA_DIR, key, ".json")
    assert path.parent == METADATA_DIR.resolve()


@pytest.mark.parametrize("candidate", TRAVERSAL_ATTEMPTS)
def test_artifact_path_refuses_traversal(candidate):
    with pytest.raises(HTTPException):
        artifact_path(METADATA_DIR, candidate, "_model.pkl")


@pytest.mark.parametrize("route", ["/api/insights/{}", "/api/runs/{}"])
def test_routes_reject_non_hex_keys(client, route):
    response = client.get(route.format("not-a-hash"))
    assert response.status_code == 400


def test_predict_route_rejects_traversal_before_loading_anything(client, tmp_path):
    response = client.post(
        "/api/predict/..%2F..%2Fetc%2Fpasswd",
        files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code in (400, 404)


def test_chat_route_rejects_non_hex_key(client):
    response = client.post("/api/chat", json={"run_key": "../../etc/passwd", "query": "hi"})
    assert response.status_code == 400
