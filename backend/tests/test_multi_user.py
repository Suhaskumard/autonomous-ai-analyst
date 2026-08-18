"""Phase 6: authentication, per-user isolation, and the admin surface.

The exit criterion this file exists to prove is "two users cannot see each
other's data". That is not one assertion — it is the same assertion repeated at
every endpoint that takes a run key, because isolation that holds on /runs and
leaks on /insights is not isolation.
"""

import pytest

import auth
import db
from tests.conftest import upload_and_wait

CLASSIFICATION = "clean_classification.csv"


# --- authentication ---------------------------------------------------------


def test_auth_disabled_needs_no_credential(client):
    """The single-operator install keeps working exactly as it did."""
    assert client.get("/api/runs").status_code == 200

    me = client.get("/api/auth/me").json()
    assert me["auth_enabled"] is False
    assert me["user_id"] == db.LOCAL_OWNER_ID


def test_enabled_auth_rejects_missing_and_bad_credentials(authenticated):
    client = authenticated["client"]

    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers={"Authorization": "Bearer nope"}).status_code == 401
    # A syntactically plausible key that is simply not ours must be rejected
    # the same way as a malformed one; anything else is an oracle.
    plausible = {"Authorization": "Bearer analyst_" + "a" * 43}
    assert client.get("/api/runs", headers=plausible).status_code == 401
    assert client.get("/api/runs", headers=authenticated["alice"]["headers"]).status_code == 200


def test_api_key_is_never_stored_in_recoverable_form(authenticated):
    api_key = authenticated["bob"]["api_key"]
    stored = db.get_user_by_key_hash(auth.hash_api_key(api_key))

    assert stored is not None
    assert api_key not in stored["api_key_hash"]
    assert len(stored["api_key_hash"]) == 64  # sha256 hex, not the key itself


def test_deactivated_account_loses_access(authenticated):
    client = authenticated["client"]
    bob = authenticated["bob"]

    assert client.get("/api/runs", headers=bob["headers"]).status_code == 200

    deactivate = f"/api/admin/users/{bob['user_id']}/deactivate"
    assert client.post(deactivate, headers=authenticated["alice"]["headers"]).status_code == 200
    assert client.get("/api/runs", headers=bob["headers"]).status_code == 401


def test_admin_endpoints_refuse_a_normal_account(authenticated):
    client = authenticated["client"]

    assert client.get("/api/admin/users", headers=authenticated["bob"]["headers"]).status_code == 403
    assert client.get("/api/admin/users", headers=authenticated["alice"]["headers"]).status_code == 200


def test_registration_is_closed_without_a_bootstrap_token(client, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "auth_bootstrap_token", None)

    assert client.post("/api/auth/register", json={"email": "first@example.com"}).status_code == 403


def test_bootstrap_creates_exactly_one_first_account(client, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "auth_bootstrap_token", "let-me-in")

    payload = {"email": "first@example.com"}
    assert client.post("/api/auth/register", json=payload, headers={"X-Bootstrap-Token": "guess"}).status_code == 403

    created = client.post("/api/auth/register", json=payload, headers={"X-Bootstrap-Token": "let-me-in"})
    assert created.status_code == 200
    body = created.json()
    assert body["is_admin"] is True
    assert body["api_key"].startswith(auth.KEY_PREFIX)

    # The door closes behind the first account.
    second = client.post(
        "/api/auth/register",
        json={"email": "second@example.com"},
        headers={"X-Bootstrap-Token": "let-me-in"},
    )
    assert second.status_code == 409


# --- isolation --------------------------------------------------------------


@pytest.fixture
def alice_run(authenticated, fixture_csv):
    """A finished run owned by Alice. Bob holds a valid key and owns nothing."""
    job = upload_and_wait(
        authenticated["client"],
        fixture_csv(CLASSIFICATION),
        headers=authenticated["alice"]["headers"],
    )
    assert job["state"] == "completed", job
    return job["result"]["run_key"]


def test_the_same_csv_uploaded_by_two_users_is_two_runs(authenticated, fixture_csv, alice_run):
    """Content-addressing must not deduplicate across owners.

    Without the owner in the cache key, Bob uploading Alice's file would hit her
    artifacts and be handed her dataset snapshot as though he had trained it.
    """
    job = upload_and_wait(
        authenticated["client"],
        fixture_csv(CLASSIFICATION),
        headers=authenticated["bob"]["headers"],
    )
    assert job["state"] == "completed", job
    assert job["result"]["run_key"] != alice_run
    # He trained his own copy rather than reusing hers.
    assert job["result"]["status"] == "trained"


def test_run_listing_is_scoped_to_the_owner(authenticated, alice_run):
    client = authenticated["client"]

    alice_runs = client.get("/api/runs", headers=authenticated["alice"]["headers"]).json()
    assert [run["run_key"] for run in alice_runs["runs"]] == [alice_run]

    bob_runs = client.get("/api/runs", headers=authenticated["bob"]["headers"]).json()
    assert bob_runs["runs"] == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/runs/{key}"),
        ("get", "/api/runs/{key}/result"),
        ("get", "/api/insights/{key}"),
        ("get", "/api/chat/{key}/conversations"),
        ("delete", "/api/runs/{key}"),
    ],
)
def test_another_users_run_is_not_reachable(authenticated, alice_run, method, path):
    """404, not 403: confirming the key exists would leak that it does."""
    client = authenticated["client"]
    url = path.format(key=alice_run)

    assert getattr(client, method)(url, headers=authenticated["bob"]["headers"]).status_code == 404
    # ...and the owner is unaffected by the attempt.
    assert getattr(client, method)(url, headers=authenticated["alice"]["headers"]).status_code == 200


def test_another_users_run_cannot_be_predicted_against(authenticated, alice_run, fixture_csv):
    client = authenticated["client"]

    with fixture_csv(CLASSIFICATION).open("rb") as handle:
        response = client.post(
            f"/api/predict/{alice_run}",
            files={"file": (CLASSIFICATION, handle, "text/csv")},
            headers=authenticated["bob"]["headers"],
        )
    assert response.status_code == 404


def test_another_users_run_cannot_be_chatted_about(authenticated, alice_run):
    response = authenticated["client"].post(
        "/api/chat",
        json={"run_key": alice_run, "query": "what is in this dataset?"},
        headers=authenticated["bob"]["headers"],
    )
    assert response.status_code == 404


def test_job_status_is_scoped_to_its_owner(authenticated, fixture_csv):
    client = authenticated["client"]

    with fixture_csv(CLASSIFICATION).open("rb") as handle:
        started = client.post(
            "/api/upload",
            files={"file": (CLASSIFICATION, handle, "text/csv")},
            data={"mode": "auto"},
            headers=authenticated["alice"]["headers"],
        )
    job_id = started.json()["job_id"]

    assert client.get(f"/api/upload/status/{job_id}", headers=authenticated["alice"]["headers"]).status_code == 200
    assert client.get(f"/api/upload/status/{job_id}", headers=authenticated["bob"]["headers"]).status_code == 404


def test_compare_refuses_a_run_the_caller_does_not_own(authenticated, alice_run):
    response = authenticated["client"].get(
        f"/api/runs/compare?keys={alice_run}",
        headers=authenticated["bob"]["headers"],
    )
    assert response.status_code == 404


def test_an_admin_does_not_get_to_read_someone_elses_data(authenticated, fixture_csv):
    """Being an admin is authority over accounts, not over their datasets."""
    client = authenticated["client"]
    job = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=authenticated["bob"]["headers"])
    bob_run = job["result"]["run_key"]

    # Alice is the admin here.
    assert client.get(f"/api/runs/{bob_run}", headers=authenticated["alice"]["headers"]).status_code == 404
    assert client.get("/api/runs", headers=authenticated["alice"]["headers"]).json()["runs"] == []


def test_an_artifact_with_no_owner_is_unreachable_while_auth_is_on(authenticated, alice_run, isolated_storage):
    """A restored volume must not hand a dataset to whoever asks for it first.

    Artifacts can outlive their registry row — a fresh database, a restored
    volume, a deleted row. The files on disk carry no owner, so with auth on
    there is nothing to authorise against and the only safe answer is 404.
    """
    client = authenticated["client"]
    assert db.delete_run(alice_run) is True

    for headers in (authenticated["alice"]["headers"], authenticated["bob"]["headers"]):
        assert client.get(f"/api/insights/{alice_run}", headers=headers).status_code == 404
        assert client.get(f"/api/runs/{alice_run}/result", headers=headers).status_code == 404

    # The metadata really is still sitting there; it is the check that stops it.
    assert (isolated_storage["metadata_dir"] / f"{alice_run}.json").exists()
