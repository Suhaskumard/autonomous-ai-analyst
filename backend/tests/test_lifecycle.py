"""Phase 6: retention, deletion, and the privacy statement.

The exit criterion is "every stored artifact has an owner and an expiry". An
expiry that nothing acts on is not an expiry, so most of this file is about the
purge actually removing bytes from disk rather than only rows from a table.
"""

from datetime import timedelta

import db
import lifecycle
from tests.conftest import upload_and_wait
from utils.helpers import now_utc

CLASSIFICATION = "clean_classification.csv"


def _artifacts(isolated_storage, run_key):
    return [
        isolated_storage["metadata_dir"] / f"{run_key}.json",
        isolated_storage["metadata_dir"] / f"{run_key}_data.csv",
        isolated_storage["model_dir"] / f"{run_key}_model.pkl",
        isolated_storage["model_dir"] / f"{run_key}_model.pkl.sig",
    ]


# --- expiry -----------------------------------------------------------------


def test_retention_off_means_no_expiry(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "retention_days", 0)
    assert lifecycle.expiry_for_new_run() is None


def test_retention_on_sets_an_expiry_that_far_ahead(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "retention_days", 7)
    expiry = lifecycle.expiry_for_new_run()

    assert expiry is not None
    assert timedelta(days=6, hours=23) < (expiry - now_utc()) <= timedelta(days=7)


def test_every_run_records_an_owner_and_an_expiry(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "retention_days", 30)
    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    run = db.get_run(job["result"]["run_key"])

    assert run["owner_id"] == db.LOCAL_OWNER_ID
    assert run["expires_at"] is not None


def test_an_unexpired_run_survives_the_purge(client, fixture_csv, monkeypatch, isolated_storage):
    from config import settings

    monkeypatch.setattr(settings, "retention_days", 30)
    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    run_key = job["result"]["run_key"]

    assert lifecycle.purge_expired()["runs_purged"] == 0
    assert all(path.exists() for path in _artifacts(isolated_storage, run_key))


def test_purging_an_expired_run_removes_its_bytes_and_its_rows(client, fixture_csv, monkeypatch, isolated_storage):
    from config import settings

    monkeypatch.setattr(settings, "retention_days", 30)
    job = upload_and_wait(client, fixture_csv(CLASSIFICATION))
    run_key = job["result"]["run_key"]
    artifacts = _artifacts(isolated_storage, run_key)
    assert all(path.exists() for path in artifacts)

    # A conversation about the dataset is as sensitive as the dataset.
    db.create_conversation("conv-1", run_key, title="questions")

    db.set_run_expiry(run_key, now_utc() - timedelta(days=1))
    summary = lifecycle.purge_expired()

    assert summary["runs_purged"] == 1
    assert summary["artifacts_removed"] == len(artifacts)
    assert summary["conversations_removed"] == 1
    assert not any(path.exists() for path in artifacts)
    assert db.get_run(run_key) is None
    assert db.get_conversation("conv-1") is None


def test_deleting_an_account_erases_only_that_account(authenticated, fixture_csv, isolated_storage):
    client = authenticated["client"]
    alice, bob = authenticated["alice"], authenticated["bob"]

    alice_run = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=alice["headers"])["result"]["run_key"]
    bob_run = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["result"]["run_key"]

    # Refuses to fire without confirmation, because it cannot be undone.
    assert client.delete("/api/account/data", headers=alice["headers"]).status_code == 400

    response = client.delete("/api/account/data?confirm=true", headers=alice["headers"])
    assert response.status_code == 200
    assert response.json()["runs_deleted"] == 1

    assert not any(path.exists() for path in _artifacts(isolated_storage, alice_run))
    assert db.get_run(alice_run) is None
    # Bob is untouched.
    assert all(path.exists() for path in _artifacts(isolated_storage, bob_run))
    assert db.get_run(bob_run) is not None


def test_the_privacy_statement_is_machine_readable(client):
    """ "Where does my data go" as data, so the UI cannot drift from the policy."""
    policy = client.get("/api/privacy").json()

    assert policy["retention_days"] == 0
    assert policy["expiry_enabled"] is False
    assert policy["storage_backend"] == "local"
    # The three claims that matter most, stated rather than implied.
    assert any("rows" in item for item in policy["never_leaves_this_machine"])
    assert policy["leaves_this_machine"]
    assert "DELETE /api/account/data" in policy["deletion"]


def test_the_retention_endpoint_is_admin_only(authenticated):
    client = authenticated["client"]

    assert client.post("/api/admin/retention/purge", headers=authenticated["bob"]["headers"]).status_code == 403
    assert client.post("/api/admin/retention/purge", headers=authenticated["alice"]["headers"]).status_code == 200
