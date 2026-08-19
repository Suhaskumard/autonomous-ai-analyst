"""Phase 8: what a hostile account cannot do.

The Phase 6 suite proves two users cannot *see* each other's data. It says
nothing about what one user can do to the other by being greedy rather than
nosy — exhaust the budget, fill the disk, or hold every worker. Isolation and
abuse resistance are different properties and the second one had no tests.

So these are written from the attacker's side: each one is an account trying to
take more than its share, and the assertion is that it is stopped, that the
refusal is attributable, and that the *other* account is unaffected. The last
part is the one that matters — a limit that stops everybody equally is an
outage, not a quota.
"""

from datetime import timedelta

import pytest

import audit
import db
from tests.conftest import upload_and_wait
from utils.helpers import now_utc

CLASSIFICATION = "clean_classification.csv"


def _spend(owner_id: str, cost: float, calls: int = 1) -> None:
    """Put usage rows in the table the way a real call would."""
    for _ in range(calls):
        db.record_usage(
            owner_id=owner_id,
            provider="fake",
            model="gemini-1.5-flash",
            input_tokens=1000,
            output_tokens=100,
            estimated_cost_usd=cost,
        )


# --- budget ------------------------------------------------------------------


def test_one_account_cannot_spend_past_its_daily_ceiling(authenticated, monkeypatch, fixture_csv):
    """The ceiling the per-conversation guard could never enforce."""
    from config import settings

    monkeypatch.setattr(settings, "principal_daily_spend_limit_usd", 1.00)
    client, alice, bob = authenticated["client"], authenticated["alice"], authenticated["bob"]

    run_key = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["result"]["run_key"]
    _spend(bob["user_id"], cost=0.60, calls=2)  # $1.20, over the line

    refused = client.post("/api/chat", json={"run_key": run_key, "query": "anything"}, headers=bob["headers"])

    assert refused.status_code == 429
    assert "daily spend" in refused.json()["detail"].lower()

    # And the point of a *per-principal* limit: alice is untouched by bob.
    alice_run = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=alice["headers"])["result"]["run_key"]
    allowed = client.post("/api/chat", json={"run_key": alice_run, "query": "anything"}, headers=alice["headers"])
    assert allowed.status_code == 200


def test_a_new_conversation_does_not_reset_the_account_ceiling(authenticated, monkeypatch, fixture_csv):
    """The exact hole in the Phase 5 guard.

    Per-conversation limits are reset by starting a conversation, which costs
    the client nothing. This asserts the account ceiling survives that.
    """
    from config import settings

    monkeypatch.setattr(settings, "principal_max_llm_calls_per_hour", 2)
    client, bob = authenticated["client"], authenticated["bob"]
    run_key = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["result"]["run_key"]

    _spend(bob["user_id"], cost=0.001, calls=2)

    # A brand new conversation every time — the old guard would allow all of them.
    for _ in range(2):
        response = client.post("/api/chat", json={"run_key": run_key, "query": "again"}, headers=bob["headers"])
        assert response.status_code == 429


def test_an_hourly_ceiling_is_a_window_not_a_lifetime_ban(authenticated, monkeypatch, fixture_csv):
    """Usage that has aged out of the window must stop counting."""
    from config import settings

    monkeypatch.setattr(settings, "principal_max_llm_calls_per_hour", 2)
    client, bob = authenticated["client"], authenticated["bob"]
    run_key = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["result"]["run_key"]

    _spend(bob["user_id"], cost=0.001, calls=5)
    # Backdate every row past the window.
    with db.session_scope() as session:
        for row in session.exec(db.select(db.UsageRecord)).all():
            row.created_at = now_utc() - timedelta(hours=2)
            session.add(row)

    allowed = client.post("/api/chat", json={"run_key": run_key, "query": "now?"}, headers=bob["headers"])
    assert allowed.status_code == 200


# --- disk --------------------------------------------------------------------


def test_one_account_cannot_fill_the_disk(authenticated, monkeypatch, fixture_csv):
    """Uploading is free and training writes a model; the ceiling is on stored runs."""
    from config import settings

    monkeypatch.setattr(settings, "principal_max_runs", 1)
    client, alice, bob = authenticated["client"], authenticated["alice"], authenticated["bob"]

    first = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])
    assert first["state"] == "completed"

    with fixture_csv("clean_regression.csv").open("rb") as handle:
        refused = client.post(
            "/api/upload",
            files={"file": ("clean_regression.csv", handle, "text/csv")},
            data={"mode": "auto"},
            headers=bob["headers"],
        )

    assert refused.status_code == 413
    assert "stored datasets" in refused.json()["detail"]

    # Refused before the bytes were read, and alice still has her allowance.
    assert upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=alice["headers"])["state"] == "completed"


# --- workers -----------------------------------------------------------------


def test_one_account_cannot_hold_every_worker(authenticated, monkeypatch, fixture_csv):
    """The third exhaustible resource, and the one the other two limits miss.

    An account under its spend ceiling and under its disk quota can still queue
    training run after training run, occupy every worker, and make the app
    unavailable for everyone else at the cost of a few small CSVs.
    """
    from config import settings

    monkeypatch.setattr(settings, "principal_max_concurrent_jobs", 2)
    client, alice, bob = authenticated["client"], authenticated["alice"], authenticated["bob"]

    # Two of bob's jobs are already in flight. Written straight to the table
    # because that is where the limit reads them from — with an out-of-process
    # queue the API that accepted the upload is not the process running it, so
    # a counter anywhere else would be per-replica and not a limit at all.
    for index in range(2):
        db.create_job(f"held-{index}", "Upload received", owner_id=bob["user_id"])

    with fixture_csv(CLASSIFICATION).open("rb") as handle:
        refused = client.post(
            "/api/upload",
            files={"file": (CLASSIFICATION, handle, "text/csv")},
            data={"mode": "auto"},
            headers=bob["headers"],
        )

    assert refused.status_code == 429
    assert "queued or training" in refused.json()["detail"]

    # Alice's workers are not bob's to hold.
    assert upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=alice["headers"])["state"] == "completed"


def test_a_finished_job_releases_its_slot(authenticated, monkeypatch, fixture_csv):
    """A ceiling that counted every job ever run would be a lifetime quota."""
    from config import settings

    monkeypatch.setattr(settings, "principal_max_concurrent_jobs", 1)
    client, bob = authenticated["client"], authenticated["bob"]

    assert upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["state"] == "completed"
    assert db.count_active_jobs(bob["user_id"]) == 0

    second = upload_and_wait(client, fixture_csv("clean_regression.csv"), headers=bob["headers"])
    assert second["state"] == "completed"


def test_a_refused_quota_is_recorded_against_the_account(authenticated, monkeypatch, fixture_csv):
    """An operator asking "who kept hitting the wall?" needs an answer."""
    from config import settings

    monkeypatch.setattr(settings, "principal_max_runs", 1)
    client, bob = authenticated["client"], authenticated["bob"]
    upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])

    with fixture_csv("clean_regression.csv").open("rb") as handle:
        client.post(
            "/api/upload",
            files={"file": ("clean_regression.csv", handle, "text/csv")},
            data={"mode": "auto"},
            headers=bob["headers"],
        )

    refusals = [e for e in db.list_audit(limit=50) if e["action"] == audit.QUOTA_REFUSED]
    assert refusals, "a refused quota left no audit trail"
    assert refusals[0]["actor_id"] == bob["user_id"]
    assert refusals[0]["target_id"] == "max_runs"


# --- key lifecycle -----------------------------------------------------------


def test_a_key_can_be_rotated_without_an_interruption(authenticated):
    """Issue, use both, revoke the old one — with no window where nothing works.

    This is the whole reason keys moved off the user row, and it is the exit
    criterion stated as a test: at no point in this sequence is the account
    without a working credential.
    """
    client, bob = authenticated["client"], authenticated["bob"]

    issued = client.post("/api/account/keys", json={"label": "rotation"}, headers=bob["headers"])
    assert issued.status_code == 200
    new_key = issued.json()["api_key"]
    new_headers = {"Authorization": f"Bearer {new_key}"}

    # Both work: this overlap is what makes a rollout possible.
    assert client.get("/api/auth/me", headers=bob["headers"]).status_code == 200
    assert client.get("/api/auth/me", headers=new_headers).status_code == 200

    keys = client.get("/api/account/keys", headers=bob["headers"]).json()["keys"]
    old_key_id = next(k["key_id"] for k in keys if k["label"] == "initial")

    revoked = client.delete(f"/api/account/keys/{old_key_id}", headers=bob["headers"])
    assert revoked.status_code == 200

    # Stated blast radius: the revoked key, and nothing else.
    assert client.get("/api/auth/me", headers=bob["headers"]).status_code == 401
    assert client.get("/api/auth/me", headers=new_headers).status_code == 200


def test_the_last_active_key_cannot_be_revoked(authenticated):
    """Otherwise an account can lock itself out and cannot issue its way back."""
    client, bob = authenticated["client"], authenticated["bob"]
    key_id = client.get("/api/account/keys", headers=bob["headers"]).json()["keys"][0]["key_id"]

    response = client.delete(f"/api/account/keys/{key_id}", headers=bob["headers"])

    assert response.status_code == 409
    assert client.get("/api/auth/me", headers=bob["headers"]).status_code == 200


def test_an_expired_key_stops_working(authenticated, client):
    """Expiry is enforced at authentication, not merely recorded."""
    bob = authenticated["bob"]
    issued = client.post("/api/account/keys", json={"label": "short", "ttl_days": 1}, headers=bob["headers"]).json()
    headers = {"Authorization": f"Bearer {issued['api_key']}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    with db.session_scope() as session:
        row = session.get(db.ApiKeyRecord, issued["key_id"])
        row.expires_at = now_utc() - timedelta(minutes=1)
        session.add(row)

    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_a_key_cannot_be_revoked_by_another_account(authenticated):
    """Someone else's key id is not an object you get to act on."""
    client, alice, bob = authenticated["client"], authenticated["alice"], authenticated["bob"]
    client.post("/api/account/keys", json={"label": "spare"}, headers=bob["headers"])
    bob_key_id = client.get("/api/account/keys", headers=bob["headers"]).json()["keys"][0]["key_id"]

    response = client.delete(f"/api/account/keys/{bob_key_id}", headers=alice["headers"])

    assert response.status_code == 404
    # And bob's key still works — the failed revocation was a no-op, not a partial one.
    assert client.get("/api/auth/me", headers=bob["headers"]).status_code == 200


def test_keys_are_never_returned_in_a_form_that_authenticates(authenticated):
    client, bob = authenticated["client"], authenticated["bob"]
    listing = client.get("/api/account/keys", headers=bob["headers"]).json()

    for key in listing["keys"]:
        assert "api_key" not in key
        assert "key_hash" not in key


def test_an_account_cannot_hoard_keys(authenticated, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "max_active_api_keys", 2)
    client, bob = authenticated["client"], authenticated["bob"]

    assert client.post("/api/account/keys", json={}, headers=bob["headers"]).status_code == 200
    third = client.post("/api/account/keys", json={}, headers=bob["headers"])

    assert third.status_code == 409


# --- audit -------------------------------------------------------------------


def test_every_destructive_action_names_a_principal_and_a_time(authenticated, fixture_csv):
    """The exit criterion, as a test."""
    client, alice, bob = authenticated["client"], authenticated["alice"], authenticated["bob"]
    run_key = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["result"]["run_key"]

    client.delete(f"/api/runs/{run_key}", headers=bob["headers"])
    client.post("/api/admin/users", json={"email": "carol@example.com"}, headers=alice["headers"])

    entries = client.get("/api/admin/audit", headers=alice["headers"]).json()["entries"]
    by_action = {entry["action"]: entry for entry in entries}

    deletion = by_action[audit.RUN_DELETED]
    assert deletion["actor_id"] == bob["user_id"]
    assert deletion["target_id"] == run_key
    assert deletion["at"], "an audit entry with no timestamp answers half the question"

    creation = by_action[audit.USER_CREATED]
    assert creation["actor_id"] == alice["user_id"]


def test_the_audit_log_outlives_what_it_records(authenticated, fixture_csv):
    """A deletion record that is deleted with the data proves nothing."""
    client, bob = authenticated["client"], authenticated["bob"]
    run_key = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["result"]["run_key"]

    client.delete("/api/account/data?confirm=true", headers=bob["headers"])

    assert db.get_run(run_key) is None
    actions = {entry["action"] for entry in db.list_audit(limit=50, actor_id=bob["user_id"])}
    assert audit.ACCOUNT_DATA_ERASED in actions


def test_the_unattended_purge_is_recorded_too(client, fixture_csv, monkeypatch):
    """The one destructive action nobody asks for still has to be attributable.

    Retention runs at startup, on every boot, with no principal behind it. It
    was for a while the only thing in this application that deleted data and
    left no record — the admin endpoint recorded its purge and the startup pass
    did not, which is exactly backwards: the unattended one is the one nobody
    will remember running.
    """
    import lifecycle
    from config import settings

    monkeypatch.setattr(settings, "retention_days", 30)
    run_key = upload_and_wait(client, fixture_csv(CLASSIFICATION))["result"]["run_key"]
    # Age it past its expiry rather than waiting thirty days for it.
    db.set_run_expiry(run_key, now_utc() - timedelta(days=1))

    summary = lifecycle.purge_expired()

    assert summary["runs_purged"] == 1
    entry = next(e for e in db.list_audit(limit=50) if e["action"] == audit.RETENTION_PURGED)
    assert entry["actor_id"] == lifecycle.SYSTEM_ACTOR
    assert entry["at"] is not None
    assert entry["detail"]["runs_purged"] == 1


def test_a_purge_that_deletes_nothing_writes_no_entry(client):
    """A row per restart saying "deleted nothing" is how a log stops being read."""
    import lifecycle

    before = len(db.list_audit(limit=500, action=audit.RETENTION_PURGED))
    lifecycle.purge_expired()
    assert len(db.list_audit(limit=500, action=audit.RETENTION_PURGED)) == before


def test_the_audit_log_is_not_readable_by_a_non_admin(authenticated):
    client, bob = authenticated["client"], authenticated["bob"]
    assert client.get("/api/admin/audit", headers=bob["headers"]).status_code == 403


def test_an_account_sees_its_own_actions_and_not_the_admin_s(authenticated, fixture_csv):
    """Self-service history, without turning it into an admin console."""
    client, alice, bob = authenticated["client"], authenticated["alice"], authenticated["bob"]
    client.post("/api/account/keys", json={"label": "mine"}, headers=bob["headers"])
    client.post("/api/admin/users", json={"email": "dave@example.com"}, headers=alice["headers"])

    mine = client.get("/api/account/audit", headers=bob["headers"]).json()["entries"]

    assert {entry["action"] for entry in mine} == {audit.KEY_ISSUED}
    assert all(entry["actor_id"] == bob["user_id"] for entry in mine)


def test_audit_entries_carry_no_dataset_content(authenticated, fixture_csv):
    """The log is read by more people than the data is."""
    client, bob = authenticated["client"], authenticated["bob"]
    run_key = upload_and_wait(client, fixture_csv(CLASSIFICATION), headers=bob["headers"])["result"]["run_key"]
    client.delete(f"/api/runs/{run_key}", headers=bob["headers"])

    entry = next(e for e in db.list_audit(limit=50) if e["action"] == audit.RUN_DELETED)
    serialised = str(entry)

    for column in ("age", "spend", "region", "churn"):
        assert column not in serialised, f"the audit log leaked the column name {column!r}"


# --- secrets out of the environment ------------------------------------------
#
# Environment variables are readable by anything that can list a process, take a
# core dump, or run `docker inspect`, and they end up in shell history and CI
# logs. Docker secrets, Kubernetes secret volumes and systemd credentials all
# present a secret as a *file*, so `<NAME>_FILE` is the one convention that lets
# any of them be pointed at this application.


@pytest.mark.parametrize(
    ("field", "variable", "secret"),
    [
        ("gemini_api_key", "GEMINI_API_KEY_FILE", "AIza-not-a-real-key"),
        ("artifact_signing_key", "ARTIFACT_SIGNING_KEY_FILE", "a-signing-key"),
        ("auth_bootstrap_token", "AUTH_BOOTSTRAP_TOKEN_FILE", "a-bootstrap-token"),
        ("database_url", "DATABASE_URL_FILE", "postgresql://u:p@host/analyst"),
    ],
)
def test_a_secret_can_come_from_a_file(monkeypatch, tmp_path, field, variable, secret):
    from config import Settings, _load_file_backed_secrets

    path = tmp_path / "secret"
    path.write_text(f"{secret}\n", encoding="utf-8")  # trailing newline: files have them
    monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.setenv(variable, str(path))

    settings = Settings(**{field: None} if field != "database_url" else {})
    # A field the constructor was handed counts as explicitly set, which is the
    # thing that must not shadow the file; clear it the way an unset env does.
    settings.model_fields_set.discard(field)
    _load_file_backed_secrets(settings)

    assert getattr(settings, field) == secret


def test_the_database_url_can_come_from_a_file_despite_having_a_default(monkeypatch, tmp_path):
    """The case a truthiness check gets wrong, and did.

    `database_url` defaults to the local SQLite file, so "does it have a value"
    is true before anyone sets anything. Testing that instead of "did the
    operator set it" made DATABASE_URL_FILE unusable: every startup reported a
    conflict with a DATABASE_URL nobody had given, and then ignored the file.
    """
    from config import Settings, _load_file_backed_secrets

    path = tmp_path / "database-url"
    path.write_text("postgresql://user:password@db/analyst", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL_FILE", str(path))

    settings = Settings()
    settings.model_fields_set.discard("database_url")
    _load_file_backed_secrets(settings)

    assert settings.database_url == "postgresql://user:password@db/analyst"


def test_an_explicit_variable_beats_the_file(monkeypatch, tmp_path):
    """Both set is a misconfiguration; resolving it silently is the hazard."""
    from config import Settings, _load_file_backed_secrets

    path = tmp_path / "secret"
    path.write_text("from-the-file", encoding="utf-8")
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", "from-the-environment")
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY_FILE", str(path))

    settings = Settings()
    _load_file_backed_secrets(settings)

    assert settings.artifact_signing_key == "from-the-environment"


def test_an_empty_secret_file_is_fatal(monkeypatch, tmp_path):
    """Falling back to "no key" would look like a configuration choice."""
    from config import Settings, _load_file_backed_secrets

    path = tmp_path / "empty"
    path.write_text("   \n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(path))

    settings = Settings(gemini_api_key=None)
    settings.model_fields_set.discard("gemini_api_key")

    with pytest.raises(ValueError, match="which is empty"):
        _load_file_backed_secrets(settings)


def test_a_missing_secret_file_is_fatal(monkeypatch, tmp_path):
    """Same reasoning: a path that does not resolve is not a quiet default."""
    from config import Settings, _load_file_backed_secrets

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(tmp_path / "absent"))

    settings = Settings(gemini_api_key=None)
    settings.model_fields_set.discard("gemini_api_key")

    with pytest.raises(OSError):
        _load_file_backed_secrets(settings)


# --- the limits stay off unless asked for ------------------------------------


@pytest.mark.parametrize(
    "setting",
    [
        "principal_max_llm_calls_per_hour",
        "principal_daily_spend_limit_usd",
        "principal_max_runs",
        "principal_storage_quota_mb",
        "principal_max_concurrent_jobs",
    ],
)
def test_every_new_limit_defaults_to_off(setting):
    """Phase 6's rule, restated: a single-operator install adds no ceilings."""
    from config import Settings

    assert not getattr(Settings(), setting)


def test_limits_do_nothing_while_auth_is_off(client, fixture_csv, monkeypatch):
    """With one implicit owner, a per-account limit is a per-machine limit."""
    from config import settings

    monkeypatch.setattr(settings, "principal_max_runs", 1)

    assert upload_and_wait(client, fixture_csv(CLASSIFICATION))["state"] == "completed"
    assert upload_and_wait(client, fixture_csv("clean_regression.csv"))["state"] == "completed"
