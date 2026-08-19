"""Phase 10: read-only, expiring share links."""

from tests.conftest import upload_and_wait


def _trained_run(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), target_column="churn")
    return job["result"]["run_key"]


def test_creating_a_link_requires_it_to_be_turned_on(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", False)
    run_key = _trained_run(client, fixture_csv)
    response = client.post(f"/api/runs/{run_key}/share")
    assert response.status_code == 403


def test_a_link_serves_the_dashboard_payload_with_no_auth(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", True)
    run_key = _trained_run(client, fixture_csv)

    created = client.post(f"/api/runs/{run_key}/share")
    assert created.status_code == 200, created.text
    token = created.json()["token"]
    assert token

    shared = client.get(f"/api/share/{token}")
    assert shared.status_code == 200, shared.text
    body = shared.json()
    assert body["run_key"] == run_key
    assert body["status"] == "shared"


def test_an_unknown_token_is_404(client, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", True)
    response = client.get("/api/share/not-a-real-token")
    assert response.status_code == 404


def test_a_revoked_link_stops_working(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", True)
    run_key = _trained_run(client, fixture_csv)
    token = client.post(f"/api/runs/{run_key}/share").json()["token"]

    revoke = client.delete(f"/api/share/{token}")
    assert revoke.status_code == 200

    shared = client.get(f"/api/share/{token}")
    assert shared.status_code == 404


def test_an_expired_link_stops_working(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", True)
    monkeypatch.setattr(settings, "share_link_default_ttl_days", 1)
    run_key = _trained_run(client, fixture_csv)
    token = client.post(f"/api/runs/{run_key}/share").json()["token"]

    import db
    from utils.helpers import now_utc

    with db.session_scope() as session:
        row = session.get(db.ShareLinkRecord, token)
        row.expires_at = now_utc()
        session.add(row)

    shared = client.get(f"/api/share/{token}")
    assert shared.status_code == 404


def test_the_token_is_never_returned_by_the_list_endpoint(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", True)
    run_key = _trained_run(client, fixture_csv)
    client.post(f"/api/runs/{run_key}/share")

    listed = client.get(f"/api/runs/{run_key}/share")
    assert listed.status_code == 200
    links = listed.json()["links"]
    assert len(links) == 1
    assert "token" not in links[0]


def test_deleting_a_run_deletes_its_share_links(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", True)
    run_key = _trained_run(client, fixture_csv)
    token = client.post(f"/api/runs/{run_key}/share").json()["token"]

    delete = client.delete(f"/api/runs/{run_key}")
    assert delete.status_code == 200

    shared = client.get(f"/api/share/{token}")
    assert shared.status_code == 404


def test_cross_owner_creation_is_refused_as_404(client, fixture_csv, monkeypatch):
    """Same 404-not-403 reasoning as every other owner-scoped endpoint."""
    from config import settings

    monkeypatch.setattr(settings, "share_links_enabled", True)

    import auth

    monkeypatch.setattr(settings, "auth_enabled", True)
    alice_record, alice_key = auth.create_user("alice-share@example.com")
    bob_record, bob_key = auth.create_user("bob-share@example.com")
    alice_headers = {"Authorization": f"Bearer {alice_key}"}
    bob_headers = {"Authorization": f"Bearer {bob_key}"}

    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), headers=alice_headers, target_column="churn")
    run_key = job["result"]["run_key"]

    response = client.post(f"/api/runs/{run_key}/share", headers=bob_headers)
    assert response.status_code == 404
