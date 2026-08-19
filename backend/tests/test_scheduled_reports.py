"""Phase 10: scheduled reports, delivered by email on an external cron's tick."""

from datetime import timedelta

import pytest

from tests.conftest import upload_and_wait


@pytest.fixture
def scheduling_on(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "scheduled_reports_enabled", True)
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_from_address", "reports@example.com")
    return settings


def _trained_run(client, fixture_csv):
    job = upload_and_wait(client, fixture_csv("clean_classification.csv"), target_column="churn")
    return job["result"]["run_key"]


def test_creating_a_schedule_requires_it_to_be_turned_on(client, fixture_csv):
    run_key = _trained_run(client, fixture_csv)
    response = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["someone@example.com"], "interval": "weekly"},
    )
    assert response.status_code == 403


def test_enabled_without_smtp_configured_is_refused(client, fixture_csv, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "scheduled_reports_enabled", True)
    monkeypatch.setattr(settings, "smtp_host", None)
    run_key = _trained_run(client, fixture_csv)

    response = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["someone@example.com"], "interval": "weekly"},
    )
    assert response.status_code == 503
    assert "SMTP_HOST" in response.json()["detail"]


def test_a_schedule_is_created_and_listed(client, fixture_csv, scheduling_on):
    run_key = _trained_run(client, fixture_csv)
    created = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["a@example.com", "b@example.com"], "interval": "daily"},
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["interval"] == "daily"
    assert body["recipients"] == ["a@example.com", "b@example.com"]
    assert body["next_run_at"]

    listed = client.get(f"/api/runs/{run_key}/report-schedules")
    assert listed.status_code == 200
    assert len(listed.json()["schedules"]) == 1


def test_an_invalid_recipient_is_refused(client, fixture_csv, scheduling_on):
    run_key = _trained_run(client, fixture_csv)
    response = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["not-an-email"], "interval": "weekly"},
    )
    assert response.status_code == 400


def test_an_unknown_interval_is_refused(client, fixture_csv, scheduling_on):
    run_key = _trained_run(client, fixture_csv)
    response = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["a@example.com"], "interval": "hourly"},
    )
    assert response.status_code == 400


def test_a_schedule_can_be_deleted(client, fixture_csv, scheduling_on):
    run_key = _trained_run(client, fixture_csv)
    schedule_id = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["a@example.com"], "interval": "weekly"},
    ).json()["schedule_id"]

    deleted = client.delete(f"/api/report-schedules/{schedule_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/runs/{run_key}/report-schedules").json()["schedules"] == []


def test_run_due_only_picks_up_schedules_that_are_actually_due(client, fixture_csv, scheduling_on, monkeypatch):
    import db
    import report_schedule

    run_key = _trained_run(client, fixture_csv)
    schedule_id = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["a@example.com"], "interval": "weekly"},
    ).json()["schedule_id"]

    enqueued = []
    monkeypatch.setattr(report_schedule, "process_due_schedule", lambda sid: enqueued.append(sid))

    # Not due yet: created just now with a one-week interval.
    assert report_schedule.run_due()["due"] == 0

    from utils.helpers import now_utc

    with db.session_scope() as session:
        row = session.get(db.ReportScheduleRecord, schedule_id)
        row.next_run_at = now_utc() - timedelta(minutes=1)
        session.add(row)

    assert report_schedule.run_due()["due"] == 1


def test_a_due_schedule_sends_an_email_and_advances_its_next_run(client, fixture_csv, scheduling_on, monkeypatch):
    import db
    import report_schedule

    run_key = _trained_run(client, fixture_csv)
    schedule_id = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["a@example.com"], "interval": "daily"},
    ).json()["schedule_id"]

    sent = {}

    def _fake_send(to_addresses, subject, body_text):
        sent["to"] = to_addresses
        sent["subject"] = subject
        sent["body"] = body_text

    monkeypatch.setattr(report_schedule, "_send_email", _fake_send)
    report_schedule.process_due_schedule(schedule_id)

    assert sent["to"] == ["a@example.com"]
    assert run_key[:10] in sent["subject"]
    assert sent["body"]

    after = db.get_report_schedule(schedule_id)
    assert after["last_status"] == "sent"
    assert after["last_run_at"] is not None


def test_a_failing_send_still_advances_the_schedule(client, fixture_csv, scheduling_on, monkeypatch):
    """A schedule that fails must not retry in a tight loop — it fails on its
    normal cadence and says so via last_status."""
    import db
    import report_schedule

    run_key = _trained_run(client, fixture_csv)
    schedule_id = client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["a@example.com"], "interval": "daily"},
    ).json()["schedule_id"]

    before = db.get_report_schedule(schedule_id)

    def _explode(*args, **kwargs):
        raise RuntimeError("smtp is down")

    monkeypatch.setattr(report_schedule, "_send_email", _explode)
    report_schedule.process_due_schedule(schedule_id)

    after = db.get_report_schedule(schedule_id)
    assert after["last_status"] == "failed"
    assert after["next_run_at"] != before["next_run_at"]


def test_deleting_a_run_deletes_its_schedules(client, fixture_csv, scheduling_on):
    import db

    run_key = _trained_run(client, fixture_csv)
    client.post(
        f"/api/runs/{run_key}/report-schedules",
        json={"recipients": ["a@example.com"], "interval": "weekly"},
    )

    assert client.delete(f"/api/runs/{run_key}").status_code == 200
    assert db.list_report_schedules(run_key=run_key) == []


def test_run_due_is_admin_only(client, fixture_csv, scheduling_on, monkeypatch):
    import auth
    from config import settings

    monkeypatch.setattr(settings, "auth_enabled", True)
    _, bob_key = auth.create_user("bob-sched@example.com")

    response = client.post("/api/report-schedules/run-due", headers={"Authorization": f"Bearer {bob_key}"})
    assert response.status_code == 403
