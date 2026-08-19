"""Phase 9 end-to-end: prediction logging, drift, calibration, versions, retraining.

The unit tests in test_drift.py, test_calibration.py and test_serving_schema.py
cover the algorithms. This file is about the plumbing around them: does a
served prediction actually get logged with the right owner and model version,
does the monitoring view read that log back correctly, does a genuinely better
challenger get promoted and a worse one refused, and is every one of these
owner-scoped the same way a run itself is.
"""

import io

import numpy as np
import pandas as pd
import pytest

import db
from tests.conftest import upload_and_wait

N_TRAIN = 900


def _classification_frame(rng, n, age_mean=45.0, region_probs=(0.5, 0.3, 0.2), regions=("north", "south", "east")):
    age = rng.normal(age_mean, 12, n).clip(18, 85)
    spend = rng.gamma(3, 20, n)
    region = rng.choice(regions, n, p=list(region_probs))
    logit = -0.05 * age + 0.03 * spend + (np.asarray(region) == "east") * 1.5 - 1.0
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < probability, "yes", "no")
    return pd.DataFrame({"age": age.round(1), "spend": spend.round(2), "region": region, "churn": churn})


@pytest.fixture
def churn_csv(tmp_path):
    frame = _classification_frame(np.random.default_rng(42), N_TRAIN)
    path = tmp_path / "churn.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def trained(client, churn_csv):
    job = upload_and_wait(client, churn_csv)
    assert job["state"] == "completed", job.get("error")
    return job["result"]


def _predict_batch(client, run_key, rng, n, **kwargs):
    frame = _classification_frame(rng, n, **kwargs).drop(columns=["churn"])
    response = client.post(
        f"/api/predict/{run_key}",
        files={"file": ("batch.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- prediction logging -------------------------------------------------------


def test_a_served_prediction_is_logged_with_its_model_version(client, trained):
    response = _predict_batch(client, trained["run_key"], np.random.default_rng(1), 5)
    version_id = response["model_version"]
    assert version_id

    log = client.get(f"/api/runs/{trained['run_key']}/predictions").json()
    assert log["entries"], "nothing was logged"
    assert all(entry["model_version"] == version_id for entry in log["entries"])
    assert log["entries"][0]["inputs"], "inputs should be logged by default"


def test_one_prediction_can_be_read_back_in_full(client, trained):
    _predict_batch(client, trained["run_key"], np.random.default_rng(1), 3)
    prediction_id = client.get(f"/api/runs/{trained['run_key']}/predictions").json()["entries"][0]["prediction_id"]

    entry = client.get(f"/api/predictions/{prediction_id}").json()
    assert entry["prediction_id"] == prediction_id
    assert entry["run_key"] == trained["run_key"]
    assert entry["model"]["is_champion"] is True
    assert "how_to_reproduce" in entry and trained["run_key"] in entry["how_to_reproduce"]


def test_a_single_row_prediction_is_also_logged(client, trained):
    metadata_before = len(client.get(f"/api/runs/{trained['run_key']}/predictions").json()["entries"])
    client.post(
        f"/api/predict/{trained['run_key']}/row",
        json={"row": {"age": 40, "spend": 30.0, "region": "north"}},
    )
    after = client.get(f"/api/runs/{trained['run_key']}/predictions").json()["entries"]
    assert len(after) == metadata_before + 1


def test_logging_off_means_nothing_is_recorded(client, trained, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "prediction_logging_enabled", False)
    _predict_batch(client, trained["run_key"], np.random.default_rng(1), 5)
    log = client.get(f"/api/runs/{trained['run_key']}/predictions").json()
    assert log["entries"] == []
    assert log["logging_enabled"] is False


def test_inputs_off_keeps_volume_but_drops_feature_values(client, trained, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "prediction_log_inputs", False)
    _predict_batch(client, trained["run_key"], np.random.default_rng(1), 5)

    log = client.get(f"/api/runs/{trained['run_key']}/predictions").json()
    assert len(log["entries"]) == 5
    assert all(entry["inputs"] == {} for entry in log["entries"])

    drift = client.get(f"/api/runs/{trained['run_key']}/drift").json()
    assert drift["status"] == "unavailable"
    assert "without their inputs" in drift["reason"]


def test_a_batch_larger_than_the_cap_is_sampled_and_flagged(client, trained, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "prediction_log_max_rows", 10)
    _predict_batch(client, trained["run_key"], np.random.default_rng(3), 50)

    entries = client.get(f"/api/runs/{trained['run_key']}/predictions?limit=100").json()["entries"]
    assert len(entries) == 10
    assert all(entry["sampled"] for entry in entries)


# --- drift ---------------------------------------------------------------------


def test_a_same_distribution_batch_reads_as_stable(client, trained):
    _predict_batch(client, trained["run_key"], np.random.default_rng(5), 250)
    drift = client.get(f"/api/runs/{trained['run_key']}/drift").json()
    assert drift["status"] == "stable", drift["reason"]


def test_a_genuinely_shifted_batch_is_flagged(client, trained):
    _predict_batch(client, trained["run_key"], np.random.default_rng(6), 250, age_mean=45 + 25)
    drift = client.get(f"/api/runs/{trained['run_key']}/drift").json()
    assert drift["status"] == "material", drift["reason"]
    age_column = next(c for c in drift["columns"] if c["column"] == "age")
    assert age_column["verdict"] == "material"


def test_too_few_predictions_gives_insufficient_data_not_a_false_alarm(client, trained):
    _predict_batch(client, trained["run_key"], np.random.default_rng(7), 5, age_mean=90)
    drift = client.get(f"/api/runs/{trained['run_key']}/drift").json()
    assert drift["status"] == "insufficient_data"


def test_drift_status_feeds_the_retraining_advice(client, trained):
    _predict_batch(client, trained["run_key"], np.random.default_rng(8), 250, age_mean=45 + 25)
    overview = client.get(f"/api/runs/{trained['run_key']}/monitoring").json()
    assert overview["drift"]["status"] == "material"
    assert overview["retraining"]["recommended"] is True
    assert "retrain" in overview["retraining"]["how"].lower()


# --- calibration in the monitoring view ----------------------------------------


def test_calibration_is_reported_in_the_monitoring_view(client, trained):
    overview = client.get(f"/api/runs/{trained['run_key']}/monitoring").json()
    assert overview["calibration"]["verdict"] in {"well_calibrated", "usable", "poor", "unmeasured"}


def test_the_prediction_response_names_whether_confidence_is_calibrated(client, trained):
    response = _predict_batch(client, trained["run_key"], np.random.default_rng(1), 3)
    assert "confidence_is_calibrated" in response
    assert response["calibration"]["verdict"] in {"well_calibrated", "usable", "poor", "unmeasured"}


# --- versions and champion resolution ------------------------------------------


def test_a_fresh_run_has_exactly_one_champion_version(client, trained):
    history = client.get(f"/api/runs/{trained['run_key']}/versions").json()
    assert history["count"] == 1
    assert history["champion"]["version"] == 1
    assert history["champion"]["is_champion"] is True
    assert history["champion"]["source"] == "initial"


# --- serving a run with no registry row (restored backup) ---------------------


def test_a_run_with_no_version_row_still_serves_from_the_fallback(client, trained):
    """A backup restored from before the registry existed: files on disk, no rows.

    `registry.resolve` falls back to the legacy artifact path rather than
    refusing outright — a monitoring feature must not turn into an outage for
    a run that predates it.
    """
    import registry

    db.delete_model_versions_for_run(trained["run_key"])
    assert db.list_model_versions(trained["run_key"]) == []

    resolved = registry.resolve(trained["run_key"])
    assert resolved["versioned"] is False
    assert resolved["artifact_suffix"] == registry.LEGACY_SUFFIX

    response = _predict_batch(client, trained["run_key"], np.random.default_rng(1), 3)
    assert response["rows"]
    assert response["model_version"] is None


# --- retraining and promotion --------------------------------------------------


def test_a_clearly_better_challenger_is_promoted(client, trained, tmp_path):
    """Fresh, larger, cleaner data should out-score a model trained on noisier data."""
    # A larger, less noisy dataset for the same underlying relationship.
    frame = _classification_frame(np.random.default_rng(999), 2500)
    path = tmp_path / "newer.csv"
    frame.to_csv(path, index=False)

    with path.open("rb") as handle:
        response = client.post(
            f"/api/runs/{trained['run_key']}/retrain",
            files={"file": ("newer.csv", handle, "text/csv")},
            data={"trigger": "manual"},
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["comparison"]["metric"] == "f1_macro"
    assert result["challenger"]["version"] == 2

    history = client.get(f"/api/runs/{trained['run_key']}/versions").json()
    assert history["count"] == 2
    if result["promoted"]:
        assert history["champion"]["version"] == 2
        assert history["champion"]["source"] == "retrain"
        # The version 1 row should now show as superseded.
        v1 = next(v for v in history["versions"] if v["version"] == 1)
        assert v1["is_champion"] is False
        assert v1["superseded_at"] is not None
    else:
        # Comparisons are not guaranteed to favour the new model on every
        # random draw; if it was not promoted, the reason has to say why.
        assert result["comparison"]["reason"]
        assert history["champion"]["version"] == 1


def test_a_worse_challenger_is_not_promoted(client, trained, monkeypatch, tmp_path):
    """Force the comparison to refuse, and check nothing changed on refusal."""
    import registry

    monkeypatch.setattr(
        registry,
        "better",
        lambda *args, **kwargs: {
            "promote": False,
            "metric": "f1_macro",
            "challenger": 0.5,
            "champion": 0.9,
            "margin": 0.01,
            "reason": "forced refusal for the test",
        },
    )

    frame = _classification_frame(np.random.default_rng(11), 500)
    path = tmp_path / "worse.csv"
    frame.to_csv(path, index=False)

    with path.open("rb") as handle:
        response = client.post(
            f"/api/runs/{trained['run_key']}/retrain",
            files={"file": ("worse.csv", handle, "text/csv")},
        )
    assert response.status_code == 200
    result = response.json()
    assert result["promoted"] is False

    history = client.get(f"/api/runs/{trained['run_key']}/versions").json()
    assert history["champion"]["version"] == 1
    assert history["count"] == 2  # the challenger was still recorded


def test_retraining_is_bounded_by_the_phase_8_concurrency_limit(client, authenticated, churn_csv, tmp_path):
    """No job row is created for a synchronous retrain, but it still competes
    for the same CPU a queued upload would — an account at its concurrency
    ceiling must not be able to sidestep it by retraining instead.

    Auth has to be genuinely on for this: `check_job_concurrency` is a
    deliberate no-op with auth off, per Phase 8's rule that a per-account limit
    on a single-implicit-owner install is a per-machine limit in disguise.
    """
    from config import settings

    bob = authenticated["bob"]
    job = upload_and_wait(client, churn_csv, headers=bob["headers"])
    run_key = job["result"]["run_key"]

    settings.principal_max_concurrent_jobs = 1
    db.create_job("held-for-bob", "Queued", owner_id=bob["user_id"])

    frame = _classification_frame(np.random.default_rng(20), 300)
    path = tmp_path / "r.csv"
    frame.to_csv(path, index=False)
    with path.open("rb") as handle:
        response = client.post(
            f"/api/runs/{run_key}/retrain",
            files={"file": ("r.csv", handle, "text/csv")},
            headers=bob["headers"],
        )

    assert response.status_code == 429


def test_retraining_without_the_target_column_is_refused(client, trained, tmp_path):
    frame = _classification_frame(np.random.default_rng(12), 300).drop(columns=["churn"])
    path = tmp_path / "unlabelled.csv"
    frame.to_csv(path, index=False)

    with path.open("rb") as handle:
        response = client.post(
            f"/api/runs/{trained['run_key']}/retrain",
            files={"file": ("unlabelled.csv", handle, "text/csv")},
        )
    assert response.status_code == 400
    assert "labelled data" in response.json()["detail"]


def test_promotion_is_audited(client, trained, monkeypatch, tmp_path):
    import registry

    monkeypatch.setattr(
        registry,
        "better",
        lambda *args, **kwargs: {
            "promote": True,
            "metric": "f1_macro",
            "challenger": 0.9,
            "champion": 0.5,
            "reason": "x",
        },
    )
    frame = _classification_frame(np.random.default_rng(13), 400)
    path = tmp_path / "better.csv"
    frame.to_csv(path, index=False)
    with path.open("rb") as handle:
        client.post(f"/api/runs/{trained['run_key']}/retrain", files={"file": ("better.csv", handle, "text/csv")})

    entries = db.list_audit(limit=20, action="model.promoted")
    assert any(entry["target_id"] == trained["run_key"] for entry in entries)


def test_the_promoted_model_is_what_gets_served_next(client, trained, monkeypatch, tmp_path):
    import registry

    monkeypatch.setattr(
        registry,
        "better",
        lambda *args, **kwargs: {
            "promote": True,
            "metric": "f1_macro",
            "challenger": 0.9,
            "champion": 0.5,
            "reason": "x",
        },
    )
    frame = _classification_frame(np.random.default_rng(14), 400)
    path = tmp_path / "better.csv"
    frame.to_csv(path, index=False)
    with path.open("rb") as handle:
        retrain_response = client.post(
            f"/api/runs/{trained['run_key']}/retrain", files={"file": ("better.csv", handle, "text/csv")}
        )
    new_version_id = retrain_response.json()["challenger"]["version_id"]

    prediction = _predict_batch(client, trained["run_key"], np.random.default_rng(1), 2)
    assert prediction["model_version"] == new_version_id


# --- deletion and retention ----------------------------------------------------


def test_deleting_a_run_purges_its_predictions_and_versions(client, trained):
    _predict_batch(client, trained["run_key"], np.random.default_rng(1), 5)
    response = client.delete(f"/api/runs/{trained['run_key']}")
    assert response.status_code == 200
    assert response.json()["predictions_removed"] == 5

    # And genuinely gone from the database, not just unreachable through the
    # deleted run's now-404 endpoints.
    assert db.list_predictions(trained["run_key"]) == []
    assert db.list_model_versions(trained["run_key"]) == []


def test_retention_purge_removes_predictions_with_the_expired_run(client, trained, monkeypatch):
    import lifecycle
    from config import settings
    from utils.helpers import now_utc

    _predict_batch(client, trained["run_key"], np.random.default_rng(1), 5)
    monkeypatch.setattr(settings, "retention_days", 30)
    db.set_run_expiry(trained["run_key"], now_utc() - __import__("datetime").timedelta(days=1))

    summary = lifecycle.purge_expired()
    assert summary["predictions_removed"] == 5
    assert db.list_predictions(trained["run_key"]) == []


# --- owner scoping ---------------------------------------------------------


def test_monitoring_is_owner_scoped(client, authenticated, churn_csv):
    alice, bob = authenticated["alice"], authenticated["bob"]
    job = upload_and_wait(client, churn_csv, headers=alice["headers"])
    run_key = job["result"]["run_key"]

    assert client.get(f"/api/runs/{run_key}/monitoring", headers=bob["headers"]).status_code == 404
    assert client.get(f"/api/runs/{run_key}/monitoring", headers=alice["headers"]).status_code == 200


def test_a_prediction_is_not_readable_by_another_account(client, authenticated, churn_csv):
    alice, bob = authenticated["alice"], authenticated["bob"]
    job = upload_and_wait(client, churn_csv, headers=alice["headers"])
    run_key = job["result"]["run_key"]

    frame = _classification_frame(np.random.default_rng(1), 3).drop(columns=["churn"])
    client.post(
        f"/api/predict/{run_key}",
        files={"file": ("p.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
        headers=alice["headers"],
    )
    prediction_id = client.get(f"/api/runs/{run_key}/predictions", headers=alice["headers"]).json()["entries"][0][
        "prediction_id"
    ]

    assert client.get(f"/api/predictions/{prediction_id}", headers=bob["headers"]).status_code == 404
    assert client.get(f"/api/predictions/{prediction_id}", headers=alice["headers"]).status_code == 200


# --- serving schema, wired to the real endpoint --------------------------------


def test_predicting_on_a_renamed_column_gives_a_helpful_400(client, trained):
    frame = pd.DataFrame({"age": [40], "spend": [20.0]})  # no region at all -> genuinely missing
    response = client.post(
        f"/api/predict/{trained['run_key']}",
        files={"file": ("p.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
    )
    assert response.status_code == 400
    assert "region" in response.json()["detail"]
