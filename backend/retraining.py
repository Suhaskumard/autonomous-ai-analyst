"""Fitting a replacement model, and making it earn the job.

The trigger for this is usually drift: the inputs the model is being sent no
longer look like the ones it learned from, so the model that was defensible in
August may not be in November. Retraining is the obvious response and, on its
own, a bad one — a model fitted on newer data is *newer*, which is not the same
as better, and "we retrained it" is not evidence of anything.

So retraining here produces a **challenger**, and the challenger has to beat the
**champion** before it is served. Both are scored on the same rows, with the same
metric the run was originally selected on, and the comparison is stored with the
version so the decision can be read back later. See `registry.py` for why the
margin exists and what the comparison's known bias is.

## Same configuration, new data

The challenger is fitted with the configuration the run already has — the same
target column, the same mode, the same manual model if one was chosen. Changing
the configuration *and* the data at once produces a model that is different for
two reasons and comparable for none. Retraining answers exactly one question:
does this configuration do better on data that includes what has changed?

## Scoring in label space

Both models are asked for predictions on the challenger's holdout, and both sets
are compared against the true labels as *strings* rather than as encoded
integers. Two `LabelEncoder`s fitted on two datasets can assign different
integers to the same class — a class absent from the newer data is enough to
shift every code after it — and comparing through them would silently score both
models against the wrong answer. Decoding first makes the comparison
independent of how either model happens to number its classes.
"""

import json
import logging
from typing import Any

import pandas as pd

import db
import registry
from config import settings
from exceptions import PipelineError
from ml.evaluate import evaluate
from ml.preprocess import prepare_dataset
from ml.train import PIPELINE_VERSION, train_models
from observability import metrics
from utils.artifacts import dump_bundle, load_bundle
from utils.helpers import METADATA_DIR, now_utc_iso
from utils.security import artifact_path
from utils.storage import publish_run

logger = logging.getLogger(__name__)


def retrain(
    run_key: str,
    owner_id: str,
    frame: pd.DataFrame,
    metadata: dict,
    trigger: str = "manual",
) -> dict:
    """Fit a challenger on `frame` and promote it only if it wins.

    Returns the full decision — both scores, the margin, and what happened —
    whether or not anything was promoted. A refusal that says only "not
    promoted" is as unfalsifiable as a promotion that says only "better".
    """
    champion = db.champion_version(run_key, owner_id=owner_id)
    if champion is None:
        raise PipelineError(
            "This run has no registered champion to compare against. Re-upload the dataset to train it "
            "under version tracking first."
        )

    target = metadata.get("target")
    if target and target not in frame.columns:
        raise PipelineError(
            f"Retraining needs labelled data: the target column {target!r} is not in the file supplied. "
            "A model cannot be scored against outcomes that are not there."
        )

    prepared = prepare_dataset(frame, target)
    trained = train_models(
        prepared["X"],
        prepared["y"],
        prepared["preprocessor"],
        mode=metadata.get("mode", "auto"),
        manual_model=_manual_model(metadata),
        tuning_budget_seconds=float(metadata.get("tuning_budget_seconds") or 0.0),
        use_smote=bool(metadata.get("smote_used")),
    )

    metric = metadata.get("selection_metric") or trained["selection_metric"]
    challenger_pipeline = trained["trained_models"][trained["selected_model"]]

    # The challenger's holdout: the only labelled rows neither model has been
    # fitted on. See the module docstring for why both are scored here rather
    # than each on its own.
    X_test, y_test = trained["X_test"], trained["y_test"]
    truth = _decode(y_test, trained.get("label_encoder"))

    challenger_score = _score(
        trained["problem_type"], challenger_pipeline, trained.get("label_encoder"), X_test, truth, metric
    )
    champion_score = _score_champion(run_key, champion, trained["problem_type"], X_test, truth, metric)

    comparison = registry.better(metric, challenger_score, champion_score)
    comparison["rows_compared"] = int(len(y_test))
    comparison["trigger"] = trigger
    comparison["evaluated_on"] = "the challenger's holdout — rows neither model was fitted on"

    version_number = db.next_model_version(run_key)
    _write_bundle(run_key, version_number, trained, prepared, challenger_pipeline)

    challenger_metadata = {
        "selected_model": trained["selected_model"],
        "row_count": int(len(frame)),
        "selection_metric": metric,
    }
    record = registry.register_challenger(
        run_key=run_key,
        owner_id=owner_id,
        version=version_number,
        metadata=challenger_metadata,
        comparison=comparison,
        metric_value=challenger_score,
    )

    if comparison["promote"]:
        # Only once it is the champion: a bundle published before it is
        # promoted is a file other replicas may load and never serve, and one
        # published after promotion is briefly missing on a replica that has
        # not synced — which `predict._load_bundle` answers with a 503 rather
        # than by quietly serving the old model under the new version's name.
        #
        # `record["artifact_suffix"]` has to be passed explicitly. Version 1
        # is always `_model.pkl`, which `ARTIFACT_SUFFIXES` already knows about
        # — a retrained version is not, and publishing without naming it here
        # mirrors only the stale version-1 bundle. Every other replica's
        # `ensure_local` would then find nothing new to fetch, and the 503
        # above would never clear: not "briefly missing", but missing forever.
        published = publish_run(run_key, [record["artifact_suffix"]])
        logger.info("Challenger published", extra={"run_key": run_key, "files": published})

    _record_history(run_key, comparison, version_number, trigger)
    metrics.observe_retraining(comparison["promote"])

    return {
        "run_key": run_key,
        "trigger": trigger,
        "promoted": bool(comparison["promote"]),
        "comparison": comparison,
        "challenger": record,
        "champion_before": champion,
        "rows_supplied": int(len(frame)),
        "selected_model": trained["selected_model"],
    }


def _manual_model(metadata: dict) -> str | None:
    """The model the original run was pinned to, if it was pinned to one."""
    return metadata.get("selected_model") if metadata.get("mode") == "manual" else None


def _decode(encoded, label_encoder):
    """Encoded targets back to their original labels, for scoring."""
    if label_encoder is None:
        return encoded
    return label_encoder.inverse_transform(encoded)


def _score(problem_type: str, pipeline, label_encoder, X_test, truth, metric: str) -> float | None:
    """One model's score on the shared holdout, in label space."""
    try:
        predictions = _decode(pipeline.predict(X_test), label_encoder)
        scores = evaluate(problem_type, truth, predictions)
        value = scores.get(metric)
        return None if value is None else float(value)
    except Exception:
        logger.warning("Could not score a model on the shared holdout", exc_info=True)
        return None


def _score_champion(run_key: str, champion: dict, problem_type: str, X_test, truth, metric: str) -> float | None:
    """The incumbent, on the challenger's holdout.

    Loaded through `load_bundle`, which verifies the signature before joblib
    sees the bytes — the champion is an artifact like any other and gets the
    same treatment here as it does at serving time.
    """
    try:
        path = registry.model_path_for(run_key, champion["artifact_suffix"])
        bundle = load_bundle(path)
    except Exception:
        logger.warning("Champion bundle could not be loaded for comparison", exc_info=True, extra={"run": run_key})
        return None

    pipeline = next(iter(bundle.get("pipelines", {}).values()), None)
    if pipeline is None:
        return None

    # A column the champion was trained on may be absent from the new file, and
    # a column it never saw may be present. The pipeline will refuse either
    # way; catching it here turns a stack trace into "the champion could not be
    # scored", which is what `registry.better` already knows how to report.
    expected = bundle.get("feature_columns")
    if expected:
        missing = [column for column in expected if column not in X_test.columns]
        if missing:
            logger.info("Champion cannot score the new holdout", extra={"missing": missing})
            return None
        X_test = X_test[expected]

    return _score(problem_type, pipeline, bundle.get("label_encoder"), X_test, truth, metric)


def _write_bundle(run_key: str, version: int, trained: dict, prepared: dict, pipeline) -> None:
    """Save the challenger under its own suffix, signed like every bundle."""
    bundle = {
        "kind": "single",
        "pipelines": {trained["selected_model"]: pipeline},
        "problem_type": trained["problem_type"],
        "label_encoder": trained.get("label_encoder"),
        "feature_columns": prepared["feature_columns"],
        "pipeline_version": PIPELINE_VERSION,
    }
    dump_bundle(bundle, registry.model_path_for(run_key, registry.suffix_for(version)))


def _record_history(run_key: str, comparison: dict, version: int, trigger: str) -> None:
    """Append this decision to the run's metadata, promoted or not.

    Kept beside the model card rather than only in the database, because the
    metadata file is what travels with the artifacts into a backup — and "why
    is version 3 being served and not version 4" is a question that outlives
    any one database.
    """
    path = artifact_path(METADATA_DIR, run_key, ".json")
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        history: list[dict[str, Any]] = list(metadata.get("retraining_history") or [])
        history.append(
            {
                "at": now_utc_iso(),
                "version": version,
                "trigger": trigger,
                "promoted": bool(comparison["promote"]),
                "metric": comparison.get("metric"),
                "challenger": comparison.get("challenger"),
                "champion": comparison.get("champion"),
                "reason": comparison.get("reason"),
            }
        )
        metadata["retraining_history"] = history
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)
    except Exception:
        # A history that failed to write must not undo a promotion that
        # succeeded: the registry row is the authority, and this is the copy.
        logger.warning("Could not append retraining history", exc_info=True, extra={"run_key": run_key})


def scheduled_candidates(owner_id: str | None = None, min_age_days: int = 30) -> list[dict]:
    """Runs old enough that a scheduled retrain would be reasonable.

    The "on a schedule" trigger, without a scheduler. An in-process timer is one
    more thing that dies with the process — the same reasoning that keeps the
    retention purge on cron in `lifecycle.py` — so this reports what is due and
    something outside calls the endpoint. `docs/monitoring.md` has the cron line.
    """
    from datetime import timedelta

    from utils.helpers import now_utc

    cutoff = now_utc() - timedelta(days=min_age_days)
    due: list[dict] = []
    for run in db.list_runs(limit=1000, owner_id=owner_id):
        champion = db.champion_version(run["run_key"], owner_id=owner_id)
        if champion is None:
            continue
        promoted = champion.get("promoted_at") or champion.get("created_at")
        if promoted and str(promoted) < cutoff.isoformat():
            due.append(
                {
                    "run_key": run["run_key"],
                    "filename": run.get("filename"),
                    "champion_version": champion["version"],
                    "promoted_at": promoted,
                }
            )
    return due


def margin() -> float:
    return settings.challenger_promotion_margin
