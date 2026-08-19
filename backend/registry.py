"""Which model is being served, and what it had to beat to get there.

Until Phase 9 a run had exactly one model, and the run key was the identity of
both the dataset and the artifact fitted from it. That identity is what made the
content-addressed cache correct, and it holds right up to the moment a model is
retrained on newer data — which is the whole point of retraining. The dataset has
changed, so a hash of the old one cannot name the new model.

So a run owns a chain of versions and exactly one of them is champion. The
champion is what `/api/predict` loads and what every logged prediction names, so
"which model said this" stays answerable after the next promotion.

## Promotion is an argument, not a timestamp

A challenger replaces the champion only when it wins on the metric the run was
selected on, by a margin, **measured on the same rows**. Each of those three is
load-bearing:

* **The same metric the run already uses.** Choosing a new metric at promotion
  time is choosing the metric that makes the new model look good, which is not a
  comparison.
* **A margin** (`CHALLENGER_PROMOTION_MARGIN`). Refit the same model on the same
  data with a different seed and the score moves; promoting on any improvement
  at all means promoting that movement, repeatedly, and calling it progress.
* **The same rows.** A challenger evaluated on its own holdout and a champion
  evaluated on its own holdout have not been compared — they have been scored on
  two different exams. Both are run against the challenger's holdout, which is
  the only set of labelled rows that is new to both of them: the champion has
  never seen it, and the challenger was not fitted on it.

That last point has a limit worth stating plainly. The challenger's holdout is
drawn from the *new* data, so a champion trained on an older distribution is
being asked about the world as it is now. That is the correct question when the
reason for retraining is drift — but it does mean the comparison favours the
challenger whenever the data has moved, which is precisely when retraining gets
triggered. The refusal path exists for that reason: when the challenger does not
clear the margin even with that advantage, it is decisively not better.

## What is not here

No automatic promotion on a schedule, and no retraining triggered by drift
without a person. `monitoring.retraining_advice` will say a model looks due, and
a drift signal is a good reason to *look*; it is a bad reason to fit a model on
whatever anomaly produced the signal and then serve it. The trigger is an
endpoint, and a person calls it.
"""

import logging
import uuid
from typing import Any

import db
import utils.helpers
from config import settings
from utils.security import artifact_path

logger = logging.getLogger(__name__)

#: The artifact suffix version 1 uses. Kept as the pre-Phase-9 name so every run
#: trained before the registry existed is describable in it without a file move.
LEGACY_SUFFIX = "_model.pkl"

#: Metrics where a smaller number is a better model. The same table training
#: uses; duplicated deliberately rather than imported, because a promotion
#: comparing in the wrong direction is silent and catastrophic and this file
#: should be readable on its own.
LOWER_IS_BETTER = {"rmse", "mae", "mape", "brier"}


def suffix_for(version: int) -> str:
    """Where version N's bundle lives, relative to the run key."""
    return LEGACY_SUFFIX if version == 1 else f"_v{version}_model.pkl"


def model_path_for(run_key: str, suffix: str):
    """The validated on-disk path of one version's bundle.

    `utils.helpers.MODEL_DIR` is read here rather than imported at the top,
    which is not stylistic. Several modules captured that name at import time
    and the test suite has to rebind it in each of them by hand — there is a
    list of them in `tests/conftest.py` with a comment explaining the problem.
    Reading the attribute at call time means this module never joins that list,
    and redirecting storage keeps working without anyone remembering to.
    """
    return artifact_path(utils.helpers.MODEL_DIR, run_key, suffix)


def register_initial(run_key: str, owner_id: str, metadata: dict) -> dict:
    """Record the model a first training produced, as version 1 and champion.

    Idempotent: a cache hit re-runs the registration path with the same run key,
    and a run that already has versions keeps the ones it has.
    """
    existing = db.list_model_versions(run_key)
    if existing:
        return next((version for version in existing if version["is_champion"]), existing[-1])

    metric_name = metadata.get("selection_metric")
    record = db.add_model_version(
        version_id=_version_id(),
        run_key=run_key,
        owner_id=owner_id,
        version=1,
        artifact_suffix=LEGACY_SUFFIX,
        selected_model=metadata.get("selected_model"),
        metric_name=metric_name,
        metric_value=_holdout_score(metadata, metric_name),
        rows_trained=int(metadata.get("row_count") or 0),
        is_champion=True,
        promoted_at=None,
        source="initial",
        notes={"reason": "First model trained for this run; there was nothing to compare it with."},
    )
    return db.promote_model_version(record["version_id"]) or record


def _version_id() -> str:
    return uuid.uuid4().hex[:16]


def _holdout_score(metadata: dict, metric_name: str | None) -> float | None:
    """This model's score on the metric the run was selected on.

    Read from `all_model_scores` for the selected model, which is the holdout
    evaluation — the one number in the metadata that no fold, search or
    selection step has seen.
    """
    if not metric_name:
        return None
    scores = (metadata.get("all_model_scores") or {}).get(metadata.get("selected_model")) or {}
    value = scores.get(metric_name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def better(metric_name: str, challenger: float | None, champion: float | None, margin: float | None = None) -> dict:
    """Is the challenger enough better to be worth swapping in?

    Returns the whole comparison rather than a boolean, because the refusal has
    to be as inspectable as the promotion — "we kept the old one" with no
    numbers is the same unfalsifiable claim as "the new one is better".
    """
    margin = settings.challenger_promotion_margin if margin is None else margin

    if challenger is None or champion is None:
        return {
            "promote": False,
            "metric": metric_name,
            "challenger": challenger,
            "champion": champion,
            "margin": margin,
            "reason": (
                "One of the two models has no score on the run's selection metric, so there is no "
                "comparison to make. The champion was kept."
            ),
        }

    lower_is_better = metric_name in LOWER_IS_BETTER
    # Relative to the incumbent's own magnitude, so one margin works for an
    # f1 of 0.9 and an RMSE of 40,000. `abs` because a score can be negative
    # (a negated error), and a margin must not flip sign with it.
    required = abs(champion) * margin
    improvement = (champion - challenger) if lower_is_better else (challenger - champion)
    promote = improvement > required

    direction = "lower" if lower_is_better else "higher"
    if promote:
        reason = (
            f"The challenger scored {challenger:.4g} against the champion's {champion:.4g} on {metric_name} "
            f"({direction} is better), an improvement of {improvement:.4g} — past the {margin:.1%} margin of "
            f"{required:.4g}. Promoted on that."
        )
    else:
        reason = (
            f"The challenger scored {challenger:.4g} against the champion's {champion:.4g} on {metric_name} "
            f"({direction} is better). That is {improvement:.4g}, short of the {margin:.1%} margin of "
            f"{required:.4g} required to promote, so the champion was kept. A difference this size is as "
            "likely to be sampling noise as improvement."
        )

    return {
        "promote": bool(promote),
        "metric": metric_name,
        "challenger": challenger,
        "champion": champion,
        "improvement": float(improvement),
        "required": float(required),
        "margin": margin,
        "lower_is_better": lower_is_better,
        "reason": reason,
    }


def register_challenger(
    run_key: str,
    owner_id: str,
    version: int,
    metadata: dict,
    comparison: dict,
    metric_value: float | None,
) -> dict:
    """Record a retrained model, and promote it if the comparison says so."""
    record = db.add_model_version(
        version_id=_version_id(),
        run_key=run_key,
        owner_id=owner_id,
        version=version,
        artifact_suffix=suffix_for(version),
        selected_model=metadata.get("selected_model"),
        metric_name=comparison.get("metric"),
        metric_value=metric_value,
        rows_trained=int(metadata.get("row_count") or 0),
        is_champion=False,
        source="retrain",
        notes={"comparison": comparison},
    )

    if not comparison.get("promote"):
        logger.info(
            "Challenger not promoted",
            extra={"run_key": run_key, "version": version, "reason": comparison.get("reason")},
        )
        return record

    promoted = db.promote_model_version(record["version_id"], notes={"comparison": comparison})
    logger.info("Challenger promoted", extra={"run_key": run_key, "version": version})
    return promoted or record


def resolve(run_key: str, owner_id: str | None = None) -> dict:
    """Which artifact to serve for this run, and what is known about it.

    Falls back to the pre-Phase-9 path when there is no registry row at all.
    That is not a hypothetical: a run whose artifacts were restored from a
    backup taken before the registry existed has files on disk and no version
    rows, and refusing to serve it would turn a monitoring feature into an
    outage. The fallback says so in `versioned`, so a caller can tell an
    unregistered model from a champion.
    """
    champion = db.champion_version(run_key, owner_id=owner_id)
    if champion is None:
        return {
            "versioned": False,
            "version_id": None,
            "version": None,
            "artifact_suffix": LEGACY_SUFFIX,
            "note": (
                "This run has no model version registered, so the original artifact is being served. "
                "Retrain it to bring it under version tracking."
            ),
        }
    return {
        "versioned": True,
        "version_id": champion["version_id"],
        "version": champion["version"],
        "artifact_suffix": champion["artifact_suffix"],
        "selected_model": champion.get("selected_model"),
        "promoted_at": champion.get("promoted_at"),
        "note": None,
    }


def history(run_key: str, owner_id: str | None = None) -> dict[str, Any]:
    """Every version of this model and how each one got, or did not get, promoted."""
    versions = db.list_model_versions(run_key, owner_id=owner_id)
    return {
        "run_key": run_key,
        "versions": versions,
        "champion": next((version for version in versions if version["is_champion"]), None),
        "count": len(versions),
    }
