"""Are the probabilities this model reports actually probabilities?

The UI shows a confidence next to every classification. That number is a claim:
"of the cases where I said 90%, about 90% were right." Nothing in this project
has ever checked it, and for most classifiers it is false by construction — a
random forest's vote share is not a probability, an SVM's Platt output is a
rough one, and a boosted tree is usually overconfident near the extremes. A
model can be excellent at ranking and badly wrong about its own certainty; those
are different properties and only the first was ever measured here.

## What is measured

**Expected calibration error.** Predictions are grouped by confidence into bins,
and each bin's mean confidence is compared with its actual accuracy. ECE is the
average gap, weighted by how many predictions fall in each bin. 0 is perfect; a
model claiming 90% and being right 70% of the time in that bin contributes 0.2
in proportion to the bin's size.

**Brier score**, the mean squared error of the probability vector. It moves with
both calibration and discrimination, which is why it is reported alongside ECE
rather than instead of it: ECE alone is happy with a model that always says "60%"
and is right 60% of the time, which is perfectly calibrated and perfectly
useless.

**The reliability curve**, kept so the UI can draw the diagonal a well-calibrated
model should lie on.

## What is done about it

`CalibratedClassifierCV` is fitted on the development split — never the holdout —
using internal cross-validation, so the calibrator never sees the data it is
scored against. Both the original and the calibrated model are then scored on the
untouched holdout, and calibration is adopted **only if it improves ECE by a
clear margin**.

The cost of that design, stated rather than buried: the adopt/reject decision is
made on the holdout, so the holdout is no longer entirely untouched for this one
binary choice. The alternative is a third split, which on a small dataset costs
more than the bias it removes. Both numbers are recorded in the metadata, so the
decision is auditable rather than asserted — the same standard the champion and
challenger comparison is held to.

Calibration is skipped when it cannot help or cannot be trusted: regressors have
no probabilities, hard-voting ensembles expose none, and a holdout too small to
fill the bins gives an ECE that is mostly sampling noise.
"""

import logging

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

#: Confidence bins for ECE and the reliability curve.
N_BINS = 10

#: Below this many holdout rows, ECE is noise: ten bins over forty rows is four
#: predictions a bin, and a single mistake moves the number by 0.25.
MIN_HOLDOUT_ROWS = 50

#: How much better calibrated the calibrated model must be before it replaces
#: the original. A margin rather than "any improvement" because the comparison
#: is made on the holdout, so a hair's-width win is as likely to be noise as
#: signal — and swapping in a second layer of fitting for noise is a bad trade.
MIN_ECE_IMPROVEMENT = 0.02

#: Above this the confidences should not be presented as probabilities without
#: a caveat. Chosen as "the claim is off by more than ten points on average",
#: which is the point at which a displayed percentage misleads a reader.
POOR_ECE = 0.10

WELL_CALIBRATED = "well_calibrated"
USABLE = "usable"
POOR = "poor"
UNMEASURED = "unmeasured"


def expected_calibration_error(y_true, y_proba, n_bins: int = N_BINS) -> tuple[float, list[dict]]:
    """ECE and the reliability curve, from predicted class probabilities.

    Uses the top-class confidence — the number the UI actually shows — rather
    than the positive-class probability, so this measures the claim being made
    to the user and works unchanged for more than two classes.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:  # binary given as P(positive)
        y_proba = np.column_stack([1 - y_proba, y_proba])

    predicted = y_proba.argmax(axis=1)
    confidence = y_proba.max(axis=1)
    correct = (predicted == y_true).astype(float)

    # Right-closed bins so a confidence of exactly 1.0 lands in the last one
    # rather than falling off the end.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    total = len(confidence)
    ece = 0.0
    curve: list[dict] = []
    for index in range(n_bins):
        mask = indices == index
        count = int(mask.sum())
        if count == 0:
            curve.append(
                {
                    "bin_lower": float(edges[index]),
                    "bin_upper": float(edges[index + 1]),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += (count / total) * abs(mean_confidence - accuracy)
        curve.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return float(ece), curve


def brier_score(y_true, y_proba, n_classes: int) -> float:
    """Multiclass Brier score: mean squared error of the whole probability vector."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])

    onehot = np.zeros_like(y_proba, dtype=float)
    rows = np.arange(len(y_true))
    valid = (y_true >= 0) & (y_true < y_proba.shape[1])
    onehot[rows[valid], y_true[valid]] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def assess(y_true, y_proba, n_classes: int) -> dict:
    """Measure how honest a set of predicted probabilities is."""
    ece, curve = expected_calibration_error(y_true, y_proba)
    return {
        "ece": ece,
        "brier": brier_score(y_true, y_proba, n_classes),
        "curve": curve,
        "verdict": _verdict(ece),
        "bins": N_BINS,
        "rows": int(len(np.asarray(y_true))),
    }


def _verdict(ece: float) -> str:
    if ece < 0.05:
        return WELL_CALIBRATED
    if ece < POOR_ECE:
        return USABLE
    return POOR


def calibrate(pipeline, problem_type: str, X_dev, y_dev, X_test, y_test, n_classes: int, enabled: bool = True) -> dict:
    """Fit a calibrated variant and keep it only if it is measurably better.

    Returns a report carrying `pipeline` — the model to serve, which is the
    original one unless calibration earned the swap — plus both sets of numbers
    so the decision can be checked rather than taken on trust.
    """
    report: dict = {
        "applied": False,
        "method": None,
        "pipeline": pipeline,
        "before": None,
        "after": None,
        "verdict": UNMEASURED,
        "reason": "",
    }

    if problem_type != "classification":
        report["reason"] = "Calibration applies to predicted probabilities; a regressor has none."
        return report
    if not hasattr(pipeline, "predict_proba"):
        report["reason"] = (
            "This model does not expose predicted probabilities, so no confidence is shown and there is "
            "nothing to calibrate."
        )
        return report

    try:
        base_proba = np.asarray(pipeline.predict_proba(X_test))
    except Exception as exc:
        report["reason"] = f"Predicted probabilities were unavailable on the holdout ({exc})."
        return report

    before = assess(y_test, base_proba, n_classes)
    report["before"] = before
    report["verdict"] = before["verdict"]

    if len(y_test) < MIN_HOLDOUT_ROWS:
        report["reason"] = (
            f"The holdout has {len(y_test)} rows; at least {MIN_HOLDOUT_ROWS} are needed before a "
            "calibration error is a measurement rather than sampling noise. Reported, not acted on."
        )
        report["verdict"] = UNMEASURED
        return report

    if not enabled:
        report["reason"] = "Measured only: CALIBRATE_PROBABILITIES is off."
        return report

    if before["verdict"] == WELL_CALIBRATED:
        report["reason"] = (
            f"Already well calibrated (ECE {before['ece']:.3f}); a second layer of fitting would add "
            "risk and complexity for nothing."
        )
        return report

    # Isotonic is more flexible and needs more data; sigmoid (Platt) is a
    # two-parameter fit that holds up on small samples. Trying both and keeping
    # the better one costs two fits and removes a guess.
    candidates = ["sigmoid"] if len(y_dev) < 1000 else ["isotonic", "sigmoid"]
    best: tuple[float, str, object, dict] | None = None

    for method in candidates:
        try:
            # cv on the development split: the calibrator's own folds mean it
            # never fits and calibrates on the same rows. `clone` because a
            # fitted pipeline is refit here from scratch.
            calibrated = CalibratedClassifierCV(clone(pipeline), method=method, cv=3)
            calibrated.fit(X_dev, y_dev)
            measured = assess(y_test, np.asarray(calibrated.predict_proba(X_test)), n_classes)
        except Exception as exc:
            logger.info("Calibration with %s failed: %s", method, exc)
            continue
        if best is None or measured["ece"] < best[0]:
            best = (measured["ece"], method, calibrated, measured)

    if best is None:
        report["reason"] = "Every calibration method failed to fit; the original probabilities are unchanged."
        return report

    improvement = before["ece"] - best[0]
    report["after"] = best[3]
    report["improvement"] = float(improvement)

    if improvement < MIN_ECE_IMPROVEMENT:
        report["reason"] = (
            f"Calibration ({best[1]}) moved ECE from {before['ece']:.3f} to {best[0]:.3f}, less than the "
            f"{MIN_ECE_IMPROVEMENT} margin required. The original model was kept: a change this small is "
            "as likely to be noise as improvement."
        )
        return report

    report.update(
        applied=True,
        method=best[1],
        pipeline=best[2],
        verdict=best[3]["verdict"],
        reason=(
            f"Calibrated with {best[1]}: expected calibration error {before['ece']:.3f} → {best[0]:.3f} on "
            "the holdout. The displayed confidences are the calibrated ones."
        ),
    )
    return report


def to_metadata(report: dict) -> dict:
    """The part of a calibration report that belongs in the run's metadata.

    Without the fitted estimator, which lives in the model bundle, and with both
    the before and after numbers kept — a promotion nobody can check is not
    evidence.
    """
    return {
        "applied": bool(report.get("applied")),
        "method": report.get("method"),
        "verdict": report.get("verdict", UNMEASURED),
        "reason": report.get("reason", ""),
        "improvement": report.get("improvement"),
        "before": report.get("before"),
        "after": report.get("after"),
        "confidence_is_calibrated": bool(report.get("applied")),
        "note": (
            "A confidence shown next to a prediction is a claim about how often that prediction is right. "
            "Expected calibration error is the average gap between the claim and the outcome, measured on "
            "data the model was not fitted on."
        ),
    }
