"""A verdict on whether a run is worth believing.

The committed 1.7%-accuracy run was presented in the UI as the "Optimized
Model" with no caveat. A score is only meaningful next to the trivial baseline
it should beat: always-predict-the-majority-class for classification,
always-predict-the-mean for regression.

The comparison is now made on macro F1 rather than accuracy. On a 95/5 target
the majority-class predictor scores 95% accuracy — which looks excellent and
beats its own accuracy baseline by exactly nothing — but 0.49 macro F1 against
a 0.49 baseline, which is visibly no better than guessing. Accuracy is still
reported; it just no longer decides the verdict.
"""

import numpy as np
from sklearn.metrics import f1_score

# A model this far below its own baseline is worse than a constant predictor.
WEAK_MARGIN = 0.02
# Cross-validated spread wider than this means the score depends on the split.
UNSTABLE_STD = 0.10

METRIC_LABEL = {"f1_macro": "macro F1", "rmse": "RMSE", "accuracy": "accuracy"}


def _classification_baseline(y_test: np.ndarray) -> float:
    """Macro F1 of always predicting the most common class in the test split."""
    y_test = np.asarray(y_test)
    if len(y_test) == 0:
        return 0.0
    labels, counts = np.unique(y_test, return_counts=True)
    majority = labels[int(np.argmax(counts))]
    constant = np.full_like(y_test, majority)
    return float(f1_score(y_test, constant, average="macro", zero_division=0))


def _regression_baseline(y_test: np.ndarray) -> float:
    """RMSE of always predicting the mean of the test split."""
    if len(y_test) == 0:
        return 0.0
    y_test = np.asarray(y_test, dtype=float)
    return float(np.sqrt(np.mean((y_test - y_test.mean()) ** 2)))


def assess_run(
    problem_type: str,
    selected_scores: dict,
    y_test,
    failed_models: dict,
    n_classes: int,
    cv_summary: dict | None = None,
    imbalance: dict | None = None,
    holdout_rows: int | None = None,
) -> dict:
    """Return the verdict for the chosen model.

    verdict is one of: "strong", "fair", "unreliable".
    """
    y_test = np.asarray(y_test)
    cv_summary = cv_summary or {}
    imbalance = imbalance or {}
    # Two kinds of statement, kept apart because only one of them downgrades a
    # verdict: a caveat is a reason to trust the run less, a note is context.
    caveats: list[str] = []
    notes: list[str] = []

    if problem_type == "classification":
        metric_name = "f1_macro"
        score = float(selected_scores.get("f1_macro", 0.0))
        baseline = _classification_baseline(y_test)
        beats_baseline = score > baseline + WEAK_MARGIN
        lift = score - baseline
        accuracy = float(selected_scores.get("accuracy", 0.0))

        if not beats_baseline:
            caveats.append(
                f"Macro F1 of {score:.3f} does not beat the {baseline:.3f} you get from "
                "always predicting the most common class."
            )
            if accuracy > 0.7:
                # The exact trap this metric change exists to catch.
                caveats.append(
                    f"Accuracy looks high at {accuracy:.1%}, but that is what a constant "
                    "predictor scores on this target — it is not evidence the model learned anything."
                )
        if n_classes > 50:
            caveats.append(f"The target has {n_classes:,} classes, so per-class scores are thin by construction.")
        if score < 0.5 and n_classes == 2:
            caveats.append("A two-class problem scoring under 0.5 macro F1 is worse than a coin flip.")
    else:
        metric_name = "rmse"
        score = float(selected_scores.get("rmse", 0.0))
        baseline = _regression_baseline(y_test)
        beats_baseline = baseline > 0 and score < baseline * (1 - WEAK_MARGIN)
        lift = baseline - score

        r2 = selected_scores.get("r2")
        if not beats_baseline:
            caveats.append(
                f"RMSE of {score:.4g} is no better than the {baseline:.4g} you get from always predicting the mean."
            )
        if isinstance(r2, (int, float)) and r2 < 0:
            caveats.append(f"R² of {r2:.3f} is negative, meaning the model fits worse than a horizontal line.")

    # Cross-validated spread: the honest statement of how much the score moves.
    cv_entry = cv_summary.get(metric_name) or {}
    cv_mean, cv_std = cv_entry.get("mean"), cv_entry.get("std")
    if cv_mean is not None:
        n_folds = len(cv_entry.get("folds") or [])
        notes.append(
            f"Cross-validated {METRIC_LABEL.get(metric_name, metric_name)} "
            f"{cv_mean:.4g} ± {cv_std:.4g} over {n_folds} folds; "
            f"{score:.4g} on the held-out {len(y_test):,} rows that no fold or search touched."
        )
        if cv_std is not None and cv_mean and abs(cv_std) > UNSTABLE_STD * max(abs(cv_mean), 1e-9):
            caveats.append(
                "That spread is wide relative to the mean — the score depends noticeably on which rows are held out."
            )

    caveats.extend(imbalance.get("warnings", []))

    if failed_models:
        caveats.append(f"{len(failed_models)} model(s) failed to train: {', '.join(failed_models)}.")

    if not beats_baseline:
        verdict = "unreliable"
        headline = "This model is not better than guessing"
    elif caveats:
        verdict = "fair"
        headline = "Usable, with caveats"
    else:
        verdict = "strong"
        headline = "Model beats the naive baseline"
        notes.insert(
            0,
            f"Beats the naive baseline by {abs(lift):.4g} on {METRIC_LABEL.get(metric_name, metric_name)}.",
        )

    # Caveats first: what limits the result matters more than what describes it.
    reasons = caveats + notes

    return {
        "verdict": verdict,
        "headline": headline,
        "reasons": reasons,
        "metric": metric_name,
        "metric_label": METRIC_LABEL.get(metric_name, metric_name),
        "score": score,
        "baseline": baseline,
        "beats_baseline": bool(beats_baseline),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "holdout_rows": int(holdout_rows if holdout_rows is not None else len(y_test)),
    }
