"""A verdict on whether a run is worth believing.

The committed 1.7%-accuracy run was presented in the UI as the "Optimized
Model" with no caveat. A score is only meaningful next to the trivial baseline
it should beat: always-predict-the-majority-class for classification,
always-predict-the-mean for regression. This module turns that comparison into
a verdict the frontend can render prominently.
"""

import numpy as np

# A model this far below its own baseline is worse than a constant predictor.
WEAK_MARGIN = 0.02


def _classification_baseline(y_test: np.ndarray) -> float:
    """Accuracy of always predicting the most common class in the test split."""
    if len(y_test) == 0:
        return 0.0
    _, counts = np.unique(y_test, return_counts=True)
    return float(counts.max() / len(y_test))


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
) -> dict:
    """Return {verdict, headline, reasons, baseline, score} for the chosen model.

    verdict is one of: "strong", "fair", "unreliable".
    """
    y_test = np.asarray(y_test)
    reasons: list[str] = []

    if problem_type == "classification":
        score = float(selected_scores.get("accuracy", 0.0))
        baseline = _classification_baseline(y_test)
        metric_name = "accuracy"
        beats_baseline = score > baseline + WEAK_MARGIN
        lift = score - baseline

        if not beats_baseline:
            reasons.append(
                f"Accuracy of {score:.1%} does not beat the {baseline:.1%} you get from "
                "always predicting the most common class."
            )
        if n_classes > 50:
            reasons.append(f"The target has {n_classes:,} classes, so per-class accuracy is thin by construction.")
        if score < 0.5 and n_classes == 2:
            reasons.append("A two-class problem scoring under 50% is worse than a coin flip.")
    else:
        score = float(selected_scores.get("rmse", 0.0))
        baseline = _regression_baseline(y_test)
        metric_name = "rmse"
        beats_baseline = baseline > 0 and score < baseline * (1 - WEAK_MARGIN)
        lift = baseline - score

        if not beats_baseline:
            reasons.append(
                f"RMSE of {score:.4g} is no better than the {baseline:.4g} you get from always predicting the mean."
            )

    if failed_models:
        reasons.append(f"{len(failed_models)} model(s) failed to train: {', '.join(failed_models)}.")

    if not beats_baseline:
        verdict = "unreliable"
        headline = "This model is not better than guessing"
    elif reasons:
        verdict = "fair"
        headline = "Usable, with caveats"
    else:
        verdict = "strong"
        headline = "Model beats the naive baseline"
        reasons.append(f"Beats the naive baseline by {abs(lift):.4g} on {metric_name}.")

    return {
        "verdict": verdict,
        "headline": headline,
        "reasons": reasons,
        "metric": metric_name,
        "score": score,
        "baseline": baseline,
        "beats_baseline": bool(beats_baseline),
    }
