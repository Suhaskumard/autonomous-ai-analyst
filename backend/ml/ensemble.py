import numpy as np
from scipy.stats import mode


def ensemble_predict(models: dict, X, problem_type: str):
    """Vote across fitted Pipelines. Legacy path — kept only for old artifacts.

    New runs build a weighted VotingClassifier/VotingRegressor in ml.train,
    which is a single fitted estimator and carries calibrated probabilities.
    This unweighted mode/mean vote gave a near-chance member the same say as
    the best one and had no confidence to report; it survives only so that
    bundles saved before that change (`kind == "ensemble"`) still serve.

    Each member is a full preprocessor+estimator Pipeline, so `X` is the raw
    feature frame — the same input every member was fitted on. Classification
    predictions are label-encoded integers; the caller inverse-transforms them.
    """
    preds = np.array([model.predict(X) for model in models.values()])
    if problem_type == "classification":
        return mode(preds, axis=0, keepdims=False).mode
    return np.mean(preds, axis=0)
