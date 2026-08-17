import numpy as np
from scipy.stats import mode


def ensemble_predict(models: dict, X, problem_type: str):
    """Vote across fitted Pipelines.

    Each member is a full preprocessor+estimator Pipeline, so `X` is the raw
    feature frame — the same input every member was fitted on. Classification
    predictions are label-encoded integers; the caller inverse-transforms them.
    """
    preds = np.array([model.predict(X) for model in models.values()])
    if problem_type == "classification":
        return mode(preds, axis=0, keepdims=False).mode
    return np.mean(preds, axis=0)
