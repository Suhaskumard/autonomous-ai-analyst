import numpy as np
from scipy.stats import mode


def ensemble_predict(models: dict, X, problem_type: str):
    preds = np.array([model.predict(X) for model in models.values()])
    if problem_type == "classification":
        return mode(preds, axis=0, keepdims=False).mode
    return np.mean(preds, axis=0)
