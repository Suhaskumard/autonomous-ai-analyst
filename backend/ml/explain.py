import numpy as np
import shap


def explain_model(model, X_sample, feature_names: list[str], top_n: int = 10) -> list[dict]:
    row_count = int(getattr(X_sample, "shape", [0])[0]) if hasattr(X_sample, "shape") else 0
    if row_count == 0:
        return []
    sample = X_sample[: min(200, row_count)]

    try:
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 0]
        importance = np.mean(np.abs(values), axis=0)
        result = []
        for i in range(len(importance)):
            feature = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            result.append({"feature": feature, "importance": float(importance[i])})
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result[:top_n]
    except Exception:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            result = []
            for i in range(min(len(fi), len(feature_names))):
                result.append({"feature": feature_names[i], "importance": float(fi[i])})
            result.sort(key=lambda x: x["importance"], reverse=True)
            return result[:top_n]
        return []
