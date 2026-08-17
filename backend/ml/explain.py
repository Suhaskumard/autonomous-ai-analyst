"""Feature attribution with names that actually match the values.

The old version computed importances over the post-one-hot matrix and labeled
them with the raw dataframe columns. With any categorical feature present,
index i of the matrix is not column i of the dataframe, so every number was
attached to the wrong feature. Names now come from
`preprocessor.get_feature_names_out()`, and the one-hot children are summed
back into their parent column for display.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    SHAP_AVAILABLE = False
    logger.warning("shap is not installed; falling back to native feature importances.")


def transformed_feature_names(preprocessor) -> list[str]:
    """Names of the columns the estimator actually sees, in matrix order."""
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception as exc:
        logger.warning("Could not read transformed feature names: %s", exc)
        return []


def map_to_source_columns(transformed_names: list[str], source_columns: list[str]) -> list[str]:
    """Map 'cat__Colour_red' back to 'Colour' so the UI shows real columns.

    Longest-prefix match, so a column named 'Colour' does not swallow the names
    belonging to 'Colour_Code'.
    """
    ordered_sources = sorted(source_columns, key=len, reverse=True)
    mapped: list[str] = []
    for name in transformed_names:
        bare = name.split("__", 1)[1] if "__" in name else name
        parent = next((src for src in ordered_sources if bare == src or bare.startswith(f"{src}_")), None)
        mapped.append(parent or bare)
    return mapped


def _aggregate(importances: np.ndarray, parents: list[str], top_n: int) -> list[dict]:
    """Sum the contribution of every one-hot child into its parent column."""
    totals: dict[str, float] = {}
    for value, parent in zip(importances, parents):
        totals[parent] = totals.get(parent, 0.0) + float(value)
    result = [{"feature": name, "importance": value} for name, value in totals.items()]
    result.sort(key=lambda item: item["importance"], reverse=True)
    return result[:top_n]


def _native_importances(estimator, n_features: int) -> np.ndarray | None:
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef, dtype=float)
        return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    return None


def explain_pipeline(
    pipeline,
    X_train,
    source_columns: list[str],
    top_n: int = 10,
    sample_size: int = 200,
) -> tuple[list[dict], str]:
    """Return (importances, method) for a fitted preprocessor+estimator Pipeline.

    `method` is "shap", "native", or "unavailable" — the caller surfaces it, so
    a silent fallback can no longer masquerade as a SHAP explanation.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
    except (AttributeError, KeyError):
        logger.warning("explain_pipeline received an object that is not a preprocessor+model Pipeline.")
        return [], "unavailable"

    names = transformed_feature_names(preprocessor)
    if not names:
        return [], "unavailable"
    parents = map_to_source_columns(names, source_columns)

    try:
        X_transformed = preprocessor.transform(X_train)
    except Exception as exc:
        logger.warning("Could not transform training rows for explanation: %s", exc)
        return [], "unavailable"

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed)

    if X_transformed.shape[0] == 0:
        return [], "unavailable"
    if X_transformed.shape[1] != len(names):
        # Would reintroduce the original misalignment bug; refuse instead.
        logger.warning(
            "Feature-name count (%d) does not match the transformed matrix width (%d); skipping explanation.",
            len(names),
            X_transformed.shape[1],
        )
        return [], "unavailable"

    sample = X_transformed[: min(sample_size, X_transformed.shape[0])]

    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(estimator, sample)
            values = explainer(sample).values
            if values.ndim == 3:
                # (rows, features, classes) -> average magnitude across classes
                values = np.abs(values).mean(axis=2)
            importance = np.mean(np.abs(values), axis=0)
            if importance.shape[0] == len(names):
                return _aggregate(importance, parents, top_n), "shap"
            logger.warning("SHAP returned %d values for %d features; falling back.", importance.shape[0], len(names))
        except Exception as exc:
            # Loud on purpose: this used to be swallowed, so a native-importance
            # result was presented to the user as a SHAP explanation.
            logger.warning("SHAP explanation failed (%s); falling back to native importances.", exc, exc_info=True)

    native = _native_importances(estimator, len(names))
    if native is not None and native.shape[0] == len(names):
        return _aggregate(native, parents, top_n), "native"

    logger.info("No explanation available for estimator %s.", type(estimator).__name__)
    return [], "unavailable"
