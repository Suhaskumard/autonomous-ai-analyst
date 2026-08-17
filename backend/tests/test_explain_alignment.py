"""Feature-importance labels must match the values they are attached to.

Regression guard for the original bug: importances were computed over the
post-one-hot matrix but labeled with the raw dataframe column list, so with any
categorical column present index i of the matrix was not column i of the frame.
"""

import numpy as np
import pytest

from ml.explain import explain_pipeline, map_to_source_columns, transformed_feature_names
from ml.preprocess import prepare_dataset
from ml.train import train_models


@pytest.fixture
def fitted(fixture_df):
    df = fixture_df("clean_classification.csv")
    prepared = prepare_dataset(df, target_column="churn")
    trained = train_models(
        prepared["X"],
        prepared["y"],
        preprocessor=prepared["preprocessor"],
        mode="manual",
        manual_model="DecisionTree",
    )
    return prepared, trained, trained["trained_models"]["DecisionTree"]


def test_one_hot_actually_widens_the_matrix(fitted):
    prepared, _, pipeline = fitted
    names = transformed_feature_names(pipeline.named_steps["preprocessor"])
    # Without this the test would pass vacuously on a purely numeric frame.
    assert len(names) > len(prepared["feature_columns"])


def test_reported_features_are_real_columns(fitted):
    prepared, trained, pipeline = fitted
    importances, method = explain_pipeline(pipeline, trained["X_train"], prepared["feature_columns"])
    assert importances
    assert method in {"shap", "native"}
    assert {item["feature"] for item in importances} <= set(prepared["feature_columns"])


def test_informative_feature_outranks_noise(fitted):
    """'region' determines the label in the fixture; 'spend' is pure noise."""
    prepared, trained, pipeline = fitted
    importances, _ = explain_pipeline(pipeline, trained["X_train"], prepared["feature_columns"])
    ranked = [item["feature"] for item in importances]
    assert ranked.index("region") < ranked.index("spend")


def test_prefix_collision_does_not_merge_columns():
    """'Colour' must not swallow the one-hot children of 'Colour_Code'."""
    names = ["num__age", "cat__Colour_red", "cat__Colour_blue", "cat__Colour_Code_x1"]
    mapped = map_to_source_columns(names, ["age", "Colour", "Colour_Code"])
    assert mapped == ["age", "Colour", "Colour", "Colour_Code"]


def test_one_hot_children_are_summed_into_the_parent(fitted):
    """Each source column appears once, not once per encoded level."""
    prepared, trained, pipeline = fitted
    importances, _ = explain_pipeline(pipeline, trained["X_train"], prepared["feature_columns"])
    labels = [item["feature"] for item in importances]
    assert len(labels) == len(set(labels))


def test_mismatched_names_refuse_rather_than_mislabel(fitted, monkeypatch):
    """If names and matrix width ever disagree, report nothing at all."""
    prepared, trained, pipeline = fitted
    monkeypatch.setattr("ml.explain.transformed_feature_names", lambda _p: ["only", "two"])
    importances, method = explain_pipeline(pipeline, trained["X_train"], prepared["feature_columns"])
    assert importances == []
    assert method == "unavailable"


def test_fallback_is_reported_not_silent(fitted, monkeypatch, caplog):
    """A SHAP failure must downgrade the reported method and log it."""
    import ml.explain as explain_module

    if not explain_module.SHAP_AVAILABLE:
        pytest.skip("shap is not installed in this environment")

    def boom(*_args, **_kwargs):
        raise RuntimeError("shap exploded")

    monkeypatch.setattr(explain_module.shap, "Explainer", boom)
    prepared, trained, pipeline = fitted
    with caplog.at_level("WARNING"):
        _, method = explain_pipeline(pipeline, trained["X_train"], prepared["feature_columns"])
    assert method == "native"
    assert any("SHAP explanation failed" in record.message for record in caplog.records)


def test_importances_are_non_negative(fitted):
    prepared, trained, pipeline = fitted
    importances, _ = explain_pipeline(pipeline, trained["X_train"], prepared["feature_columns"])
    assert all(item["importance"] >= 0 for item in importances)
    assert not any(np.isnan(item["importance"]) for item in importances)
