"""Target and task detection on adversarial columns.

Regression guard for Exhibit A: the last column was a free-text clinical
narrative, 14,555 unique paragraphs were treated as classes, and a decision
tree "won" with 1.7% accuracy without anything flagging it.
"""

import pandas as pd
import pytest

from exceptions import TargetValidationError
from ml.preprocess import MAX_ONEHOT_CATEGORIES, prepare_dataset
from ml.train import detect_problem_type, train_models


def test_free_text_target_is_refused_with_an_explanation(fixture_df):
    with pytest.raises(TargetValidationError) as excinfo:
        prepare_dataset(fixture_df("free_text_target.csv"))
    message = str(excinfo.value)
    assert "Findings" in message
    assert "unique values" in message or "characters per value" in message


def test_identifier_target_is_refused(fixture_df):
    with pytest.raises(TargetValidationError):
        prepare_dataset(fixture_df("id_target.csv"))


def test_explicit_target_overrides_the_last_column_heuristic(fixture_df):
    """The escape hatch from Exhibit A: name a usable target instead."""
    prepared = prepare_dataset(fixture_df("free_text_target.csv"), target_column="state")
    assert prepared["target_col"] == "state"
    assert prepared["target_auto_detected"] is False
    assert "Findings" in prepared["dropped_columns"]


def test_unknown_target_name_is_rejected(fixture_df):
    with pytest.raises(TargetValidationError) as excinfo:
        prepare_dataset(fixture_df("clean_classification.csv"), target_column="nope")
    assert "not in the dataset" in str(excinfo.value)


def test_constant_target_is_rejected():
    df = pd.DataFrame({"a": range(50), "b": range(50), "y": ["same"] * 50})
    with pytest.raises(TargetValidationError):
        prepare_dataset(df, target_column="y")


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([1.5, 2.5, 3.5, 4.5] * 40), "regression"),  # few uniques, still continuous
        (pd.Series([0, 1, 2] * 60), "classification"),
        (pd.Series(range(500)), "regression"),
        (pd.Series(["a", "b"] * 40), "classification"),
        (pd.Series([True, False] * 40), "classification"),
    ],
)
def test_problem_type_detection(series, expected):
    assert detect_problem_type(series) == expected


def test_low_cardinality_regression_is_not_called_classification():
    """The old rule (<=20 uniques => classification) got this wrong."""
    prices = pd.Series([19.99, 29.99, 39.99, 49.99, 59.99] * 50)
    assert detect_problem_type(prices) == "regression"


def test_high_cardinality_feature_is_target_encoded_not_exploded(fixture_df):
    """180 cities become one dense column, not 180 sparse ones.

    Capping one-hot at 50 kept the width sane but threw the tail away. Target
    encoding keeps every level and costs one column, which is the whole reason
    the categorical-native path exists.
    """
    df = fixture_df("high_cardinality.csv")
    prepared = prepare_dataset(df, target_column="label")

    assert "city" in prepared["high_cardinality_columns"], "should be target-encoded, not dropped"
    assert "city" not in prepared["categorical_columns"], "should not go down the one-hot path"

    trained = train_models(
        prepared["X"],
        prepared["y"],
        preprocessor=prepared["preprocessor"],
        mode="manual",
        manual_model="DecisionTree",
    )
    preprocessor = trained["trained_models"]["DecisionTree"].named_steps["preprocessor"]
    width = preprocessor.transform(prepared["X"]).shape[1]
    # num + city -> 2 columns, nowhere near the 180 a raw one-hot would make.
    assert width <= MAX_ONEHOT_CATEGORIES, f"encoding exploded to {width} columns"
    assert any("Target-encoded" in warning for warning in prepared["warnings"])


def test_target_encoded_names_still_map_back_to_their_source_column(fixture_df):
    """Attribution must survive the new encoder, not just the one-hot one."""
    from ml.explain import map_to_source_columns, transformed_feature_names

    df = fixture_df("high_cardinality.csv")
    prepared = prepare_dataset(df, target_column="label")
    trained = train_models(
        prepared["X"],
        prepared["y"],
        preprocessor=prepared["preprocessor"],
        mode="manual",
        manual_model="DecisionTree",
    )
    preprocessor = trained["trained_models"]["DecisionTree"].named_steps["preprocessor"]

    names = transformed_feature_names(preprocessor)
    parents = map_to_source_columns(names, prepared["feature_columns"])

    assert "city" in parents
    assert set(parents) <= set(prepared["feature_columns"])


def test_moderate_cardinality_still_goes_through_one_hot():
    """The categorical-native path is for wide columns, not for every column."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "colour": ["red", "green", "blue"] * 40,
            "num": list(range(120)),
            "label": ["a", "b"] * 60,
        }
    )
    prepared = prepare_dataset(df, target_column="label")
    assert "colour" in prepared["categorical_columns"]
    assert prepared["high_cardinality_columns"] == []


def test_free_text_feature_columns_are_dropped(fixture_df):
    prepared = prepare_dataset(fixture_df("free_text_target.csv"), target_column="state")
    assert "Findings" not in prepared["feature_columns"]
    assert any("free-text" in warning for warning in prepared["warnings"])
