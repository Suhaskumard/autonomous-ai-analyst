"""The preprocessor must be fitted on training rows only.

Regression guard for the original bug: imputer/scaler/encoder were
fit_transform-ed on the entire dataset and only then was the data split, so
test-set statistics bled into training and every score was optimistic.
"""

import numpy as np

from ml.preprocess import prepare_dataset
from ml.train import train_models


def _fit(df, target, model):
    prepared = prepare_dataset(df, target_column=target)
    trained = train_models(
        prepared["X"], prepared["y"], preprocessor=prepared["preprocessor"], mode="manual", manual_model=model
    )
    return prepared, trained, trained["trained_models"][model]


def test_scaler_statistics_come_from_the_training_split(fixture_df):
    df = fixture_df("drifting_scale.csv")
    prepared, trained, pipeline = _fit(df, "y", "DecisionTree")
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]

    numeric = prepared["numeric_columns"]
    train_mean = trained["X_train"][numeric].mean().to_numpy()
    full_mean = prepared["X"][numeric].mean().to_numpy()

    assert np.allclose(scaler.mean_, train_mean)
    # The fixture's scale shifts halfway through the file, so a whole-file fit
    # would produce a visibly different mean. This is the actual leak detector.
    assert not np.allclose(scaler.mean_, full_mean)


def test_scaler_saw_exactly_the_training_rows(fixture_df):
    df = fixture_df("drifting_scale.csv")
    _, trained, pipeline = _fit(df, "y", "DecisionTree")
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(trained["X_train"])


def test_encoder_categories_come_from_the_training_split(fixture_df):
    df = fixture_df("clean_classification.csv")
    _, trained, pipeline = _fit(df, "churn", "DecisionTree")
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    train_regions = set(trained["X_train"]["region"].unique())
    assert set(encoder.categories_[0]) <= train_regions


def test_preprocessor_is_unfitted_before_training(fixture_df):
    """prepare_dataset must hand back a preprocessor that has fitted nothing."""
    prepared = prepare_dataset(fixture_df("clean_classification.csv"), target_column="churn")
    assert not hasattr(prepared["preprocessor"], "transformers_")


def test_serving_artifact_is_a_single_pipeline(fixture_df):
    _, _, pipeline = _fit(fixture_df("clean_classification.csv"), "churn", "RandomForest")
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
