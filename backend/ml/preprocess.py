from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_target_column(df: pd.DataFrame) -> str:
    lowered = {c.lower(): c for c in df.columns}
    if "target" in lowered:
        return lowered["target"]
    if "label" in lowered:
        return lowered["label"]
    return df.columns[-1]


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def preprocess_dataset(df: pd.DataFrame) -> dict[str, Any]:
    target_col = detect_target_column(df)
    before_rows = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    dropped_target_rows = before_rows - len(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    # Convert sparse matrix to dense array if necessary
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    return {
        "X_raw": X,
        "X_processed": X_processed,
        "y": y,
        "target_col": target_col,
        "dropped_target_rows": dropped_target_rows,
        "preprocessing_pipeline": preprocessor,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "feature_columns": X.columns.tolist(),
    }
