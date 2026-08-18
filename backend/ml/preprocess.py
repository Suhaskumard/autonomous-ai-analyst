"""Dataset preparation.

Nothing here fits anything. The preprocessor is *built* but never fitted: it is
handed to train.py, which puts it inside a Pipeline and fits it on the training
split only. Fitting an imputer/scaler/encoder on the whole frame before the
split leaks test-set statistics into training and makes every reported score
optimistic by an unknown margin, which is exactly what this module used to do.
"""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from exceptions import TargetValidationError

# A categorical feature contributes at most this many one-hot columns; rarer
# levels are folded into a single "infrequent" column. Without a ceiling, one
# free-text column explodes into thousands of features and OOMs the fit.
MAX_ONEHOT_CATEGORIES = 50

# Above this many distinct values, one-hot encoding is the wrong tool: it makes
# a wide sparse block whose rare columns carry almost no signal, and capping it
# throws the tail away. Such columns are target-encoded instead — one dense
# column per class, fitted with sklearn's internal cross-fitting so the encoding
# of a row never uses that row's own target.
TARGET_ENCODE_MIN_CATEGORIES = 25

# A feature column this text-like is dropped rather than encoded.
FREE_TEXT_UNIQUE_RATIO = 0.5
FREE_TEXT_MEAN_LENGTH = 60

# A *target* this text-like is refused outright, with an explanation.
TARGET_MAX_CLASSES = 100
TARGET_MAX_CLASS_RATIO = 0.5


def detect_target_column(df: pd.DataFrame) -> str:
    """Heuristic fallback used only when the caller does not name a target."""
    lowered = {c.lower(): c for c in df.columns}
    for candidate in ("target", "label", "class", "outcome", "y"):
        if candidate in lowered:
            return lowered[candidate]
    return df.columns[-1]


def resolve_target_column(df: pd.DataFrame, target_column: str | None) -> tuple[str, bool]:
    """Return (target, was_auto_detected), validating an explicit choice."""
    if target_column:
        if target_column not in df.columns:
            raise TargetValidationError(
                f"Target column '{target_column}' is not in the dataset. "
                f"Available columns: {', '.join(map(str, df.columns[:20]))}" + ("..." if len(df.columns) > 20 else "")
            )
        return target_column, False
    return detect_target_column(df), True


def _text_profile(series: pd.Series) -> tuple[int, float, float]:
    """(n_unique, unique ratio, mean string length) for a column."""
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    ratio = n_unique / max(len(non_null), 1)
    mean_len = float(non_null.astype(str).str.len().mean()) if len(non_null) else 0.0
    return n_unique, ratio, mean_len


def validate_target(y: pd.Series, target_col: str, auto_detected: bool) -> None:
    """Refuse targets that cannot be a label, with a message that says why.

    This is the guard that was missing when a free-text clinical narrative was
    picked as the target: 14,555 unique paragraphs were treated as 14,555
    classes, and the run "succeeded" at 1.7% accuracy.
    """
    non_null = y.dropna()
    if non_null.empty:
        raise TargetValidationError(f"Target column '{target_col}' is entirely empty.")

    if non_null.nunique() < 2:
        raise TargetValidationError(
            f"Target column '{target_col}' has a single value across all "
            f"{len(non_null):,} rows, so there is nothing to predict."
        )

    if pd.api.types.is_numeric_dtype(non_null):
        return

    n_unique, ratio, mean_len = _text_profile(non_null)
    hint = (
        " Pick a different target column."
        if not auto_detected
        else " It was auto-detected as the last column; choose the real target explicitly."
    )
    if n_unique > TARGET_MAX_CLASSES and ratio > TARGET_MAX_CLASS_RATIO:
        raise TargetValidationError(
            f"'{target_col}' has {n_unique:,} unique values across {len(non_null):,} rows "
            f"— this looks like free text or an ID, not a label.{hint}"
        )
    if mean_len > FREE_TEXT_MEAN_LENGTH:
        raise TargetValidationError(
            f"'{target_col}' averages {mean_len:.0f} characters per value "
            f"— this looks like free text, not a label.{hint}"
        )
    if n_unique > TARGET_MAX_CLASSES:
        raise TargetValidationError(
            f"'{target_col}' has {n_unique:,} distinct classes, above the "
            f"{TARGET_MAX_CLASSES} supported. Group the values into fewer categories first.{hint}"
        )


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """Sort features into (numeric, one-hot, target-encoded, dropped-as-free-text)."""
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    candidates = [c for c in X.columns if c not in numeric]

    categorical: list[str] = []
    high_cardinality: list[str] = []
    dropped: list[str] = []
    for col in candidates:
        n_unique, ratio, mean_len = _text_profile(X[col])
        if ratio > FREE_TEXT_UNIQUE_RATIO or mean_len > FREE_TEXT_MEAN_LENGTH:
            dropped.append(col)
        elif n_unique >= TARGET_ENCODE_MIN_CATEGORIES:
            high_cardinality.append(col)
        else:
            categorical.append(col)
    return numeric, categorical, high_cardinality, dropped


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    high_cardinality_cols: list[str] | None = None,
) -> ColumnTransformer:
    """An *unfitted* ColumnTransformer. train.py fits it on the training split."""
    high_cardinality_cols = high_cardinality_cols or []

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=MAX_ONEHOT_CATEGORIES,
                    sparse_output=False,
                ),
            ),
        ]
    )
    # TargetEncoder is supervised: it needs y, which ColumnTransformer forwards
    # from Pipeline.fit. Its internal cross-fitting is what stops a category's
    # encoding from memorising the rows it was computed from.
    target_encoded_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target", TargetEncoder(target_type="auto", random_state=42)),
        ]
    )

    transformers = [
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ]
    if high_cardinality_cols:
        transformers.append(("cat_high", target_encoded_pipeline, high_cardinality_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prepare_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    """Choose and validate the target, sort feature types, build a preprocessor.

    Returns raw (unfitted, unsplit) X and y. All fitting happens downstream,
    after the train/test split.
    """
    target_col, auto_detected = resolve_target_column(df, target_column)

    before_rows = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    dropped_target_rows = before_rows - len(df)

    y = df[target_col]
    validate_target(y, target_col, auto_detected)

    X = df.drop(columns=[target_col])
    if X.shape[1] == 0:
        raise TargetValidationError("The dataset has no feature columns once the target is removed.")

    num_cols, cat_cols, high_card_cols, dropped_cols = split_feature_types(X)
    if not num_cols and not cat_cols and not high_card_cols:
        raise TargetValidationError(
            "Every feature column looks like free text or a unique identifier, so there is nothing usable to train on."
        )

    X = X[num_cols + cat_cols + high_card_cols]
    capped = [c for c in cat_cols if X[c].nunique() > MAX_ONEHOT_CATEGORIES]

    warnings: list[str] = []
    if dropped_cols:
        warnings.append(
            f"Dropped {len(dropped_cols)} free-text/identifier column(s) that cannot be encoded: "
            f"{', '.join(dropped_cols[:5])}{'...' if len(dropped_cols) > 5 else ''}."
        )
    if high_card_cols:
        warnings.append(
            f"Target-encoded {len(high_card_cols)} high-cardinality column(s) instead of one-hot: "
            f"{', '.join(high_card_cols[:5])}{'...' if len(high_card_cols) > 5 else ''}. "
            "Each becomes a dense column fitted with internal cross-fitting, so no row sees its own target."
        )
    if capped:
        warnings.append(
            f"Capped one-hot encoding at {MAX_ONEHOT_CATEGORIES} categories for "
            f"{', '.join(capped[:5])}{'...' if len(capped) > 5 else ''}; "
            "rarer values are grouped as 'infrequent'."
        )
    if dropped_target_rows:
        warnings.append(f"Dropped {dropped_target_rows} rows with a missing target value.")

    return {
        "X": X,
        "y": y,
        "target_col": target_col,
        "target_auto_detected": auto_detected,
        "dropped_target_rows": dropped_target_rows,
        "preprocessor": build_preprocessor(num_cols, cat_cols, high_card_cols),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "high_cardinality_columns": high_card_cols,
        "dropped_columns": dropped_cols,
        "feature_columns": X.columns.tolist(),
        "warnings": warnings,
    }
