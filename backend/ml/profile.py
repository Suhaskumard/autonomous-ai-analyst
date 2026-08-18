"""Per-column data profile, computed before any training happens.

The upload screen used to ask for a target column with no information about
what the columns contain. This produces the profile that screen renders: type,
missing share, cardinality, a distribution sparkline, and — importantly — a
verdict on whether each column can serve as a target, so the picker can
disable the ones that would fail.
"""

from typing import Any

import numpy as np
import pandas as pd

from ml.preprocess import (
    FREE_TEXT_MEAN_LENGTH,
    FREE_TEXT_UNIQUE_RATIO,
    MAX_ONEHOT_CATEGORIES,
    TARGET_MAX_CLASS_RATIO,
    TARGET_MAX_CLASSES,
    detect_target_column,
)
from utils.helpers import sanitize_text

SPARKLINE_BINS = 12
TOP_CATEGORIES = 8


def _numeric_sparkline(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"kind": "histogram", "bins": [], "counts": []}
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=SPARKLINE_BINS)
    return {
        "kind": "histogram",
        "bins": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
    }


def _categorical_sparkline(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().astype(str).value_counts()
    top = counts.head(TOP_CATEGORIES)
    payload = {
        "kind": "categories",
        "labels": [sanitize_text(label, max_len=32) for label in top.index.tolist()],
        "counts": [int(value) for value in top.to_numpy()],
    }
    remainder = int(counts.iloc[TOP_CATEGORIES:].sum()) if len(counts) > TOP_CATEGORIES else 0
    if remainder:
        # Never a generated 9th colour: the tail folds into one "Other" bar.
        payload["labels"].append("Other")
        payload["counts"].append(remainder)
    return payload


def _target_verdict(series: pd.Series, n_rows: int) -> tuple[bool, str]:
    """Can this column be a target? Mirrors preprocess.validate_target."""
    non_null = series.dropna()
    if non_null.empty:
        return False, "Column is entirely empty."
    n_unique = int(non_null.nunique())
    if n_unique < 2:
        return False, "Only one distinct value — nothing to predict."
    if pd.api.types.is_numeric_dtype(non_null):
        return True, "Numeric — usable as a regression or classification target."

    ratio = n_unique / max(len(non_null), 1)
    mean_len = float(non_null.astype(str).str.len().mean())
    if n_unique > TARGET_MAX_CLASSES and ratio > TARGET_MAX_CLASS_RATIO:
        return False, f"{n_unique:,} unique values across {len(non_null):,} rows — looks like free text or an ID."
    if mean_len > FREE_TEXT_MEAN_LENGTH:
        return False, f"Averages {mean_len:.0f} characters per value — looks like free text."
    if n_unique > TARGET_MAX_CLASSES:
        return False, f"{n_unique:,} classes, above the {TARGET_MAX_CLASSES} supported."
    return True, f"{n_unique:,} classes — usable as a classification target."


def _feature_note(series: pd.Series, is_numeric: bool) -> str | None:
    """Warn about features the pipeline will drop or cap, before training."""
    if is_numeric:
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    n_unique = int(non_null.nunique())
    ratio = n_unique / max(len(non_null), 1)
    mean_len = float(non_null.astype(str).str.len().mean())
    if ratio > FREE_TEXT_UNIQUE_RATIO or mean_len > FREE_TEXT_MEAN_LENGTH:
        return "Will be dropped as free text / identifier."
    if n_unique > MAX_ONEHOT_CATEGORIES:
        return f"Will be one-hot capped at {MAX_ONEHOT_CATEGORIES} categories; rarer values grouped."
    return None


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Column-by-column profile plus a suggested target."""
    n_rows = int(df.shape[0])
    columns: list[dict[str, Any]] = []

    for name in df.columns:
        series = df[name]
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        missing = int(series.isna().sum())
        non_null = series.dropna()
        usable_as_target, target_reason = _target_verdict(series, n_rows)

        entry: dict[str, Any] = {
            "name": str(name),
            "dtype": "numeric" if is_numeric else "categorical",
            "missing": missing,
            "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0.0,
            "unique": int(series.nunique(dropna=True)),
            "unique_pct": round(series.nunique(dropna=True) / n_rows * 100, 2) if n_rows else 0.0,
            "usable_as_target": usable_as_target,
            "target_reason": target_reason,
            "feature_note": _feature_note(series, is_numeric),
            "sparkline": _numeric_sparkline(series) if is_numeric else _categorical_sparkline(series),
        }

        if is_numeric and not non_null.empty:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if not numeric.empty:
                entry["stats"] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "std": float(numeric.std()) if len(numeric) > 1 else 0.0,
                }
        elif not non_null.empty:
            top = non_null.astype(str).value_counts()
            entry["stats"] = {
                "most_frequent": sanitize_text(top.index[0], max_len=40),
                "most_frequent_count": int(top.iloc[0]),
                "sample": [sanitize_text(v, max_len=40) for v in non_null.astype(str).head(3).tolist()],
            }

        columns.append(entry)

    suggested = detect_target_column(df) if len(df.columns) else None
    return {
        "rows": n_rows,
        "columns": int(df.shape[1]),
        "suggested_target": str(suggested) if suggested is not None else None,
        "column_profiles": columns,
    }
