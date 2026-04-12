import pandas as pd


def data_quality_report(df: pd.DataFrame) -> dict:
    missing_counts = df.isna().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    row_count = int(df.shape[0])

    warnings: list[str] = []
    if duplicate_rows > 0:
        warnings.append(f"Found {duplicate_rows} duplicate rows; removed duplicates.")
    high_missing = [c for c, n in missing_counts.items() if row_count > 0 and (n / row_count) > 0.4]
    if high_missing:
        warnings.append(f"Columns with >40% missing values: {', '.join(high_missing)}.")

    return {
        "rows": row_count,
        "columns": int(df.shape[1]),
        "missing_counts": missing_counts,
        "duplicate_rows": duplicate_rows,
        "warnings": warnings,
    }


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)
