from datetime import datetime
from io import StringIO
from pathlib import Path
import re

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "saved_models"
METADATA_DIR = BASE_DIR / "models" / "metadata"


def ensure_storage_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_csv_flexible(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    text = file_bytes.decode("utf-8-sig", errors="ignore")
    warnings: list[str] = []

    if len(text.strip()) == 0:
        raise ValueError("Uploaded file appears empty or unreadable.")

    printable_chars = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
    printable_ratio = printable_chars / max(len(text), 1)
    if printable_ratio < 0.85:
        raise ValueError("Uploaded file does not look like a valid text CSV (possible binary/corrupted file).")

    parsers = [
        {"sep": None, "engine": "python", "on_bad_lines": "skip"},
        {"sep": ",", "engine": "python", "on_bad_lines": "skip"},
    ]

    last_error = None
    for parser in parsers:
        try:
            df = pd.read_csv(StringIO(text), **parser)
            df = sanitize_dataframe(df)
            if df.empty:
                warnings.append("Parsed dataset is empty after handling malformed rows.")
            if parser["sep"] is None:
                warnings.append("Auto-detected delimiter and skipped malformed CSV rows when needed.")
            else:
                warnings.append("Used fallback CSV parser and skipped malformed rows.")
            return df, warnings
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not parse CSV content: {last_error}")


def sanitize_text(value: object, max_len: int = 80) -> str:
    text = str(value)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    # Keep output UI-safe and human-readable. Remove non-ASCII noise.
    text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:max_len] if len(text) > max_len else text) if text else ""


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    clean_columns: list[str] = []
    seen: set[str] = set()
    for i, col in enumerate(df.columns):
        c = sanitize_text(col, max_len=60)
        if not c:
            c = f"column_{i+1}"
        original = c
        suffix = 2
        while c in seen:
            c = f"{original}_{suffix}"
            suffix += 1
        seen.add(c)
        clean_columns.append(c)
    df.columns = clean_columns

    # Sanitize object-like cells to prevent unreadable glyphs in charts/chat output.
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].apply(lambda v: sanitize_text(v, max_len=120) if pd.notna(v) else v)
        unreadable_ratio = (df[col].astype(str).str.len() == 0).mean()
        if unreadable_ratio > 0.6:
            raise ValueError(
                f"Column '{col}' appears unreadable/corrupted after sanitization. "
                "Please verify file encoding and upload a clean CSV."
            )
    return df
