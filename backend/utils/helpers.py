import csv
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd

from exceptions import DatasetError

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "saved_models"
METADATA_DIR = BASE_DIR / "models" / "metadata"
# Uploads waiting for an out-of-process worker. A temp file in the system temp
# directory is fine when the pipeline runs in the web process, but an RQ worker
# is a different container with a different /tmp — the handover has to happen
# somewhere both of them mount, which is the same volume as the artifacts.
SPOOL_DIR = BASE_DIR / "models" / "spool"

# Unicode general categories that are never legitimate data: control (Cc),
# format/zero-width (Cf), surrogates (Cs), unassigned (Cn). Everything else —
# every script on earth — is kept.
_UNPRINTABLE_CATEGORIES = {"Cc", "Cf", "Cs", "Cn"}


def ensure_storage_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    """Timezone-aware UTC now. datetime.utcnow() is deprecated and naive."""
    return datetime.now(UTC).replace(microsecond=0)


def now_utc_iso() -> str:
    return now_utc().isoformat()


def as_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to a datetime that came back from the database naive.

    Timestamp columns are naive and hold UTC — SQLite drops the offset on write
    and the Postgres engine pins the session timezone to match. That is a
    coherent contract right up to the moment Python compares a stored value
    with `now_utc()`, which is aware, and gets `TypeError: can't compare
    offset-naive and offset-aware datetimes`. Anything doing that comparison
    goes through here.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


@dataclass
class ParseReport:
    """What actually happened while reading the file.

    The old code skipped malformed rows with `on_bad_lines="skip"` and never
    counted them, then emitted a generic warning that never said how many rows
    vanished. Every number here is reported to the user.
    """

    delimiter: str | None = None
    rows_parsed: int = 0
    rows_skipped: int = 0
    rows_truncated: int = 0
    rows_padded: int = 0
    skipped_examples: list[str] = field(default_factory=list)
    encoding: str = "utf-8"
    replaced_characters: int = 0

    @property
    def rows_affected(self) -> int:
        return self.rows_skipped + self.rows_truncated + self.rows_padded

    @property
    def warnings(self) -> list[str]:
        messages: list[str] = []
        if self.rows_skipped:
            preview = f" First one: {self.skipped_examples[0]}" if self.skipped_examples else ""
            messages.append(
                f"Skipped {self.rows_skipped:,} malformed row(s) the parser could not read; "
                f"{self.rows_parsed:,} rows were kept.{preview}"
            )
        if self.rows_truncated:
            # pandas keeps these rows and quietly drops the surplus values, so
            # the count has to be measured against the header rather than
            # inferred from anything the parser reports.
            messages.append(
                f"{self.rows_truncated:,} row(s) had more values than the header has columns; "
                "the extra values were dropped."
            )
        if self.rows_padded:
            messages.append(
                f"{self.rows_padded:,} row(s) had fewer values than the header has columns; "
                "the missing values were treated as empty."
            )
        if self.replaced_characters:
            messages.append(
                f"Removed {self.replaced_characters:,} control/zero-width character(s). "
                "Letters, accents, and non-Latin scripts were preserved."
            )
        if self.encoding != "utf-8":
            messages.append(f"File was not valid UTF-8; decoded as {self.encoding}.")
        return messages

    def as_dict(self) -> dict:
        return {
            "delimiter": self.delimiter,
            "rows_parsed": self.rows_parsed,
            "rows_skipped": self.rows_skipped,
            "rows_truncated": self.rows_truncated,
            "rows_padded": self.rows_padded,
            "rows_affected": self.rows_affected,
            "skipped_examples": self.skipped_examples,
            "encoding": self.encoding,
            "replaced_characters": self.replaced_characters,
            "warnings": self.warnings,
        }


def _decode(file_bytes: bytes) -> tuple[str, str]:
    """Decode without destroying non-English text."""
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding), encoding.replace("-sig", "")
        except (UnicodeDecodeError, UnicodeError):
            continue
    return file_bytes.decode("utf-8", errors="replace"), "utf-8 (with replacements)"


CANDIDATE_DELIMITERS = (",", ";", "\t", "|")
_SHAPE_SAMPLE_LINES = 200


def detect_delimiter(text: str) -> str:
    """Pick the delimiter that splits the file most consistently.

    Scored on how many of the sampled lines agree on a field count, with the
    field count itself as the tie-break — so a file of prose does not get
    "detected" as single-column when a real delimiter is present.
    """
    sample = [line for line in text.splitlines()[:_SHAPE_SAMPLE_LINES] if line.strip()]
    if not sample:
        return ","

    best = (0.0, 0, ",")
    for candidate in CANDIDATE_DELIMITERS:
        counts: dict[int, int] = {}
        for row in csv.reader(sample, delimiter=candidate):
            counts[len(row)] = counts.get(len(row), 0) + 1
        if not counts:
            continue
        modal_fields = max(counts, key=lambda fields: counts[fields])
        if modal_fields < 2:
            continue
        agreement = counts[modal_fields] / len(sample)
        if (agreement, modal_fields) > (best[0], best[1]):
            best = (agreement, modal_fields, candidate)
    return best[2]


def audit_row_shapes(text: str, delimiter: str) -> tuple[int, int]:
    """Count data rows with more / fewer fields than the header.

    pandas keeps both kinds — dropping the surplus values in one case and
    filling with NaN in the other — without reporting either. Measuring the
    raw rows against the header is the only way to tell the user the truth.
    """
    try:
        reader = csv.reader(StringIO(text), delimiter=delimiter)
        header = next(reader, None)
        if not header:
            return 0, 0
        expected = len(header)
        over_long = short = 0
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) > expected:
                over_long += 1
            elif len(row) < expected:
                short += 1
        return over_long, short
    except csv.Error:
        return 0, 0


def read_csv_flexible(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Backwards-compatible wrapper returning (dataframe, warnings)."""
    df, report = read_csv_with_report(file_bytes)
    return df, report.warnings


def read_csv_with_report(file_bytes: bytes) -> tuple[pd.DataFrame, ParseReport]:
    text, encoding = _decode(file_bytes)

    if len(text.strip()) == 0:
        raise DatasetError("Uploaded file appears empty or unreadable.")

    # A binary file is mostly characters no text encoding would produce. This
    # counts Unicode printability, so a Japanese or Arabic CSV passes.
    printable = sum(1 for ch in text if ch in "\r\n\t" or unicodedata.category(ch) not in _UNPRINTABLE_CATEGORIES)
    if printable / max(len(text), 1) < 0.85:
        raise DatasetError("Uploaded file does not look like a valid text CSV (possible binary/corrupted file).")

    delimiter = detect_delimiter(text)
    # Ordered fallbacks, but always with a known delimiter — pandas' own
    # sniffer (sep=None) hides which one it chose, and the row-shape audit
    # below needs to split the file the same way the parser did.
    attempts: list[str] = list(dict.fromkeys([delimiter, ",", ";", "\t", "|"]))

    last_error: Exception | None = None
    for separator in attempts:
        skipped: list[str] = []

        def record_bad_line(line: list[str], _skipped: list[str] = skipped) -> None:
            # A callable on_bad_lines both drops the row AND lets us count it.
            if len(_skipped) < 3:
                _skipped.append(", ".join(map(str, line))[:120])
            else:
                _skipped.append("")

        try:
            # index_col=False stops pandas from silently promoting surplus
            # leading fields into an implicit index, which turns a ragged file
            # into a plausible-looking but wrong frame.
            df = pd.read_csv(
                StringIO(text),
                sep=separator,
                engine="python",
                index_col=False,
                on_bad_lines=record_bad_line,
            )
        except Exception as exc:
            last_error = exc
            continue

        over_long, short = audit_row_shapes(text, separator)
        df, replaced = sanitize_dataframe(df)
        report = ParseReport(
            delimiter=separator,
            rows_parsed=int(df.shape[0]),
            rows_skipped=len(skipped),
            rows_truncated=over_long,
            rows_padded=short,
            skipped_examples=[item for item in skipped[:3] if item],
            encoding=encoding,
            replaced_characters=replaced,
        )
        return df, report

    raise ValueError(f"Could not parse CSV content: {last_error}")


def sanitize_text(value: object, max_len: int = 80) -> str:
    """Make a value safe to render without discarding its language.

    The previous implementation stripped every character outside ASCII 32–126,
    which mangled or rejected any non-English dataset. This normalises to NFC
    and removes only control, format, and surrogate characters.
    """
    text = unicodedata.normalize("NFC", str(value))
    text = "".join(ch for ch in text if ch in "\t\n\r" or unicodedata.category(ch) not in _UNPRINTABLE_CATEGORIES)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len] if len(text) > max_len else text


def _count_removed(original: object, cleaned: str) -> int:
    return max(len(str(original)) - len(cleaned), 0)


def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clean column names and object cells. Returns (df, characters removed)."""
    if df is None or df.empty:
        return df, 0

    removed = 0
    clean_columns: list[str] = []
    seen: set[str] = set()
    for i, col in enumerate(df.columns):
        cleaned = sanitize_text(col, max_len=60)
        removed += _count_removed(col, cleaned)
        if not cleaned:
            cleaned = f"column_{i + 1}"
        base = cleaned
        suffix = 2
        while cleaned in seen:
            cleaned = f"{base}_{suffix}"
            suffix += 1
        seen.add(cleaned)
        clean_columns.append(cleaned)
    df.columns = clean_columns

    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        before = df[col].astype(str).str.len().sum()
        df[col] = df[col].apply(lambda v: sanitize_text(v, max_len=200) if pd.notna(v) else v)
        after = df[col].astype(str).str.len().sum()
        removed += max(int(before) - int(after), 0)

        # A column that is empty after cleaning was genuinely unreadable — not
        # merely non-English, which now survives untouched.
        blank_ratio = (df[col].astype(str).str.strip().str.len() == 0).mean()
        if blank_ratio > 0.9:
            raise ValueError(
                f"Column '{col}' is empty after removing control characters. "
                "Please verify the file encoding and upload a clean CSV."
            )
    return df, removed
