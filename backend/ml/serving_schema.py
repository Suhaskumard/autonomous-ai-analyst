"""Does the data being predicted on still look like the data trained on?

The failure this exists to stop is quiet. A column gets renamed upstream —
`customer_age` becomes `customerAge`, `spend` becomes `spend_usd` — and the
frame arriving at the model is missing a feature it needs. The old check said
`Missing required features: ['spend']` and stopped there, which is correct and
almost useless: the caller is looking at a file that plainly has a spend column
in it, and the message does not say that the one it has is spelled differently.

Worse is the case where nothing raises at all. A column that arrives as text
because one row had `"1,234"` in it goes into a numeric pipeline, gets imputed
to the training median, and produces a confident prediction from a value nobody
supplied. That is the silent-failure shape this whole project was written about,
moved from training time to serving time: a number that looks like an answer and
is not.

So this checks three things and names all of them:

* **Missing columns**, with a suggested source for each where one is plausible.
  Suggestions come from normalising case, spaces, underscores and dashes first —
  which catches the overwhelming majority of real renames — and only then from
  fuzzy matching, at a threshold high enough that it stays quiet rather than
  guessing.
* **Type breakage**: a feature that was numeric during training and is arriving
  as something that will not convert. Reported with an example of the offending
  value, because "column X is not numeric any more" is a sentence someone has to
  go and investigate, and `"1,234"` is the investigation already done.
* **Unexpected columns**, listed but never fatal. Predicting on a file that
  still has the target in it, or extra ones the model does not use, is normal
  and the pipeline ignores them.

Nothing here imputes, coerces, or repairs. The point is to refuse clearly.
"""

import difflib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: How close a fuzzy match must be before it is offered as a suggestion. High
#: on purpose: a wrong suggestion sends someone to rename the wrong column, and
#: no suggestion at all still leaves them a correct error message.
FUZZY_CUTOFF = 0.8

#: Proportion of non-null values that must fail numeric conversion before the
#: column is called broken rather than dirty. A couple of bad cells in a
#: thousand rows is what the training pipeline's own imputation is for; a third
#: of the column is a different kind of column arriving.
TYPE_BREAK_RATE = 0.25


@dataclass
class SchemaCheck:
    """What is wrong with the frame, in a shape both an API and a human can read."""

    missing: list[str] = field(default_factory=list)
    #: {missing column: the column it was probably renamed from}
    suggestions: dict[str, str] = field(default_factory=dict)
    type_mismatches: list[dict[str, Any]] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Unexpected columns are not a failure; the other two are."""
        return not self.missing and not self.type_mismatches

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "missing": self.missing,
            "renamed": self.suggestions,
            "type_mismatches": self.type_mismatches,
            "unexpected": self.unexpected,
            "message": self.message(),
        }

    def message(self) -> str:
        """One sentence per problem, each naming the fix."""
        if self.ok:
            return "The supplied columns match what this model was trained on."

        parts: list[str] = []
        if self.missing:
            renamed = [column for column in self.missing if column in self.suggestions]
            absent = [column for column in self.missing if column not in self.suggestions]
            if renamed:
                pairs = ", ".join(f"{self.suggestions[column]!r} → {column!r}" for column in renamed)
                parts.append(
                    f"{len(renamed)} required column(s) look renamed rather than absent — rename {pairs} and try again."
                )
            if absent:
                parts.append(f"Missing required column(s) with no obvious substitute: {', '.join(absent)}.")
        for mismatch in self.type_mismatches:
            parts.append(
                f"Column {mismatch['column']!r} was numeric during training but {mismatch['bad_rate']:.0%} of "
                f"its values will not convert (for example {mismatch['example']!r}). Predicting on it would "
                "silently substitute the training median and return a confident answer for a value you did "
                "not supply."
            )
        return " ".join(parts)


def check(frame: pd.DataFrame, expected: list[str], numeric_columns: set[str] | None = None) -> SchemaCheck:
    """Compare a frame about to be predicted on with the model's expectations.

    `numeric_columns` is what the training data held as numeric — taken from the
    run's stored profile, because the fitted pipeline knows which transformer a
    column goes to but not, portably, what dtype it arrived as.
    """
    present = list(frame.columns)
    missing = [column for column in expected if column not in present]
    unexpected = [column for column in present if column not in expected]

    result = SchemaCheck(
        missing=missing,
        suggestions=_suggest_renames(missing, unexpected),
        unexpected=unexpected,
    )

    for column in expected:
        if column in missing or not (numeric_columns and column in numeric_columns):
            continue
        mismatch = _numeric_breakage(column, frame[column])
        if mismatch is not None:
            result.type_mismatches.append(mismatch)

    return result


def _normalise(name: str) -> str:
    """Case, spaces, underscores and dashes removed — the usual rename axes."""
    return "".join(character for character in str(name).lower() if character.isalnum())


def _suggest_renames(missing: list[str], candidates: list[str]) -> dict[str, str]:
    """Pair each missing column with the unexpected one it probably came from.

    Exact normalised matches first and unconditionally: `customerAge` and
    `customer_age` are the same column by any reading, and a fuzzy score should
    not get a vote on that. Only what is left goes to `difflib`, and each
    candidate is claimed at most once — one source column cannot be the answer
    to two different missing ones.
    """
    suggestions: dict[str, str] = {}
    available = list(candidates)

    by_normalised = {}
    for candidate in available:
        by_normalised.setdefault(_normalise(candidate), candidate)

    for column in missing:
        match = by_normalised.get(_normalise(column))
        if match is not None and match in available:
            suggestions[column] = match
            available.remove(match)

    for column in missing:
        if column in suggestions or not available:
            continue
        close = difflib.get_close_matches(column, available, n=1, cutoff=FUZZY_CUTOFF)
        if close:
            suggestions[column] = close[0]
            available.remove(close[0])

    return suggestions


def _numeric_breakage(column: str, series: pd.Series) -> dict[str, Any] | None:
    """A column that was numeric and is not arriving that way any more."""
    if pd.api.types.is_numeric_dtype(series):
        return None

    supplied = series.dropna()
    if supplied.empty:
        return None

    converted = pd.to_numeric(supplied, errors="coerce")
    bad = converted.isna()
    bad_rate = float(bad.mean())
    if bad_rate < TYPE_BREAK_RATE:
        return None

    return {
        "column": column,
        "expected": "numeric",
        "bad_rate": bad_rate,
        "example": str(supplied[bad].iloc[0]),
    }


def numeric_columns_from_metadata(metadata: dict) -> set[str]:
    """Which features the training data held as numbers, from the stored profile."""
    stats = metadata.get("summary_stats") or {}
    features = set(metadata.get("features") or [])
    return {
        column
        for column, entry in stats.items()
        if isinstance(entry, dict) and entry.get("type") == "numeric" and (not features or column in features)
    }
