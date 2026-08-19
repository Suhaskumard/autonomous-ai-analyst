"""Has the data moved since the model was trained?

Every phase before this one judges a model at the moment it is fitted. The
holdout score, the health verdict, the calibration figure — all of them describe
a model on the day it was born, and none of them is ever revisited. A model that
was defensible in August and is quietly wrong by November fails in exactly the
way this project was written about: confidently, with a number, and with nothing
saying otherwise.

Drift is the cheapest early warning available, because it needs no labels. You
cannot know whether November's predictions were *right* until November's
outcomes arrive, which may be never. You can know that November's inputs no
longer look like August's, and that is enough to raise a hand.

## What is compared

The **reference** is the distribution of each feature at training time, written
into the run's metadata as `drift_reference`. The **live** side is the inputs of
predictions actually served, read back from the prediction log. So this measures
what the model is being asked about, not what it was trained on — which is the
only comparison that can notice the world changing.

## The measure

Population Stability Index, per column:

    PSI = Σ (live_p − ref_p) · ln(live_p / ref_p)

summed over the reference's bins. It is the symmetrised relative entropy between
two discretisations, it is the measure this field has used for decades, and its
conventional bands (0.1, 0.25) are what the thresholds here are. Bands, not a
significance test: PSI has no p-value, and pretending otherwise would dress a
heuristic up as a result. `docs/monitoring.md` says so where an operator will
read it.

Three decisions inside that are worth stating:

* **The reference's bins are the bins.** Re-binning on the live data would
  compare two different discretisations and measure the binning, not the drift.
  Live values outside the reference's range fall into the end bins, which is why
  a shifted-but-overlapping distribution registers and a wholly-new range
  saturates.
* **Zero proportions are floored, not skipped.** ln(0) is the reason naive PSI
  implementations return infinity the first time a bin empties. The floor makes
  an emptied bin a large finite contribution, which is the honest reading: it
  moved a lot, not immeasurably.
* **A category the training data never saw gets its own bucket.** PSI over a
  fixed label set cannot see a new level at all — it is simply not one of the
  bins — and a new level is one of the most common real breakages. It is
  counted, and reported separately from the number.

## What this is not

It is not an accuracy measurement, and it must never be read as one. A feature
can move a long way without the model getting worse, and a model can rot while
every input distribution holds still. What drift buys is a *question asked
early*, and the answer still needs labels.
"""

import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Reference payload format. Bumped if the shape changes, so a run trained by an
#: older pipeline can be recognised rather than misread.
REFERENCE_SCHEMA = 1

#: Bins per numeric column. Fewer than the ten the UI's histograms use, and
#: that is a finding rather than a preference: PSI's noise floor scales with
#: bin count against a fixed sample, and at ten bins a same-distribution batch
#: of a few hundred rows read as "material" drift on pure sampling noise more
#: than a third of the time in this module's own verification (`test_drift.py`
#: has the measurement). Five is where a modest, real shift — a mean moving by
#: half a standard deviation — is still caught reliably while noise stays quiet
#: at the row counts `MIN_ROWS_FOR_VERDICT` allows a verdict at all.
NUMERIC_BINS = 5

#: Categories kept explicitly; everything rarer is pooled. A long tail of
#: one-off values is noise in a stability index, and an unbounded category list
#: is an unbounded metadata payload.
MAX_CATEGORIES = 20

#: Proportion floor, so an emptied bin is a large finite PSI term instead of
#: infinity. Chosen well below the smallest proportion a 10-bin reference can
#: legitimately produce.
EPSILON = 1e-6

#: The conventional PSI bands.
MODERATE_THRESHOLD = 0.10
MATERIAL_THRESHOLD = 0.25

#: Below this many live rows the number is noise dressed as a measurement.
#:
#: 200, not the 30 an early draft of this module used and its own test suite
#: then caught: at 30 rows, comparing a reference against a fresh sample of the
#: *identical* distribution read as drift on every single trial in repeated
#: simulation, and even at 100 rows it did on roughly two draws in five. 200 is
#: where that false-positive rate reaches zero across sixty simulated trials at
#: five bins. The number is empirical, not a round default, and if `NUMERIC_
#: BINS` ever changes this should be re-measured rather than assumed to still
#: hold.
MIN_ROWS_FOR_VERDICT = 200

STABLE = "stable"
MODERATE = "moderate"
MATERIAL = "material"
INSUFFICIENT = "insufficient_data"
UNAVAILABLE = "unavailable"

#: The bucket every value the training data never saw is counted in.
UNSEEN = "__unseen__"
#: The bucket rare training categories were pooled into.
OTHER = "__other__"


# --- building the reference --------------------------------------------------


def build_reference(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """The training-time distribution of each feature, for later comparison.

    Computed over the same frame the model was fitted on and stored beside it.
    Only the feature columns: the target is not an input, and a served
    prediction has no target to compare.
    """
    profile: dict[str, Any] = {}
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        try:
            profile[column] = (
                _numeric_reference(series) if pd.api.types.is_numeric_dtype(series) else _categorical_reference(series)
            )
        except Exception:  # pragma: no cover - a column pandas cannot summarise
            logger.warning("Could not profile column %r for drift", column, exc_info=True)

    return {"schema": REFERENCE_SCHEMA, "rows": int(len(df)), "columns": profile}


def _numeric_reference(series: pd.Series) -> dict[str, Any]:
    """The reference distribution for one numeric column: quantile bins.

    Quantile edges, not equal-width ones — this is the one design choice in
    this module that is not optional. An equal-width histogram over a skewed
    column (spend, income, almost anything measured in money) puts most of the
    training mass in one or two bins and leaves the rest holding a handful of
    rows each. PSI then spends most of its signal on those thin bins, where a
    handful of live rows landing slightly differently swings the proportion by
    a large fraction — a same-distribution batch of sixty rows read as
    "material" drift on both `age` and `spend` during this feature's own
    verification, purely from bin-count noise in a long gamma tail, before this
    was changed to quantiles.

    Quantile bins start with the same mass in each of them by construction, so
    a live batch has to actually move to move the proportions much. Duplicate
    edges are collapsed (`np.unique`) rather than left in, which is what
    handles a column with repeated values at its quantile boundaries — a
    payment amount rounded to the cent, an age in whole years — where the raw
    quantile computation would otherwise propose a bin boundary that separates
    nothing.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    null_rate = float(1 - (len(values) / len(series))) if len(series) else 0.0

    if values.empty or values.nunique() <= 1:
        # A constant column has no distribution to bin. Recorded so that it
        # becoming non-constant later is still visible as a change.
        constant = float(values.iloc[0]) if not values.empty else None
        return {"kind": "numeric", "constant": constant, "edges": [], "proportions": [], "null_rate": null_rate}

    quantiles = np.linspace(0.0, 1.0, NUMERIC_BINS + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 3:
        # Fewer than two bins survived deduplication — a column dominated by
        # one repeated value. Fall back to equal-width, which at least produces
        # *a* comparison; a heavily-repeated value is rare enough outside that
        # case that optimising the common path for it is not worth it.
        edges = np.histogram_bin_edges(values, bins=min(NUMERIC_BINS, max(2, values.nunique())))

    counts, _ = np.histogram(values, bins=edges)
    total = int(counts.sum()) or 1
    return {
        "kind": "numeric",
        "edges": [float(edge) for edge in edges],
        "proportions": [float(count) / total for count in counts],
        "null_rate": null_rate,
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _categorical_reference(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(str)
    null_rate = float(1 - (len(values) / len(series))) if len(series) else 0.0
    if values.empty:
        return {"kind": "categorical", "categories": {}, "null_rate": null_rate}

    counts = values.value_counts()
    kept = counts.head(MAX_CATEGORIES)
    total = int(counts.sum())
    categories = {str(label): int(count) / total for label, count in kept.items()}
    pooled = total - int(kept.sum())
    if pooled > 0:
        categories[OTHER] = pooled / total
    return {"kind": "categorical", "categories": categories, "null_rate": null_rate}


# --- reading a reference back ------------------------------------------------


def reference_from_metadata(metadata: dict) -> dict[str, Any] | None:
    """The run's reference distribution, or None if it has none usable.

    A run trained before Phase 9 has no `drift_reference`, and re-training every
    existing run to get one is not a reasonable upgrade path. So the display
    payloads are read instead: `charts.numeric_histograms` carries real binned
    counts, and `charts.categorical_bars` carries real value counts. They are a
    *worse* reference — charts cover only the first five columns of each kind
    and only the top ten categories, because they were built to be drawn rather
    than compared — and the result says so in `derived_from`, so a thin answer
    is never mistaken for a complete one.
    """
    reference = metadata.get("drift_reference")
    if isinstance(reference, dict) and reference.get("columns"):
        return {**reference, "derived_from": "training_profile"}

    salvaged = _reference_from_charts(metadata)
    if salvaged["columns"]:
        return salvaged
    return None


def _reference_from_charts(metadata: dict) -> dict[str, Any]:
    charts = metadata.get("charts") or {}
    features = set(metadata.get("features") or [])
    columns: dict[str, Any] = {}

    for column, payload in (charts.get("numeric_histograms") or {}).items():
        if features and column not in features:
            continue  # the target is charted too, and is not an input
        edges = _edges_from_interval_labels(payload.get("labels") or [])
        counts = [float(count) for count in payload.get("counts") or []]
        if not edges or len(counts) != len(edges) - 1 or sum(counts) <= 0:
            continue
        total = sum(counts)
        columns[column] = {
            "kind": "numeric",
            "edges": edges,
            "proportions": [count / total for count in counts],
            "null_rate": 0.0,
        }

    for column, payload in (charts.get("categorical_bars") or {}).items():
        if features and column not in features:
            continue
        labels = payload.get("labels") or []
        counts = [float(count) for count in payload.get("counts") or []]
        if not labels or len(labels) != len(counts) or sum(counts) <= 0:
            continue
        total = sum(counts)
        columns[column] = {
            "kind": "categorical",
            "categories": {str(label): count / total for label, count in zip(labels, counts, strict=False)},
            "null_rate": 0.0,
        }

    return {
        "schema": REFERENCE_SCHEMA,
        "rows": int(metadata.get("row_count") or 0),
        "columns": columns,
        "derived_from": "charts",
        "note": (
            "This run predates the training-time drift profile, so its reference was recovered from the "
            "chart payloads: at most five numeric and five categorical columns, and only the ten most "
            "common values of each. Re-train the run for a complete reference."
        ),
    }


_INTERVAL = re.compile(r"[\[(]\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*[\])]")


def _edges_from_interval_labels(labels: list[str]) -> list[float]:
    """Bin edges out of pandas' `(19.94, 25.9]` interval strings.

    Parsed rather than recomputed because the counts that go with them were
    produced by exactly these cuts; deriving edges any other way would pair
    counts with boundaries they were not counted under.
    """
    edges: list[float] = []
    for index, label in enumerate(labels):
        match = _INTERVAL.match(str(label))
        if match is None:
            return []
        low, high = float(match.group(1)), float(match.group(2))
        if index == 0:
            edges.append(low)
        elif not math.isclose(low, edges[-1], rel_tol=1e-6, abs_tol=1e-9):
            return []  # not a contiguous set of cuts; refuse rather than guess
        edges.append(high)
    return edges if len(edges) >= 2 else []


# --- comparing ---------------------------------------------------------------


def compare(reference: dict[str, Any] | None, frame: pd.DataFrame, min_rows: int = MIN_ROWS_FOR_VERDICT) -> dict:
    """Score `frame` against the training-time reference.

    Returns one entry per reference column plus an overall status, which is the
    worst individual verdict — drift in one feature the model leans on is drift,
    and averaging it away with nine stable columns is how a monitor learns to
    say nothing.
    """
    if not reference or not reference.get("columns"):
        return {
            "status": UNAVAILABLE,
            "reason": (
                "This run has no reference distribution to compare against. It was trained before "
                "prediction monitoring existed and its chart payloads could not be read as one; "
                "re-train it to enable drift detection."
            ),
            "columns": [],
            "rows_compared": int(len(frame)),
        }

    rows = int(len(frame))
    columns = [_compare_column(name, spec, frame.get(name)) for name, spec in reference["columns"].items()]
    columns = [entry for entry in columns if entry is not None]
    measured = [entry for entry in columns if entry["psi"] is not None]

    if rows < min_rows:
        # Reported, not withheld: an operator watching a new model should see
        # the numbers accumulating. But the verdict stays "insufficient", so a
        # PSI from nine rows never turns into an alert.
        status = INSUFFICIENT
        reason = (
            f"{rows} prediction{'s' if rows != 1 else ''} logged; {min_rows} are needed before a "
            "distribution comparison means anything."
        )
    elif not measured:
        status = UNAVAILABLE
        reason = "None of the reference columns appeared in the logged inputs."
    else:
        status = _worst(entry["verdict"] for entry in measured)
        reason = _explain(status, measured)

    return {
        "status": status,
        "reason": reason,
        "rows_compared": rows,
        "min_rows": min_rows,
        "reference_rows": reference.get("rows"),
        "derived_from": reference.get("derived_from", "training_profile"),
        "reference_note": reference.get("note"),
        "thresholds": {"moderate": MODERATE_THRESHOLD, "material": MATERIAL_THRESHOLD},
        # Worst first: the point of the list is what to look at.
        "columns": sorted(columns, key=lambda entry: (entry["psi"] is None, -(entry["psi"] or 0.0))),
    }


def _compare_column(name: str, spec: dict, series: pd.Series | None) -> dict | None:
    if series is None:
        return {
            "column": name,
            "kind": spec.get("kind"),
            "psi": None,
            "verdict": UNAVAILABLE,
            "detail": "This column was not present in the logged inputs.",
        }

    if spec.get("kind") == "numeric":
        return _compare_numeric(name, spec, series)
    return _compare_categorical(name, spec, series)


def _compare_numeric(name: str, spec: dict, series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce")
    null_rate = float(values.isna().mean()) if len(values) else 0.0
    values = values.dropna()

    edges = spec.get("edges") or []
    if len(edges) < 2:
        # A column that was constant at training time. PSI needs bins; what is
        # meaningful instead is simply whether it is still that constant.
        constant = spec.get("constant")
        moved = bool(len(values)) and (constant is None or not np.allclose(values, constant))
        return {
            "column": name,
            "kind": "numeric",
            "psi": None,
            "verdict": MATERIAL if moved else STABLE,
            "detail": (
                f"Constant at {constant} during training and no longer constant."
                if moved
                else f"Constant at {constant} during training and still is."
            ),
            "null_rate": null_rate,
            "reference_null_rate": spec.get("null_rate"),
        }

    if values.empty:
        return {
            "column": name,
            "kind": "numeric",
            "psi": None,
            "verdict": UNAVAILABLE,
            "detail": "Every logged value for this column was missing or non-numeric.",
            "null_rate": null_rate,
            "reference_null_rate": spec.get("null_rate"),
        }

    # The reference's own edges, with the ends opened out so live values beyond
    # the training range land in the end bins instead of being dropped. A value
    # the model has never seen the like of is the most interesting kind there
    # is; silently discarding it would make the measure blindest exactly when
    # it matters.
    bins = list(edges)
    bins[0], bins[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=bins)
    total = int(counts.sum()) or 1
    live = [float(count) / total for count in counts]

    psi = _psi(spec["proportions"], live)
    return {
        "column": name,
        "kind": "numeric",
        "psi": psi,
        "verdict": _verdict(psi),
        "detail": _numeric_detail(spec, values, psi),
        "null_rate": null_rate,
        "reference_null_rate": spec.get("null_rate"),
        "reference_proportions": spec["proportions"],
        "live_proportions": live,
        "edges": edges,
        "live_mean": float(values.mean()),
        "reference_mean": spec.get("mean"),
        "out_of_range_rate": float(((values < edges[0]) | (values > edges[-1])).mean()),
    }


def _compare_categorical(name: str, spec: dict, series: pd.Series) -> dict:
    values = series.dropna().astype(str)
    null_rate = float(series.isna().mean()) if len(series) else 0.0
    reference = dict(spec.get("categories") or {})
    if not reference or values.empty:
        return {
            "column": name,
            "kind": "categorical",
            "psi": None,
            "verdict": UNAVAILABLE,
            "detail": "No categories to compare.",
            "null_rate": null_rate,
            "reference_null_rate": spec.get("null_rate"),
        }

    known = {label for label in reference if label not in {OTHER, UNSEEN}}
    counts = values.value_counts()
    total = int(counts.sum())

    live = {label: float(counts.get(label, 0)) / total for label in known}
    unseen_labels = [str(label) for label in counts.index if label not in known]
    unseen_mass = sum(float(counts[label]) for label in unseen_labels) / total

    # The training data's pooled tail and anything genuinely new share one
    # bucket, because from the reference's point of view they are the same
    # thing: mass outside the categories it named.
    keys = sorted(known) + [OTHER]
    reference_vector = [reference.get(label, 0.0) for label in sorted(known)] + [reference.get(OTHER, 0.0)]
    live_vector = [live[label] for label in sorted(known)] + [unseen_mass]

    psi = _psi(reference_vector, live_vector)
    verdict = _verdict(psi)
    detail = f"PSI {psi:.3f} across {len(known)} known categories."
    if unseen_labels:
        shown = ", ".join(repr(label) for label in unseen_labels[:5])
        detail = (
            f"{len(unseen_labels)} value(s) never seen in training ({shown}"
            f"{', …' if len(unseen_labels) > 5 else ''}) account for {unseen_mass:.1%} of rows. " + detail
        )
        # A brand-new category is a schema-shaped problem wearing drift's
        # clothes — a renamed level, a new region code, an upstream enum
        # change. Worth escalating past what its mass alone would score.
        if unseen_mass >= 0.05 and verdict == STABLE:
            verdict = MODERATE

    return {
        "column": name,
        "kind": "categorical",
        "psi": psi,
        "verdict": verdict,
        "detail": detail,
        "null_rate": null_rate,
        "reference_null_rate": spec.get("null_rate"),
        "categories": keys,
        "reference_proportions": reference_vector,
        "live_proportions": live_vector,
        "unseen_categories": unseen_labels[:20],
        "unseen_rate": unseen_mass,
    }


def _psi(reference: list[float], live: list[float]) -> float:
    """Population Stability Index between two proportion vectors."""
    total = 0.0
    for ref, obs in zip(reference, live, strict=False):
        ref = max(float(ref), EPSILON)
        obs = max(float(obs), EPSILON)
        total += (obs - ref) * math.log(obs / ref)
    return float(total)


def _verdict(psi: float) -> str:
    if psi >= MATERIAL_THRESHOLD:
        return MATERIAL
    if psi >= MODERATE_THRESHOLD:
        return MODERATE
    return STABLE


_SEVERITY = {STABLE: 0, UNAVAILABLE: 0, INSUFFICIENT: 0, MODERATE: 1, MATERIAL: 2}


def _worst(verdicts) -> str:
    return max(verdicts, key=lambda verdict: _SEVERITY.get(verdict, 0), default=STABLE)


def _numeric_detail(spec: dict, values: pd.Series, psi: float) -> str:
    reference_mean = spec.get("mean")
    if reference_mean is None:
        return f"PSI {psi:.3f}."
    live_mean = float(values.mean())
    direction = "up" if live_mean > reference_mean else "down"
    return f"PSI {psi:.3f}; mean {direction} from {reference_mean:.4g} to {live_mean:.4g}."


def _explain(status: str, columns: list[dict]) -> str:
    if status == STABLE:
        return f"All {len(columns)} comparable feature(s) are within the stable band (PSI < {MODERATE_THRESHOLD})."
    moved = [entry["column"] for entry in columns if entry["verdict"] in {MODERATE, MATERIAL}]
    shown = ", ".join(moved[:5]) + (", …" if len(moved) > 5 else "")
    if status == MATERIAL:
        return (
            f"{len(moved)} feature(s) have moved materially since training ({shown}). "
            "Predictions are still being served; this is a reason to check them, not evidence that they are wrong."
        )
    return f"{len(moved)} feature(s) have moved moderately since training ({shown})."
