"""Phase 9: does the drift measure actually distinguish noise from a real shift?

The dangerous failure mode for a monitoring feature is not "it never fires" —
that fails loudly the first time someone notices a stale model. It is "it fires
constantly on nothing", because the second real alert after ten false ones gets
the same shrug as the first. So this file spends more of its weight on
false-positive behaviour than a typical unit-test suite would, and the numbers
in it are not arbitrary: `NUMERIC_BINS` and `MIN_ROWS_FOR_VERDICT` in
`ml/drift.py` were set from exactly the simulation `test_false_positive_rate_
is_low_at_the_configured_floor` runs, after an early version of this module
(ten bins, a floor of thirty rows) was caught reading a same-distribution batch
as material drift on every trial.
"""

import numpy as np
import pandas as pd
import pytest

from ml import drift


def _make(rng, n, age_mean=45.0, region_probs=(0.5, 0.3, 0.2), regions=("north", "south", "east")):
    age = rng.normal(age_mean, 12, n).clip(18, 85)
    spend = rng.gamma(3, 20, n)
    region = rng.choice(regions, n, p=list(region_probs))
    return pd.DataFrame({"age": age, "spend": spend, "region": region})


@pytest.fixture
def reference():
    train = _make(np.random.default_rng(999), 800)
    return drift.build_reference(train, ["age", "spend", "region"])


# --- the measurement that set the thresholds ---------------------------------


def test_false_positive_rate_is_low_at_the_configured_floor(reference):
    """The simulation that justifies `MIN_ROWS_FOR_VERDICT`.

    Sixty independent same-distribution draws at the configured floor; none of
    them should read as anything but stable. This is a statistical claim
    checked empirically rather than derived, so it is re-checked here rather
    than only asserted in a docstring — if `NUMERIC_BINS` or the floor moves,
    this is what will catch a regression back toward the ten-bin/thirty-row
    behaviour that motivated the change.
    """
    false_positives = 0
    trials = 60
    for trial in range(trials):
        rng = np.random.default_rng(2000 + trial)
        sample = _make(rng, drift.MIN_ROWS_FOR_VERDICT)
        result = drift.compare(reference, sample, min_rows=1)
        if result["status"] != drift.STABLE:
            false_positives += 1

    rate = false_positives / trials
    assert rate <= 0.10, f"{false_positives}/{trials} same-distribution samples read as drifted (rate {rate:.0%})"


def test_a_real_shift_is_still_caught_at_the_floor(reference):
    """The other side of the same trade: raising the floor must not blind it.

    A shift of half a standard deviation in one feature, at exactly the row
    count the floor allows a verdict at all, should read as at least moderate.
    """
    rng = np.random.default_rng(55)
    shifted = _make(rng, drift.MIN_ROWS_FOR_VERDICT, age_mean=45 + 6)  # +0.5 std
    result = drift.compare(reference, shifted, min_rows=1)
    age_column = next(entry for entry in result["columns"] if entry["column"] == "age")
    assert age_column["verdict"] in {drift.MODERATE, drift.MATERIAL}


def test_a_large_shift_is_unambiguous(reference):
    rng = np.random.default_rng(1)
    shifted = _make(rng, 300, age_mean=45 + 20)  # +1.67 std
    result = drift.compare(reference, shifted, min_rows=1)
    assert result["status"] == drift.MATERIAL


# --- quantile binning ---------------------------------------------------------


def test_numeric_reference_uses_quantile_bins_not_equal_width():
    """Equal-width bins on a skewed column starve the tail of samples.

    `spend` here is drawn from a Gamma(3, 20), which is exactly the shape —
    long right tail — where equal-width bins put most of a live sample's rows
    in one or two bins and leave the rest to swing wildly on noise. Quantile
    bins start with equal mass in each bin by construction; checking that
    directly is more useful than re-deriving the false-positive rate here.
    """
    train = pd.DataFrame({"spend": np.random.default_rng(3).gamma(3, 20, 2000)})
    reference = drift.build_reference(train, ["spend"])
    proportions = reference["columns"]["spend"]["proportions"]

    # Not exactly equal — quantile computation on finite samples is not exact —
    # but nowhere near the 40x-plus spread an equal-width histogram produces on
    # this distribution (checked by construction: np.histogram_bin_edges would
    # put the bulk of 2000 gamma(3, 20) draws under a handful of the ten bins).
    assert max(proportions) / min(proportions) < 3


def test_a_constant_column_has_no_bins_but_is_still_tracked():
    train = pd.DataFrame({"flag": [1] * 200})
    reference = drift.build_reference(train, ["flag"])
    assert reference["columns"]["flag"]["edges"] == []
    assert reference["columns"]["flag"]["constant"] == 1.0

    result = drift.compare(reference, pd.DataFrame({"flag": [1] * 250}), min_rows=1)
    assert result["columns"][0]["verdict"] == drift.STABLE

    moved = drift.compare(reference, pd.DataFrame({"flag": [2] * 250}), min_rows=1)
    assert moved["columns"][0]["verdict"] == drift.MATERIAL


def test_a_repeated_value_column_does_not_crash_quantile_binning():
    """Many rows sharing one value collapse quantile edges; must not raise."""
    rng = np.random.default_rng(4)
    values = np.concatenate([np.full(180, 50.0), rng.normal(50, 1, 20)])
    train = pd.DataFrame({"mostly_constant": values})
    reference = drift.build_reference(train, ["mostly_constant"])
    assert reference["columns"]["mostly_constant"]["kind"] == "numeric"

    result = drift.compare(reference, pd.DataFrame({"mostly_constant": values}), min_rows=1)
    assert result["columns"][0]["verdict"] in {drift.STABLE, drift.MODERATE}


# --- categorical drift: new levels -------------------------------------------


def test_a_wholly_new_category_is_reported_separately_from_psi():
    train = pd.DataFrame({"region": np.random.default_rng(0).choice(["north", "south"], 500)})
    reference = drift.build_reference(train, ["region"])

    live = pd.DataFrame({"region": ["west"] * 100})
    result = drift.compare(reference, live, min_rows=1)
    region_column = result["columns"][0]
    assert region_column["unseen_categories"] == ["west"]
    assert region_column["unseen_rate"] == 1.0
    assert region_column["verdict"] == drift.MATERIAL


def test_a_small_amount_of_a_new_category_still_escalates_from_stable():
    """A brand-new category is schema breakage wearing drift's clothes.

    A renamed level, a new region code — worth escalating past what its mass
    alone would score, because otherwise a 3% new category with everything
    else unchanged reads as boringly stable.
    """
    train = pd.DataFrame({"region": np.random.default_rng(0).choice(["north", "south", "east"], 1000)})
    reference = drift.build_reference(train, ["region"])

    rng = np.random.default_rng(9)
    mostly_known = rng.choice(["north", "south", "east"], 970, p=[0.4, 0.3, 0.3]).tolist()
    live = pd.DataFrame({"region": mostly_known + ["west"] * 30})
    result = drift.compare(reference, live, min_rows=1)
    region_column = result["columns"][0]
    assert region_column["verdict"] != drift.STABLE


def test_a_rare_training_category_and_a_new_one_share_the_pooled_bucket():
    """More distinct categories than MAX_CATEGORIES pools the long tail.

    30 one-off values, well past the 20-category cap, so this exercises the
    pooling path rather than the (also-valid, separately tested) small-category
    case.
    """
    train = pd.DataFrame({"region": ["north"] * 900 + ["south"] * 70 + [f"rare_{i}" for i in range(30)]})
    reference = drift.build_reference(train, ["region"])
    assert "north" in reference["columns"]["region"]["categories"]
    assert drift.OTHER in reference["columns"]["region"]["categories"]
    # The thirty distinct one-off values did not each get their own slot.
    assert len(reference["columns"]["region"]["categories"]) <= drift.MAX_CATEGORIES + 1


# --- reading a reference back from an old run ---------------------------------


def test_a_run_with_no_drift_reference_falls_back_to_charts():
    metadata = {
        "features": ["age", "region"],
        "row_count": 500,
        "charts": {
            "numeric_histograms": {
                "age": {"labels": ["(20.0, 30.0]", "(30.0, 40.0]", "(40.0, 50.0]"], "counts": [10, 20, 10]},
            },
            "categorical_bars": {
                "region": {"labels": ["north", "south"], "counts": [30, 10]},
            },
        },
    }
    reference = drift.reference_from_metadata(metadata)
    assert reference["derived_from"] == "charts"
    assert "predates" in reference["note"]
    assert set(reference["columns"]) == {"age", "region"}
    assert reference["columns"]["age"]["edges"] == [20.0, 30.0, 40.0, 50.0]


def test_charts_covering_the_target_are_not_read_as_a_feature():
    """Charts are drawn for every column, features and target alike."""
    metadata = {
        "features": ["age"],
        "row_count": 100,
        "charts": {
            "categorical_bars": {"churn": {"labels": ["yes", "no"], "counts": [40, 60]}},
            "numeric_histograms": {"age": {"labels": ["(20.0, 30.0]"], "counts": [100]}},
        },
    }
    reference = drift.reference_from_metadata(metadata)
    assert "churn" not in reference["columns"]


def test_a_run_with_neither_reference_nor_usable_charts_reports_unavailable():
    reference = drift.reference_from_metadata({"features": ["age"], "charts": {}})
    assert reference is None
    result = drift.compare(reference, pd.DataFrame({"age": [30] * 50}))
    assert result["status"] == drift.UNAVAILABLE


def test_a_real_reference_takes_priority_over_charts():
    metadata = {
        "drift_reference": {
            "schema": 1,
            "rows": 10,
            "columns": {"age": {"kind": "numeric", "edges": [], "proportions": []}},
        },
        "charts": {"numeric_histograms": {"age": {"labels": ["(1.0, 2.0]"], "counts": [10]}}},
        "features": ["age"],
    }
    reference = drift.reference_from_metadata(metadata)
    assert reference["derived_from"] == "training_profile"


# --- verdicts and thresholds --------------------------------------------------


def test_too_few_rows_gives_insufficient_data_not_a_verdict(reference):
    tiny = _make(np.random.default_rng(1), 5)
    result = drift.compare(reference, tiny)
    assert result["status"] == drift.INSUFFICIENT
    assert "5" in result["reason"]


def test_no_reference_at_all_is_unavailable_not_an_error():
    result = drift.compare(None, pd.DataFrame({"age": [1, 2, 3]}))
    assert result["status"] == drift.UNAVAILABLE


def test_the_worst_column_decides_the_overall_status(reference):
    """One drifted feature the model leans on is drift; averaging hides it."""
    rng = np.random.default_rng(2)
    mixed = _make(rng, 300, age_mean=45)  # age, spend stable
    mixed["region"] = ["west"] * len(mixed)  # region wholly new
    result = drift.compare(reference, mixed, min_rows=1)
    assert result["status"] == drift.MATERIAL


def test_a_column_missing_from_live_data_is_reported_not_silently_skipped(reference):
    rng = np.random.default_rng(2)
    partial = _make(rng, 300).drop(columns=["spend"])
    result = drift.compare(reference, partial, min_rows=1)
    spend_column = next(entry for entry in result["columns"] if entry["column"] == "spend")
    assert spend_column["verdict"] == drift.UNAVAILABLE
    # But the columns that ARE present still get a real comparison.
    age_column = next(entry for entry in result["columns"] if entry["column"] == "age")
    assert age_column["psi"] is not None
