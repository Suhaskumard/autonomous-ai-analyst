"""Phase 9: does the model refuse clearly when serving data has drifted in shape?

`ml/serving_schema.py` replaces `Missing required features: [...]` with a
message that names what actually happened — a rename, a genuine absence, or a
column that silently stopped being numeric. These tests are about the message
matching the failure, not about pandas.
"""

import pandas as pd

from ml import serving_schema as ss

EXPECTED = ["customer_age", "spend", "region"]
NUMERIC = {"customer_age", "spend"}


def test_a_matching_frame_is_ok():
    frame = pd.DataFrame({"customer_age": [30], "spend": [1.0], "region": ["n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.ok
    assert result.message() == "The supplied columns match what this model was trained on."


def test_extra_columns_are_reported_but_not_fatal():
    frame = pd.DataFrame({"customer_age": [30], "spend": [1.0], "region": ["n"], "churn": ["yes"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.ok
    assert result.unexpected == ["churn"]


# --- renames --------------------------------------------------------------


def test_a_case_rename_is_suggested():
    frame = pd.DataFrame({"customerAge": [30], "spend": [1.0], "region": ["n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert not result.ok
    assert result.suggestions == {"customer_age": "customerAge"}
    assert "customerAge" in result.message() and "customer_age" in result.message()


def test_an_underscore_to_dash_rename_is_suggested():
    frame = pd.DataFrame({"customer-age": [30], "spend": [1.0], "region": ["n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.suggestions == {"customer_age": "customer-age"}


def test_a_close_typo_is_suggested_by_fuzzy_match():
    frame = pd.DataFrame({"customer_agee": [30], "spend": [1.0], "region": ["n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.suggestions.get("customer_age") == "customer_agee"


def test_a_genuinely_absent_column_gets_no_suggestion():
    frame = pd.DataFrame({"spend": [1.0], "region": ["n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.missing == ["customer_age"]
    assert "customer_age" not in result.suggestions
    assert "no obvious substitute" in result.message()


def test_a_wildly_different_name_is_not_offered_as_a_guess():
    """A wrong suggestion sends someone to rename the wrong column."""
    frame = pd.DataFrame({"z9_totally_unrelated": [30], "spend": [1.0], "region": ["n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert "customer_age" not in result.suggestions


def test_each_candidate_is_claimed_at_most_once():
    """One source column cannot be the answer to two different missing ones."""
    frame = pd.DataFrame({"cust_age": [30], "region": ["n"]})  # spend genuinely absent
    result = ss.check(frame, ["customer_age", "cust_age_2", "spend"], {"customer_age", "cust_age_2", "spend"})
    claimed = list(result.suggestions.values())
    assert len(claimed) == len(set(claimed))


# --- type breakage ----------------------------------------------------------


def test_a_column_that_will_not_convert_is_flagged():
    frame = pd.DataFrame(
        {
            "customer_age": [30, 31, 32, 33],
            "spend": ["1,234", "2,345", "n/a", "3,456"],
            "region": ["n"] * 4,
        }
    )
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert not result.ok
    assert result.type_mismatches[0]["column"] == "spend"
    assert result.type_mismatches[0]["example"] == "1,234"
    assert "silently substitute" in result.message()


def test_a_few_dirty_cells_do_not_trip_the_type_check():
    """A couple of bad cells is what training-time imputation is for."""
    frame = pd.DataFrame(
        {
            "customer_age": [30] * 10,
            "spend": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "x"],
            "region": ["n"] * 10,
        }
    )
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.ok


def test_an_already_numeric_column_is_never_flagged():
    frame = pd.DataFrame({"customer_age": [30.0], "spend": [1.0], "region": ["n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.ok


def test_a_missing_column_is_not_also_reported_as_type_broken():
    frame = pd.DataFrame({"spend": ["not-a-number"] * 4, "region": ["n"] * 4})
    result = ss.check(frame, EXPECTED, NUMERIC)
    # customer_age missing entirely; spend's own breakage still gets caught.
    assert "customer_age" in result.missing
    assert any(m["column"] == "spend" for m in result.type_mismatches)


def test_an_all_null_column_is_not_flagged_as_type_broken():
    """Nothing to convert is not the same failure as something that fails to."""
    frame = pd.DataFrame({"customer_age": [None, None], "spend": [1.0, 2.0], "region": ["n", "n"]})
    result = ss.check(frame, EXPECTED, NUMERIC)
    assert result.type_mismatches == []


# --- numeric_columns_from_metadata -------------------------------------------


def test_numeric_columns_read_from_summary_stats():
    metadata = {
        "features": ["age", "region"],
        "summary_stats": {
            "age": {"type": "numeric"},
            "region": {"type": "categorical"},
            "churn": {"type": "categorical"},  # the target; not a feature
        },
    }
    assert ss.numeric_columns_from_metadata(metadata) == {"age"}


def test_numeric_columns_is_empty_for_a_run_with_no_profile():
    assert ss.numeric_columns_from_metadata({}) == set()


# --- as_dict shape for the API ------------------------------------------------


def test_as_dict_has_the_shape_the_route_returns():
    frame = pd.DataFrame({"customerAge": [30], "spend": [1.0], "region": ["n"]})
    payload = ss.check(frame, EXPECTED, NUMERIC).as_dict()
    assert set(payload) == {"ok", "missing", "renamed", "type_mismatches", "unexpected", "message"}
    assert payload["ok"] is False
