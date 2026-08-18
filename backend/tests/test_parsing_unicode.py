"""Honest parsing feedback and Unicode survival.

Regression guards for two Phase 3 items: malformed rows were dropped without
being counted, and every character outside ASCII 32–126 was stripped — which
mangled or refused any non-English dataset.
"""

import pandas as pd
import pytest

from utils.helpers import ParseReport, read_csv_with_report, sanitize_text

JAPANESE_CSV = ("名前,年齢,都市,購入\n田中太郎,34,東京,はい\n鈴木花子,28,大阪,いいえ\n佐藤健,45,名古屋,はい\n").encode()


def test_japanese_columns_and_values_survive():
    df, report = read_csv_with_report(JAPANESE_CSV)
    assert list(df.columns) == ["名前", "年齢", "都市", "購入"]
    assert df["都市"].tolist() == ["東京", "大阪", "名古屋"]
    assert report.rows_skipped == 0


@pytest.mark.parametrize(
    "text",
    ["Ünïcodé", "日本語テキスト", "Ελληνικά", "العربية", "Ĳsselmeer", "naïve café", "emoji 🙂 ok"],
)
def test_sanitize_preserves_every_script(text):
    assert sanitize_text(text) == text


def test_sanitize_still_removes_control_and_zero_width_characters():
    assert sanitize_text("a\x00b\u200bc\x1fd") == "abcd"


def test_sanitize_normalises_to_nfc():
    decomposed = "é"  # e + combining acute
    assert sanitize_text(decomposed) == "é"
    assert len(sanitize_text(decomposed)) == 1


def test_ragged_rows_are_counted_not_silently_dropped():
    """One row with too many fields, one with too few — both reported."""
    payload = b"a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n12,13\n"
    _, report = read_csv_with_report(payload)

    assert report.rows_truncated == 1, "the 5-field row must be counted"
    assert report.rows_padded == 1, "the 2-field row must be counted"
    assert report.rows_affected == 2
    message = " ".join(report.warnings)
    assert "more values than the header" in message
    assert "fewer values than the header" in message


def test_clean_file_reports_no_shape_problems(fixture_csv):
    _, report = read_csv_with_report(fixture_csv("clean_classification.csv").read_bytes())
    assert report.rows_affected == 0
    assert report.warnings == []


def test_surplus_fields_do_not_become_a_phantom_index():
    """pandas would otherwise absorb the extra fields into an implicit index,
    producing a plausible-looking frame with the values in the wrong columns."""
    df, report = read_csv_with_report(b"a,b\n1,2\n3,4,5,6\n7,8\n")

    assert list(df.columns) == ["a", "b"]
    assert df.shape == (3, 2)
    assert df["a"].tolist() == [1, 3, 7]
    assert report.rows_truncated == 1


def test_delimiter_detection_picks_the_consistent_one():
    from utils.helpers import detect_delimiter

    assert detect_delimiter("a,b,c\n1,2,3\n4,5,6\n") == ","
    assert detect_delimiter("a;b;c\n1;2;3\n4;5;6\n") == ";"
    assert detect_delimiter("a\tb\tc\n1\t2\t3\n") == "\t"


def test_semicolon_and_tab_delimiters_are_handled():
    for payload in (b"a;b;c\n1;2;3\n4;5;6\n", b"a\tb\tc\n1\t2\t3\n4\t5\t6\n"):
        df, _ = read_csv_with_report(payload)
        assert list(df.columns) == ["a", "b", "c"]
        assert df.shape == (2, 3)


def test_utf16_file_is_decoded():
    df, report = read_csv_with_report("a,b\n1,2\n3,4\n".encode("utf-16"))
    assert list(df.columns) == ["a", "b"]
    assert "utf-16" in report.encoding


def test_binary_content_is_still_rejected():
    with pytest.raises(ValueError, match="binary|corrupted"):
        read_csv_with_report(bytes(range(256)) * 40)


def test_parse_report_serialises_for_the_api():
    payload = ParseReport(delimiter=",", rows_parsed=10, rows_skipped=2, skipped_examples=["1,2,3"]).as_dict()
    assert payload["rows_skipped"] == 2
    assert payload["warnings"]


def test_japanese_dataset_trains_without_mangling(client, tmp_path):
    """Phase 3 exit criterion: a Japanese-language CSV trains intact."""
    rows = 120
    frame = pd.DataFrame(
        {
            "年齢": [20 + (i % 50) for i in range(rows)],
            "都市": [["東京", "大阪", "名古屋"][i % 3] for i in range(rows)],
            "支出": [100.0 + (i % 30) * 5 for i in range(rows)],
            "購入": ["はい" if (20 + (i % 50)) > 45 else "いいえ" for i in range(rows)],
        }
    )
    path = tmp_path / "japanese.csv"
    frame.to_csv(path, index=False, encoding="utf-8")

    from tests.conftest import upload_and_wait

    job = upload_and_wait(client, path, mode="auto", target_column="購入")

    assert job["state"] == "completed", job.get("error")
    result = job["result"]
    assert result["target"] == "購入"
    assert set(result["features"]) == {"年齢", "都市", "支出"}
    assert set(result["class_names"]) == {"はい", "いいえ"}
    # Importance labels keep their Japanese column names.
    assert {item["feature"] for item in result["feature_importance"]} <= {"年齢", "都市", "支出"}
