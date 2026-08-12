from aic_pipeline.ocr_filter import filter_ocr_lines


def test_sports_bib_number_is_searchable() -> None:
    searchable, rejected = filter_ocr_lines([{"text": "84"}])

    assert searchable == ["84"]
    assert rejected == []


def test_timestamp_and_punctuation_remain_filtered() -> None:
    searchable, rejected = filter_ocr_lines([{"text": "19:30"}, {"text": "-"}])

    assert searchable == []
    assert {item["reason"] for item in rejected} == {
        "timestamp_or_date", "non_semantic_short_text"
    }
