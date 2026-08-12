from aic_pipeline.qa_gemini import local_exact_ocr_answer, qa_grounding_query


def test_time_answer_slot_is_removed_but_bib_is_preserved() -> None:
    query = ("Trong cảnh vận động viên xe đạp mang số áo 84, "
             "đồng hồ thời gian ở góc màn hình đang hiển thị bao nhiêu?")

    assert qa_grounding_query(query) == "Trong cảnh vận động viên xe đạp mang số áo 84"
