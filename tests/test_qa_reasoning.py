from __future__ import annotations

from aic_pipeline.qa_reasoning import answers_equivalent, classify_question, temporal_frame_ids


def test_question_router_detects_specialized_reasoning() -> None:
    assert classify_question("Con cá đang bơi theo hướng nào?").kind == "motion"
    assert classify_question("Biển số xe ghi chữ và số gì?").kind == "ocr"
    assert classify_question("Người phụ nữ đang cầm vật gì?").kind == "object"
    assert classify_question("Quai mũ có màu gì?").kind == "color"
    assert classify_question("Loài động vật nào đi cùng lực lượng đang di chuyển?").kind == "object"


def test_answer_equivalence_handles_vietnamese_synonyms() -> None:
    assert answers_equivalent("Bốt bảo vệ", "Trạm bảo vệ")
    assert answers_equivalent("Con rùa biển", "Rùa biển")
    assert not answers_equivalent("màu cam", "màu xanh")


def test_temporal_frame_ids_are_ordered_and_bounded() -> None:
    values = temporal_frame_ids(100, 201, 150, 5)
    assert values == sorted(values)
    assert len(values) == 5
    assert values[0] >= 100 and values[-1] < 201
