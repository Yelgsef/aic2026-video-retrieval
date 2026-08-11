from __future__ import annotations

from aic_pipeline.fusion_profiles import accepted_profile, classify_retrieval_query, retrieval_profile


def test_retrieval_profile_router() -> None:
    assert classify_retrieval_query("ba người đứng bên trái") == "count"
    assert classify_retrieval_query("người phụ nữ đứng bên phải") == "spatial"
    assert classify_retrieval_query("một con chim đang bay") == "count"
    assert classify_retrieval_query("cảnh hoàng hôn trên biển") == "semantic"
    assert classify_retrieval_query("biển báo ghi gì", "ocr") == "ocr"


def test_profile_weights_are_normalized() -> None:
    profile = retrieval_profile("hai chiếc xe bên trái")
    assert profile.kind == "count"
    assert abs(sum(profile.weights.values()) - 1.0) < 1e-9


def test_only_cross_validated_profile_is_enabled() -> None:
    assert accepted_profile("QA", "Trong cảnh có bao nhiêu người?") is not None
    assert accepted_profile("KIS", "ba người đứng bên trái") is None
