import pytest

from aic_pipeline.query_transformer import (LocalQueryTransformer, QueryTransformerUnavailable,
                                            StructuredQuery)
from aic_pipeline.query_understanding import build_query_plan


def test_qwen_status_reports_missing_model(tmp_path) -> None:
    transformer = LocalQueryTransformer(tmp_path, tmp_path / "index")

    status = transformer.status()

    assert status["available"] is False
    assert status["model_available"] is False
    assert status["backend"] is None


def test_requested_qwen_fails_clearly_when_model_is_missing(tmp_path) -> None:
    transformer = LocalQueryTransformer(tmp_path, tmp_path / "index")

    with pytest.raises(QueryTransformerUnavailable, match="Qwen is unavailable"):
        transformer.expand("người đi xe đạp")


def test_enhancement_corrects_clothing_translation_and_keeps_source(tmp_path, monkeypatch) -> None:
    transformer = LocalQueryTransformer(tmp_path, tmp_path / "index")
    generated = StructuredQuery(
        query_vi="Một người mặc áo xanh",
        query_en="A person wearing blue shorts",
    )
    monkeypatch.setattr(transformer, "expand", lambda _query: generated)

    plan, expansion = transformer.enhance(build_query_plan("Một người mặc áo xanh"))

    assert plan.transformer_used is True
    assert plan.variants[0] == "Một người mặc áo xanh"
    assert "blue shirt" in plan.variants[-1]
    assert expansion.query_en.endswith("blue shirt")
