import numpy as np
import pandas as pd

from aic_pipeline.siglip_retrieval import (
    exact_search,
    fuse_variant_scores,
    l2_normalize,
    rank_with_shot_dedup,
)


def test_l2_normalize() -> None:
    result = l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(result, [[0.6, 0.8]])


def test_exact_search_orders_by_cosine() -> None:
    embeddings = l2_normalize(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    metadata = pd.DataFrame({"frame_id": [10, 20, 30]})
    result = exact_search(embeddings, metadata, np.array([1, 0]), top_k=2)
    assert result["frame_id"].tolist() == [10, 30]
    assert result["rank"].tolist() == [1, 2]


def test_max_fusion_and_shot_dedup() -> None:
    variant_scores = np.array([[0.9, 0.8, 0.1], [0.2, 0.7, 0.95]])
    scores = fuse_variant_scores(variant_scores, "max")
    metadata = pd.DataFrame(
        {"video_id": ["v", "v", "v"], "shot_id": [1, 1, 2], "frame_id": [10, 11, 20]}
    )
    result = rank_with_shot_dedup(scores, metadata, top_k=2)
    assert result["frame_id"].tolist() == [20, 10]


def test_zscore_fusion_calibrates_variants() -> None:
    scores = np.array([[100.0, 101.0, 102.0], [0.0, 0.0, 1.0]])
    fused = fuse_variant_scores(scores, "zscore_max")
    assert fused.shape == (3,)
    assert int(fused.argmax()) == 2
