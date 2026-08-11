from __future__ import annotations

import numpy as np
import pytest

from aic_pipeline.retrieval_fusion import aggregate_scene_scores


def test_visual_pooling_limits_single_frame_false_positive() -> None:
    scores = np.array([0.99, 0.10, 0.10, 0.80, 0.79, 0.78], dtype=np.float32)
    scenes = np.array([0, 0, 0, 1, 1, 1])
    maximum = aggregate_scene_scores(scores, scenes, 2, "max")
    blended = aggregate_scene_scores(scores, scenes, 2, "max_top3")
    assert maximum[0] > maximum[1]
    assert blended[0] - blended[1] < maximum[0] - maximum[1]


def test_visual_pooling_handles_short_and_empty_scenes() -> None:
    result = aggregate_scene_scores(np.array([0.5]), np.array([1]), 3, "top3_mean")
    assert np.isneginf(result[0])
    assert result[1] == pytest.approx(0.5)
    assert np.isneginf(result[2])


def test_visual_pooling_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        aggregate_scene_scores(np.array([0.5]), np.array([0]), 1, "median")
