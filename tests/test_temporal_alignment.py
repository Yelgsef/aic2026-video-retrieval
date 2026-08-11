from __future__ import annotations

import numpy as np

from aic_pipeline.temporal_alignment import TemporalAligner


def test_single_scale_fusion_matches_original_formula() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    times = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    groups = [np.array([[1.0, 0.0, 0.0]], dtype=np.float32)]
    frame_scores = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pooled = TemporalAligner._short_clip_embeddings(embeddings, times, 2.0)
    expected = 0.55 * frame_scores + 0.45 * (pooled @ groups[0].T).T
    actual = TemporalAligner._fuse_short_clip_scores(
        frame_scores, embeddings, times, groups, (2.0,))
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_multiscale_fusion_is_finite_and_preserves_shape() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    times = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    groups = [np.array([[1.0, 0.0, 0.0]], dtype=np.float32)]
    scores = TemporalAligner._fuse_short_clip_scores(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32), embeddings, times, groups,
        (1.0, 2.0, 4.0))
    assert scores.shape == (1, 3)
    assert np.isfinite(scores).all()
