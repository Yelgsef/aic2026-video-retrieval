from __future__ import annotations

from aic_pipeline.scene_evidence import adaptive_evidence_budget, adaptive_frame_ids


def test_adaptive_budget_grows_and_is_bounded() -> None:
    assert adaptive_evidence_budget(2, 1) == 10
    assert adaptive_evidence_budget(120, 20) == 25


def test_adaptive_frames_preserve_representatives_and_bounds() -> None:
    scene = {"start_frame": 100, "end_frame_exclusive": 300, "start_time": 4.0,
             "end_time": 12.0, "shot_count": 2,
             "representative_frames": [{"frame_id": 120}, {"frame_id": 250}]}
    values = adaptive_frame_ids(scene)
    assert 120 in values and 250 in values
    assert values == sorted(values)
    assert all(100 <= value < 300 for value in values)
    assert 8 <= len(values) <= 25
