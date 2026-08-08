from aic_pipeline.sampling import sample_shot


def test_sample_shot_uses_half_open_range_and_keeps_anchors() -> None:
    rows = sample_shot(10, 21, stride=5, boundary_radius=2)
    by_frame = {row["frame_id"]: row for row in rows}

    assert min(by_frame) == 10
    assert max(by_frame) == 20
    assert 21 not in by_frame
    assert by_frame[10]["is_mandatory"]
    assert by_frame[15]["is_mandatory"]
    assert by_frame[20]["is_mandatory"]


def test_short_shot_always_has_samples_without_duplicates() -> None:
    rows = sample_shot(100, 103, stride=5, boundary_radius=2)
    frame_ids = [row["frame_id"] for row in rows]

    assert frame_ids == [100, 101, 102]
    assert len(frame_ids) == len(set(frame_ids))
