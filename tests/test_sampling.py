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


def test_global_coverage_anchor_every_67_frames() -> None:
    rows = sample_shot(50, 150, stride=5, boundary_radius=0,
                       include_anchors=False, coverage_anchor_interval=67)
    by_frame = {row["frame_id"]: row for row in rows}

    assert 67 in by_frame
    assert 134 in by_frame
    assert by_frame[67]["is_mandatory"]
    assert "coverage_67" in by_frame[67]["sampling_reason"]


def test_coverage_anchor_is_global_across_shot_boundaries() -> None:
    first = sample_shot(0, 60, stride=100, include_anchors=False,
                        coverage_anchor_interval=67)
    second = sample_shot(60, 100, stride=100, include_anchors=False,
                         coverage_anchor_interval=67)

    assert [row["frame_id"] for row in first] == [0]
    assert [row["frame_id"] for row in second] == [60, 67]
    assert "coverage_67" in second[-1]["sampling_reason"]
