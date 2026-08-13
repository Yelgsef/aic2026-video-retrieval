from __future__ import annotations

import pandas as pd

from aic_pipeline.kis_review import build_kis_review_pack, requested_occurrence


def test_requested_occurrence_supports_vietnamese_ordinals() -> None:
    assert requested_occurrence("vỗ tay lần thứ hai") == 2
    assert requested_occurrence("chiếc xe đầu tiên cán đích") == 1
    assert requested_occurrence("củ măng cuối cùng vào giỏ") == "last"


def test_review_pack_groups_videos_and_preserves_ranking() -> None:
    results = [
        {"rank": 1, "video_id": "A", "scene_index": 2, "start_frame": 10,
         "end_frame_exclusive": 31, "start_time": 1.0, "end_time": 3.0,
         "best_frame_id": 20, "fusion_score": 0.9, "scene_context": "one"},
        {"rank": 2, "video_id": "B", "scene_index": 1, "start_frame": 0,
         "end_frame_exclusive": 11, "start_time": 0.0, "end_time": 1.0,
         "best_frame_id": 5, "fusion_score": 0.8, "scene_context": "two"},
    ]
    frames = pd.DataFrame({"video_id": ["A"] * 5 + ["B"] * 3,
                           "frame_id": [10, 15, 20, 25, 30, 0, 5, 10],
                           "pts_time": [1, 1.5, 2, 2.5, 3, 0, 0.5, 1]})
    pack = build_kis_review_pack("lần thứ hai", results, frames,
                                 maximum_videos=2, scenes_per_video=1,
                                 frames_per_scene=3)
    assert [item["video_id"] for item in pack["videos"]] == ["A", "B"]
    assert pack["requested_occurrence"] == 2
    assert any(frame["is_focus"] for frame in pack["videos"][0]["scenes"][0]["storyboard"])
