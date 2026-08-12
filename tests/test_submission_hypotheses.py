from aic_pipeline.submission import build_qa_submissions, build_ranked_submissions, build_trake_submissions


def test_kis_submission_has_official_format() -> None:
    rows = build_ranked_submissions("KIS", [{"video_id": "V1", "scene_index": 1,
                                              "best_frame_id": 10,
                                              "representative_frame_ids": [10],
                                              "fusion_score": 0.9}], 1)

    assert rows[0]["format"] == "V1,10"


def test_qa_submissions_are_complete_and_scene_diverse() -> None:
    results = [
        {"video_id": "V1", "scene_index": 1, "best_frame_id": 10,
         "top_visual_frame_ids": [10, 12], "representative_frame_ids": [8]},
        {"video_id": "V2", "scene_index": 2, "best_frame_id": 20,
         "top_visual_frame_ids": [20, 22], "representative_frame_ids": [18]},
    ]
    rows = build_qa_submissions(results, ["xanh"], 4)

    assert rows[0]["format"] == "V1,10,xanh"
    assert rows[1]["format"] == "V2,20,xanh"
    assert all(row["answer"] == "xanh" for row in rows)


def test_trake_submissions_keep_order_and_exact_primary_alignment() -> None:
    events = [{"frame_id": 100}, {"frame_id": 200}, {"frame_id": 300}]
    rows = build_trake_submissions("V1", events, list(range(0, 401, 5)), 20)

    assert rows[0]["frame_ids"] == [100, 200, 300]
    assert rows[0]["format"] == "V1,100,200,300"
    assert all(a < b < c for a, b, c in (row["frame_ids"] for row in rows))
