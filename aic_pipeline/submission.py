from __future__ import annotations

from typing import Any


def build_ranked_submissions(task: str, results: list[dict[str, Any]], maximum: int = 100) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    submissions, seen = [], set()
    for result in results:
        frame_id = int(result.get("best_frame_id") or result["representative_frame_ids"][0])
        key = (str(result["video_id"]), frame_id)
        if key in seen:
            continue
        seen.add(key)
        item = {"rank": len(submissions) + 1, "video_id": key[0], "frame_id": frame_id,
                "scene_index": int(result["scene_index"]), "score": float(result["fusion_score"])}
        if task == "QA":
            item["answer"] = None
            item["requires_vqa"] = True
        submissions.append(item)
        if len(submissions) >= maximum:
            break
    return submissions
