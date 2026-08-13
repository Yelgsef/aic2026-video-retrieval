from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def requested_occurrence(query: str) -> int | str | None:
    folded = " ".join(query.casefold().split())
    match = re.search(r"(?:lần\s+thứ|thứ)\s*(\d+)", folded)
    if match:
        return max(1, int(match.group(1)))
    words = {"một": 1, "nhất": 1, "hai": 2, "ba": 3,
             "bốn": 4, "tư": 4, "năm": 5}
    match = re.search(r"(?:lần\s+thứ|thứ)\s*(một|nhất|hai|ba|bốn|tư|năm)\b", folded)
    if match:
        return words[match.group(1)]
    if "đầu tiên" in folded or "lần đầu" in folded or "bắt đầu" in folded:
        return 1
    if "cuối cùng" in folded:
        return "last"
    return None


def _storyboard(scene: dict[str, Any], frames: pd.DataFrame,
                maximum: int) -> list[dict[str, Any]]:
    local = frames[(frames.video_id.astype(str) == str(scene["video_id"]))
                   & (frames.frame_id.astype(int) >= int(scene["start_frame"]))
                   & (frames.frame_id.astype(int) < int(scene["end_frame_exclusive"]))].sort_values("frame_id")
    if local.empty:
        return [{"frame_id": int(scene["best_frame_id"]),
                 "pts_time": float(scene["start_time"]), "is_focus": True}]
    count = min(maximum, len(local))
    positions = np.linspace(0, len(local) - 1, count).round().astype(int)
    selected = local.iloc[np.unique(positions)].copy()
    focus = int(scene["best_frame_id"])
    if not selected.frame_id.astype(int).eq(focus).any():
        nearest = local.iloc[(local.frame_id.astype(int) - focus).abs().argmin()]
        selected = pd.concat((selected, nearest.to_frame().T), ignore_index=True)
    selected = selected.sort_values("frame_id").drop_duplicates("frame_id")
    return [{"frame_id": int(row.frame_id), "pts_time": float(row.pts_time),
             "is_focus": int(row.frame_id) == focus}
            for row in selected.itertuples(index=False)]


def build_kis_review_pack(query: str, results: list[dict[str, Any]], frames: pd.DataFrame,
                          maximum_videos: int = 3, scenes_per_video: int = 3,
                          frames_per_scene: int = 9) -> dict[str, Any]:
    """Create a bounded human-verification pack without changing KIS ranking."""
    video_ids = []
    for result in results:
        video_id = str(result["video_id"])
        if video_id not in video_ids:
            video_ids.append(video_id)
        if len(video_ids) == maximum_videos:
            break
    videos = []
    for video_rank, video_id in enumerate(video_ids, 1):
        candidates = [result for result in results if str(result["video_id"]) == video_id]
        selected = candidates[:scenes_per_video]
        scenes = []
        for scene in selected:
            scenes.append({"retrieval_rank": int(scene["rank"]),
                           "scene_index": int(scene["scene_index"]),
                           "start_frame": int(scene["start_frame"]),
                           "end_frame_exclusive": int(scene["end_frame_exclusive"]),
                           "start_time": float(scene["start_time"]),
                           "end_time": float(scene["end_time"]),
                           "focus_frame_id": int(scene["best_frame_id"]),
                           "fusion_score": float(scene["fusion_score"]),
                           "context": str(scene.get("scene_context", "")),
                           "storyboard": _storyboard(scene, frames, frames_per_scene)})
        videos.append({"video_rank": video_rank, "video_id": video_id,
                       "best_retrieval_rank": int(candidates[0]["rank"]),
                       "best_score": float(candidates[0]["fusion_score"]),
                       "scenes": scenes})
    return {"task": "KIS", "requested_occurrence": requested_occurrence(query),
            "instructions": "Review scenes in temporal order; select the exact frame only after verifying the requested occurrence.",
            "videos": videos}
