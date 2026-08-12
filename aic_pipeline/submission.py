from __future__ import annotations

from typing import Any

import numpy as np


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
                "scene_index": int(result["scene_index"]), "score": float(result["fusion_score"]),
                "format": f"{key[0]},{frame_id}"}
        if task == "QA":
            item["answer"] = None
            item["requires_vqa"] = True
        submissions.append(item)
        if len(submissions) >= maximum:
            break
    return submissions


def build_qa_submissions(results: list[dict[str, Any]], answers: list[str],
                         maximum: int = 100) -> list[dict[str, Any]]:
    """Rank complete <video, frame, answer> hypotheses with scene diversity."""
    clean_answers = list(dict.fromkeys(str(value).strip() for value in answers if str(value).strip()))
    if not clean_answers or maximum <= 0:
        return []
    candidates: list[tuple[int, int, dict[str, Any], int]] = []
    for scene_rank, result in enumerate(results, 1):
        frame_ids = [int(result["best_frame_id"])]
        frame_ids.extend(int(value) for value in result.get("top_visual_frame_ids", []))
        frame_ids.extend(int(value) for value in result.get("representative_frame_ids", []))
        for frame_rank, frame_id in enumerate(dict.fromkeys(frame_ids)):
            candidates.append((frame_rank, scene_rank, result, frame_id))
    # First cover different scenes, then spend deeper ranks on nearby frames.
    candidates.sort(key=lambda value: (value[0], value[1]))
    output, seen = [], set()
    for frame_rank, scene_rank, result, frame_id in candidates:
        for answer_rank, answer in enumerate(clean_answers):
            key = (str(result["video_id"]), frame_id, answer.casefold())
            if key in seen:
                continue
            seen.add(key)
            output.append({"rank": len(output) + 1, "video_id": key[0],
                           "frame_id": frame_id, "answer": answer,
                           "scene_index": int(result["scene_index"]),
                           "source_scene_rank": scene_rank,
                           "source_frame_rank": frame_rank + 1,
                           "answer_variant_rank": answer_rank + 1,
                           "format": f"{key[0]},{frame_id},{answer}"})
            if len(output) >= maximum:
                return output
    return output


def build_trake_submissions(video_id: str, events: list[dict[str, Any]],
                            sampled_frame_ids: list[int], maximum: int = 100,
                            neighbor_radius: int = 3) -> list[dict[str, Any]]:
    """Build ordered sequence hypotheses around the aligned sampled peaks."""
    if not events or maximum <= 0:
        return []
    available = np.asarray(sorted(set(int(value) for value in sampled_frame_ids)), dtype=np.int64)
    if not len(available):
        return []
    groups: list[list[int]] = []
    for event in events:
        target = int(event["frame_id"])
        center = int(np.argmin(np.abs(available - target)))
        offsets = [0]
        for radius in range(1, neighbor_radius + 1):
            offsets.extend((-radius, radius))
        groups.append([int(available[position]) for position in
                       (center + np.asarray(offsets, dtype=np.int64))
                       if 0 <= position < len(available)])
    beam: list[tuple[float, list[int]]] = [(0.0, [])]
    for group in groups:
        expanded: list[tuple[float, list[int]]] = []
        center = group[0]
        for cost, sequence in beam:
            for frame_id in group:
                if sequence and frame_id <= sequence[-1]:
                    continue
                expanded.append((cost + abs(frame_id - center), sequence + [frame_id]))
        expanded.sort(key=lambda value: (value[0], value[1]))
        beam = expanded[:max(maximum * 4, 100)]
    output = []
    for cost, sequence in beam[:maximum]:
        output.append({"rank": len(output) + 1, "video_id": str(video_id),
                       "frame_ids": sequence, "offset_cost": float(cost),
                       "format": ",".join([str(video_id), *(str(value) for value in sequence)])})
    return output
