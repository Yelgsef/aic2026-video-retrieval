from __future__ import annotations

import math
from typing import Any

import numpy as np


def adaptive_evidence_budget(duration_seconds: float, shot_count: int,
                             minimum: int = 8, maximum: int = 25) -> int:
    """Allocate more visual evidence to long or multi-shot semantic scenes."""
    duration_budget = minimum + int(round(4 * math.log1p(max(0.0, duration_seconds) / 3)))
    shot_budget = minimum + max(0, int(shot_count) - 1) * 2
    return min(maximum, max(minimum, duration_budget, shot_budget))


def adaptive_frame_ids(scene: dict[str, Any], minimum: int = 8,
                       maximum: int = 25) -> list[int]:
    start, end = int(scene["start_frame"]), int(scene["end_frame_exclusive"])
    if end <= start:
        return []
    duration = float(scene.get("end_time", 0.0)) - float(scene.get("start_time", 0.0))
    budget = min(end - start, adaptive_evidence_budget(
        duration, int(scene.get("shot_count", len(scene.get("shot_ids", [])) or 1)), minimum, maximum))
    representatives = [int(row["frame_id"]) for row in scene.get("representative_frames", [])
                       if start <= int(row["frame_id"]) < end]
    uniform = np.linspace(start, end - 1, budget).round().astype(int).tolist()
    # Preserve trusted keyframes first, then fill temporal gaps uniformly.
    selected = list(dict.fromkeys(representatives))[:budget]
    for frame_id in uniform:
        if frame_id not in selected:
            selected.append(frame_id)
        if len(selected) == budget:
            break
    if len(selected) < budget:
        for frame_id in range(start, end):
            if frame_id not in selected:
                selected.append(frame_id)
            if len(selected) == budget:
                break
    return sorted(selected)
