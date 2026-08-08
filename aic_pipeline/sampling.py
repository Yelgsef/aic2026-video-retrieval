from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .io import atomic_write_json, atomic_write_parquet
from .paths import PipelinePaths


def _add_reason(
    reasons: dict[int, set[str]],
    frame_id: int,
    reason: str,
    start_frame: int,
    end_frame_exclusive: int,
) -> None:
    if start_frame <= frame_id < end_frame_exclusive:
        reasons[frame_id].add(reason)


def sample_shot(
    start_frame: int,
    end_frame_exclusive: int,
    stride: int = 5,
    boundary_radius: int = 2,
    include_anchors: bool = True,
) -> list[dict[str, Any]]:
    """Return sampled frame IDs for one half-open shot range [start, end)."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    if start_frame < 0 or end_frame_exclusive <= start_frame:
        raise ValueError(
            f"Invalid shot range: [{start_frame}, {end_frame_exclusive})"
        )

    reasons: dict[int, set[str]] = defaultdict(set)
    for frame_id in range(start_frame, end_frame_exclusive, stride):
        reasons[frame_id].add(f"stride_{stride}")

    if include_anchors:
        last_frame = end_frame_exclusive - 1
        middle_frame = (start_frame + last_frame) // 2
        _add_reason(reasons, start_frame, "shot_start", start_frame, end_frame_exclusive)
        _add_reason(reasons, middle_frame, "shot_middle", start_frame, end_frame_exclusive)
        _add_reason(reasons, last_frame, "shot_end", start_frame, end_frame_exclusive)

        for offset in range(boundary_radius + 1):
            _add_reason(
                reasons,
                start_frame + offset,
                "start_boundary",
                start_frame,
                end_frame_exclusive,
            )
            _add_reason(
                reasons,
                last_frame - offset,
                "end_boundary",
                start_frame,
                end_frame_exclusive,
            )

    rows: list[dict[str, Any]] = []
    for frame_id, frame_reasons in sorted(reasons.items()):
        is_mandatory = any(not value.startswith("stride_") for value in frame_reasons)
        rows.append(
            {
                "frame_id": frame_id,
                "sampling_reason": "|".join(sorted(frame_reasons)),
                "is_mandatory": is_mandatory,
            }
        )
    return rows


def build_video_sampling_manifest(
    paths: PipelinePaths,
    video_id: str,
    shot_json_path: Path | None = None,
    stride: int = 5,
    boundary_radius: int = 2,
    include_anchors: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    paths.ensure_output_dirs()
    shot_json_path = shot_json_path or paths.workspace / "shots" / f"{video_id}.json"
    destination = paths.samples / f"{video_id}.parquet"
    report_path = paths.reports / f"sampling_{video_id}.json"
    if destination.exists() and not force:
        raise FileExistsError(
            f"Sampling manifest already exists: {destination}. Use --force to rebuild."
        )

    shot_data = json.loads(shot_json_path.read_text(encoding="utf-8"))
    ranges = shot_data.get("ranges") or []
    intra_labels = shot_data.get("intra_labels") or [None] * len(ranges)
    inter_labels = shot_data.get("inter_labels") or [None] * len(ranges)
    if len(intra_labels) != len(ranges) or len(inter_labels) != len(ranges):
        raise ValueError("Shot ranges and label counts do not match")
    if not shot_data.get("ranges_end_exclusive", False):
        raise ValueError("Expected half-open OmniShotCut ranges [start, end)")

    videos = pd.read_parquet(paths.manifests / "videos.parquet")
    match = videos.loc[videos["video_id"] == video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one video manifest row for {video_id}, got {len(match)}")
    video = match.iloc[0]
    frame_count = int(video["frame_count"])
    fps = float(video["fps"])

    rows: list[dict[str, Any]] = []
    for shot_id, ((start_frame, end_frame), intra_label, inter_label) in enumerate(
        zip(ranges, intra_labels, inter_labels)
    ):
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        for sample in sample_shot(
            start_frame,
            end_frame,
            stride=stride,
            boundary_radius=boundary_radius,
            include_anchors=include_anchors,
        ):
            rows.append(
                {
                    "video_id": video_id,
                    "shot_id": shot_id,
                    "shot_start_frame": start_frame,
                    "shot_end_frame_exclusive": end_frame,
                    "shot_length": end_frame - start_frame,
                    "frame_id": sample["frame_id"],
                    "pts_time": sample["frame_id"] / fps,
                    "sampling_reason": sample["sampling_reason"],
                    "is_mandatory": sample["is_mandatory"],
                    "intra_label": intra_label,
                    "inter_label": inter_label,
                }
            )

    samples = pd.DataFrame(rows).sort_values(["frame_id", "shot_id"]).reset_index(drop=True)
    samples.insert(0, "sample_id", range(len(samples)))

    invalid_ranges = samples.loc[
        (samples["frame_id"] < samples["shot_start_frame"])
        | (samples["frame_id"] >= samples["shot_end_frame_exclusive"])
    ]
    duplicate_frames = int(samples["frame_id"].duplicated().sum())
    range_gaps = [
        int(ranges[index + 1][0]) - int(ranges[index][1])
        for index in range(len(ranges) - 1)
    ]
    missing_shots = set(range(len(ranges))) - set(samples["shot_id"].unique())
    final_end = int(ranges[-1][1]) if ranges else 0

    if len(invalid_ranges):
        raise ValueError(f"Found {len(invalid_ranges)} samples outside shot ranges")
    if duplicate_frames:
        raise ValueError(f"Found {duplicate_frames} duplicate frame assignments")
    if missing_shots:
        raise ValueError(f"Shots without samples: {sorted(missing_shots)}")
    if ranges and (int(ranges[0][0]) != 0 or final_end != frame_count):
        raise ValueError(
            f"Shot coverage [0, {final_end}) does not match frame count {frame_count}"
        )
    if any(gap != 0 for gap in range_gaps):
        raise ValueError("Shot ranges contain gaps or overlaps")

    atomic_write_parquet(samples, destination)
    reason_counter: Counter[str] = Counter()
    for value in samples["sampling_reason"]:
        reason_counter.update(value.split("|"))

    report = {
        "pipeline_version": "0.1.0",
        "video_id": video_id,
        "source_shots": str(shot_json_path.resolve()),
        "frame_count": frame_count,
        "fps": fps,
        "shot_count": len(ranges),
        "stride": stride,
        "boundary_radius": boundary_radius,
        "include_anchors": include_anchors,
        "sample_count": len(samples),
        "mandatory_sample_count": int(samples["is_mandatory"].sum()),
        "sample_ratio": len(samples) / frame_count,
        "compression_ratio": 1.0 - len(samples) / frame_count,
        "duplicate_frame_assignments": duplicate_frames,
        "invalid_range_samples": len(invalid_ranges),
        "shots_without_samples": len(missing_shots),
        "reason_counts": dict(sorted(reason_counter.items())),
        "elapsed_seconds": round(time.time() - started, 3),
        "output": str(destination),
    }
    atomic_write_json(report, report_path)
    return report
