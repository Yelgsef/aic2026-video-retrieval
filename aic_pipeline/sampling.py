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
    coverage_anchor_interval: int = 0,
) -> list[dict[str, Any]]:
    """Return sampled frame IDs for one half-open shot range [start, end)."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    if coverage_anchor_interval < 0:
        raise ValueError("coverage_anchor_interval must be non-negative")
    if start_frame < 0 or end_frame_exclusive <= start_frame:
        raise ValueError(
            f"Invalid shot range: [{start_frame}, {end_frame_exclusive})"
        )

    reasons: dict[int, set[str]] = defaultdict(set)
    for frame_id in range(start_frame, end_frame_exclusive, stride):
        reasons[frame_id].add(f"stride_{stride}")

    # Hard global coverage guarantee: every [k*N, (k+1)*N) window contributes
    # its first frame. These anchors remain enabled even in stride-only mode.
    if coverage_anchor_interval:
        first_anchor = ((start_frame + coverage_anchor_interval - 1)
                        // coverage_anchor_interval) * coverage_anchor_interval
        for frame_id in range(first_anchor, end_frame_exclusive, coverage_anchor_interval):
            reasons[frame_id].add(f"coverage_{coverage_anchor_interval}")

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
    coverage_anchor_interval: int = 0,
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
    if not shot_data.get("ranges_end_exclusive", False):
        raise ValueError("Expected half-open OmniShotCut ranges [start, end)")

    videos = pd.read_parquet(paths.manifests / "videos.parquet")
    match = videos.loc[videos["video_id"] == video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one video manifest row for {video_id}, got {len(match)}")
    video = match.iloc[0]
    manifest_frame_count = int(video["frame_count"])
    reported_decoded = shot_data.get("decoded_frame_count")
    decoded_frame_count = int(reported_decoded if reported_decoded is not None else
                              (ranges[-1][1] if ranges else manifest_frame_count))
    if abs(decoded_frame_count - manifest_frame_count) > 1:
        raise ValueError(
            f"Decoded frame count {decoded_frame_count} differs from manifest "
            f"frame count {manifest_frame_count} by more than one frame"
        )
    # Container metadata occasionally over-reports one trailing H.264 frame.
    # OmniShotCut ranges describe frames that were actually decoded, so use
    # that count rather than manufacturing an unreadable final frame.
    frame_count = decoded_frame_count
    # A few OmniShotCut outputs contain the final frame index rather than the
    # half-open endpoint. Normalize this one-frame discrepancy.
    if ranges and int(ranges[-1][1]) == frame_count - 1:
        ranges = [list(value) for value in ranges]
        ranges[-1][1] = frame_count
    fps = float(video["fps"])

    rows: list[dict[str, Any]] = []
    for shot_id, (start_frame, end_frame) in enumerate(ranges):
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        for sample in sample_shot(
            start_frame,
            end_frame,
            stride=stride,
            boundary_radius=boundary_radius,
            include_anchors=include_anchors,
            coverage_anchor_interval=coverage_anchor_interval,
        ):
            rows.append(
                {
                    "video_id": video_id,
                    "shot_id": shot_id,
                    "shot_start_frame": start_frame,
                    "shot_end_frame_exclusive": end_frame,
                    "frame_id": sample["frame_id"],
                    "pts_time": sample["frame_id"] / fps,
                    "sampling_reason": sample["sampling_reason"],
                    "is_mandatory": sample["is_mandatory"],
                }
            )

    samples = pd.DataFrame(rows).sort_values(["frame_id", "shot_id"]).reset_index(drop=True)

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
    expected_coverage_anchors = (
        set(range(0, frame_count, coverage_anchor_interval))
        if coverage_anchor_interval else set()
    )
    sampled_frame_ids = set(samples["frame_id"].astype(int))
    missing_coverage_anchors = expected_coverage_anchors - sampled_frame_ids

    if len(invalid_ranges):
        raise ValueError(f"Found {len(invalid_ranges)} samples outside shot ranges")
    if duplicate_frames:
        raise ValueError(f"Found {duplicate_frames} duplicate frame assignments")
    if missing_shots:
        raise ValueError(f"Shots without samples: {sorted(missing_shots)}")
    if missing_coverage_anchors:
        raise ValueError(
            "Missing hard coverage anchors: "
            f"{sorted(missing_coverage_anchors)[:20]}"
        )
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
        "manifest_frame_count": manifest_frame_count,
        "decoded_frame_count": decoded_frame_count,
        "fps": fps,
        "shot_count": len(ranges),
        "stride": stride,
        "boundary_radius": boundary_radius,
        "include_anchors": include_anchors,
        "coverage_anchor_interval": coverage_anchor_interval,
        "coverage_anchor_count": len(expected_coverage_anchors),
        "missing_coverage_anchor_count": len(missing_coverage_anchors),
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
