from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .io import atomic_write_json, atomic_write_parquet
from .paths import PipelinePaths


@dataclass
class VideoAudit:
    video_id: str
    group: str
    video_path: str
    alternate_video_paths: list[str]
    video_copy_count: int
    duplicate_sizes_match: bool
    size_bytes: int
    fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    video_readable: bool = False
    has_keyframes: bool = False
    keyframe_count: int = 0
    has_map: bool = False
    map_rows: int = 0
    has_clip: bool = False
    clip_rows: int = 0
    clip_dim: int = 0
    has_objects: bool = False
    object_file_count: int = 0
    has_metadata: bool = False
    has_transcript_json: bool = False
    has_transcript_srt: bool = False
    counts_match: bool = False
    frame_ids_monotonic: bool | None = None
    frame_ids_in_video: bool | None = None
    issues: list[str] = field(default_factory=list)


def _index_directories(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_dir() and path.name not in result:
            result[path.name] = path
    return result


def _read_video_header(path: Path) -> dict[str, int | float | bool | None]:
    capture = cv2.VideoCapture(str(path))
    try:
        readable = bool(capture.isOpened())
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if readable else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if readable else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
        duration = frame_count / fps if fps > 0 else None
        return {
            "video_readable": readable,
            "fps": fps or None,
            "frame_count": frame_count or None,
            "duration_seconds": duration,
            "width": width or None,
            "height": height or None,
        }
    finally:
        capture.release()


def _read_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_int(value: str | int | float | None) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: str | int | float | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _build_frame_rows(
    video_id: str,
    mapping: list[dict[str, str]],
    image_by_number: dict[int, Path],
    object_dir: Path | None,
    clip_rows: int,
) -> Iterable[dict[str, object]]:
    for clip_row, item in enumerate(mapping):
        number = _safe_int(item.get("n"))
        frame_id = _safe_int(item.get("frame_idx"))
        pts_time = _safe_float(item.get("pts_time"))
        image_path = image_by_number.get(number) if number is not None else None
        object_path = object_dir / f"{number:03d}.json" if object_dir and number else None
        yield {
            "video_id": video_id,
            "keyframe_number": number,
            "frame_id": frame_id,
            "pts_time": pts_time,
            "fps": _safe_float(item.get("fps")),
            "image_path": str(image_path) if image_path else None,
            "image_exists": bool(image_path and image_path.exists()),
            "clip_row": clip_row if clip_row < clip_rows else None,
            "object_path": str(object_path) if object_path and object_path.exists() else None,
            "object_exists": bool(object_path and object_path.exists()),
        }


def audit_dataset(paths: PipelinePaths, progress_every: int = 50) -> dict[str, object]:
    started = time.time()
    paths.ensure_output_dirs()

    physical_video_paths = sorted((paths.archive / "video").rglob("*.mp4"))
    video_copies: dict[str, list[Path]] = {}
    for physical_path in physical_video_paths:
        video_copies.setdefault(physical_path.stem, []).append(physical_path)
    canonical_videos = [
        (video_id, sorted(copies, key=lambda path: str(path).lower()))
        for video_id, copies in sorted(video_copies.items())
    ]
    keyframe_dirs = _index_directories(paths.archive / "keyframes")
    object_dirs = _index_directories(paths.archive / "objects")

    audits: list[VideoAudit] = []
    all_frame_rows: list[dict[str, object]] = []

    for index, (video_id, copies) in enumerate(canonical_videos, start=1):
        video_path = copies[0]
        group = video_path.parent.name
        keyframe_dir = keyframe_dirs.get(video_id)
        object_dir = object_dirs.get(video_id)
        map_path = paths.archive / "map-keyframes" / f"{video_id}.csv"
        clip_path = paths.archive / "clip-features-32" / f"{video_id}.npy"
        metadata_path = paths.archive / "media-info" / f"{video_id}.json"
        transcript_json = paths.archive / "transcripts" / f"{video_id}.json"
        transcript_srt = paths.archive / "transcripts" / f"{video_id}.srt"

        audit = VideoAudit(
            video_id=video_id,
            group=group,
            video_path=str(video_path.resolve()),
            alternate_video_paths=[str(path.resolve()) for path in copies[1:]],
            video_copy_count=len(copies),
            duplicate_sizes_match=len({path.stat().st_size for path in copies}) == 1,
            size_bytes=video_path.stat().st_size,
        )
        for key, value in _read_video_header(video_path).items():
            setattr(audit, key, value)

        images: list[Path] = []
        image_by_number: dict[int, Path] = {}
        if keyframe_dir:
            images = sorted(
                keyframe_dir.glob("*.jpg"),
                key=lambda item: _safe_int(item.stem) or -1,
            )
            image_by_number = {
                number: image
                for image in images
                if (number := _safe_int(image.stem)) is not None
            }
        audit.has_keyframes = bool(images)
        audit.keyframe_count = len(images)

        mapping: list[dict[str, str]] = []
        if map_path.exists():
            audit.has_map = True
            try:
                mapping = _read_map(map_path)
                audit.map_rows = len(mapping)
            except (OSError, csv.Error, UnicodeError) as error:
                audit.issues.append(f"map_read_error:{type(error).__name__}")

        if clip_path.exists():
            audit.has_clip = True
            try:
                features = np.load(clip_path, mmap_mode="r")
                if features.ndim == 2:
                    audit.clip_rows, audit.clip_dim = map(int, features.shape)
                else:
                    audit.issues.append(f"invalid_clip_shape:{features.shape}")
                del features
            except (OSError, ValueError) as error:
                audit.issues.append(f"clip_read_error:{type(error).__name__}")

        audit.has_objects = bool(object_dir and object_dir.exists())
        audit.object_file_count = (
            sum(1 for _ in object_dir.glob("*.json")) if object_dir else 0
        )
        audit.has_metadata = metadata_path.exists()
        audit.has_transcript_json = transcript_json.exists()
        audit.has_transcript_srt = transcript_srt.exists()

        if mapping:
            frame_ids = [_safe_int(row.get("frame_idx")) for row in mapping]
            valid_frame_ids = [value for value in frame_ids if value is not None]
            audit.frame_ids_monotonic = (
                len(valid_frame_ids) == len(frame_ids)
                and all(a <= b for a, b in zip(valid_frame_ids, valid_frame_ids[1:]))
            )
            audit.frame_ids_in_video = bool(
                audit.frame_count
                and len(valid_frame_ids) == len(frame_ids)
                and all(0 <= value < audit.frame_count for value in valid_frame_ids)
            )

        audit.counts_match = bool(
            audit.has_keyframes
            and audit.has_map
            and audit.has_clip
            and audit.keyframe_count == audit.map_rows == audit.clip_rows
        )

        if not audit.video_readable:
            audit.issues.append("video_unreadable")
        if not audit.duplicate_sizes_match:
            audit.issues.append("duplicate_video_size_mismatch")
        if not audit.has_map:
            audit.issues.append("missing_map")
        if not audit.has_clip:
            audit.issues.append("missing_clip")
        if not audit.has_keyframes:
            audit.issues.append("missing_keyframes")
        if audit.has_map and audit.has_clip and audit.has_keyframes and not audit.counts_match:
            audit.issues.append("artifact_count_mismatch")
        if audit.frame_ids_monotonic is False:
            audit.issues.append("non_monotonic_frame_ids")
        if audit.frame_ids_in_video is False:
            audit.issues.append("frame_id_out_of_range")
        if audit.has_objects and audit.has_keyframes and audit.object_file_count != audit.keyframe_count:
            audit.issues.append("object_count_mismatch")

        if mapping:
            all_frame_rows.extend(
                _build_frame_rows(
                    video_id=video_id,
                    mapping=mapping,
                    image_by_number=image_by_number,
                    object_dir=object_dir,
                    clip_rows=audit.clip_rows,
                )
            )

        audits.append(audit)
        if progress_every and (index % progress_every == 0 or index == len(canonical_videos)):
            print(f"Audited {index}/{len(canonical_videos)} logical videos")

    videos_frame = pd.DataFrame(asdict(item) for item in audits)
    videos_frame["issues"] = videos_frame["issues"].map(
        lambda values: "|".join(values) if values else ""
    )
    keyframes_frame = pd.DataFrame(all_frame_rows)

    videos_path = paths.manifests / "videos.parquet"
    keyframes_path = paths.manifests / "provided_keyframes.parquet"
    atomic_write_parquet(videos_frame, videos_path)
    atomic_write_parquet(keyframes_frame, keyframes_path)

    issue_counter: Counter[str] = Counter()
    for audit in audits:
        issue_counter.update(audit.issues)

    report = {
        "pipeline_version": "0.1.0",
        "archive": str(paths.archive),
        "elapsed_seconds": round(time.time() - started, 3),
        "physical_video_files_total": len(physical_video_paths),
        "videos_total": len(audits),
        "duplicate_video_ids": sum(1 for copies in video_copies.values() if len(copies) > 1),
        "duplicate_video_files": len(physical_video_paths) - len(audits),
        "video_bytes_total": int(videos_frame["size_bytes"].sum()) if len(videos_frame) else 0,
        "provided_keyframes_total": len(keyframes_frame),
        "videos_readable": int(videos_frame["video_readable"].sum()) if len(videos_frame) else 0,
        "videos_with_complete_visual_artifacts": int(videos_frame["counts_match"].sum()) if len(videos_frame) else 0,
        "videos_with_transcript_json": int(videos_frame["has_transcript_json"].sum()) if len(videos_frame) else 0,
        "issues": dict(sorted(issue_counter.items())),
        "outputs": {
            "videos": str(videos_path),
            "provided_keyframes": str(keyframes_path),
        },
    }
    atomic_write_json(report, paths.reports / "dataset_audit.json")
    return report
