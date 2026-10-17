from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import pandas as pd

from .io import atomic_write_json, atomic_write_parquet
from .paths import PipelinePaths


@dataclass
class VideoAudit:
    video_id: str
    video_path: str
    video_copy_count: int
    size_bytes: int
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    video_readable: bool


def _read_video_header(path: Path) -> dict[str, int | float | bool | None]:
    capture = cv2.VideoCapture(str(path))
    try:
        readable = bool(capture.isOpened())
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if readable else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
        duration = frame_count / fps if fps > 0 else None
        return {
            "video_readable": readable,
            "fps": fps or None,
            "frame_count": frame_count or None,
            "duration_seconds": duration,
        }
    finally:
        capture.release()


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
    audits: list[VideoAudit] = []

    for index, (video_id, copies) in enumerate(canonical_videos, start=1):
        video_path = copies[0]
        header = _read_video_header(video_path)
        audits.append(VideoAudit(
            video_id=video_id,
            video_path=str(video_path.resolve()),
            video_copy_count=len(copies),
            size_bytes=video_path.stat().st_size,
            fps=header["fps"],
            frame_count=header["frame_count"],
            duration_seconds=header["duration_seconds"],
            video_readable=bool(header["video_readable"]),
        ))
        if progress_every and (index % progress_every == 0 or index == len(canonical_videos)):
            print(f"Audited {index}/{len(canonical_videos)} logical videos")

    columns = list(VideoAudit.__dataclass_fields__)
    videos_frame = pd.DataFrame((asdict(item) for item in audits), columns=columns)
    videos_path = paths.manifests / "videos.parquet"
    atomic_write_parquet(videos_frame, videos_path)

    report = {
        "pipeline_version": "0.1.0",
        "archive": str(paths.archive),
        "elapsed_seconds": round(time.time() - started, 3),
        "physical_video_files_total": len(physical_video_paths),
        "videos_total": len(audits),
        "duplicate_video_ids": sum(1 for copies in video_copies.values() if len(copies) > 1),
        "duplicate_video_files": len(physical_video_paths) - len(audits),
        "video_bytes_total": int(videos_frame["size_bytes"].sum()) if len(videos_frame) else 0,
        "videos_readable": int(videos_frame["video_readable"].sum()) if len(videos_frame) else 0,
        "outputs": {"videos": str(videos_path)},
    }
    atomic_write_json(report, paths.reports / "dataset_audit.json")
    return report
