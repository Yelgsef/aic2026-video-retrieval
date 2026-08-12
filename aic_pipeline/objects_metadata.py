from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .io import atomic_write_json, atomic_write_parquet
from .paths import PipelinePaths


OBJECT_SYNONYMS = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "human face": "face",
    "mobile phone": "phone",
    "cell phone": "phone",
    "motorbike": "motorcycle",
    "land vehicle": "vehicle",
    "automobile": "car",
    "television": "tv",
    "television set": "tv",
}


OBJECT_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("frame_id", pa.int64()),
        ("pts_time", pa.float64()),
        ("keyframe_number", pa.int32()),
        ("detection_index", pa.int16()),
        ("label", pa.string()),
        ("label_normalized", pa.string()),
        ("openimages_mid", pa.string()),
        ("class_id", pa.string()),
        ("confidence", pa.float32()),
        ("ymin", pa.float32()),
        ("xmin", pa.float32()),
        ("ymax", pa.float32()),
        ("xmax", pa.float32()),
    ]
)


def normalize_object_label(label: str) -> str:
    normalized = " ".join(label.strip().lower().split())
    return OBJECT_SYNONYMS.get(normalized, normalized)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_parallel_detections(data: dict[str, Any]) -> list[dict[str, Any]]:
    scores = data.get("detection_scores", [])
    labels = data.get("detection_class_entities", [])
    mids = data.get("detection_class_names", [])
    class_ids = data.get("detection_class_labels", [])
    boxes = data.get("detection_boxes", [])
    count = min(len(scores), len(labels), len(boxes))
    detections: list[dict[str, Any]] = []
    for index in range(count):
        box = boxes[index]
        if not isinstance(box, list) or len(box) != 4:
            continue
        coordinates = [_as_float(value) for value in box]
        confidence = _as_float(scores[index])
        if confidence is None or any(value is None for value in coordinates):
            continue
        detections.append(
            {
                "detection_index": index,
                "label": str(labels[index]),
                "openimages_mid": str(mids[index]) if index < len(mids) else None,
                "class_id": str(class_ids[index]) if index < len(class_ids) else None,
                "confidence": confidence,
                "ymin": coordinates[0],
                "xmin": coordinates[1],
                "ymax": coordinates[2],
                "xmax": coordinates[3],
            }
        )
    return detections


def _flush_object_rows(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=OBJECT_SCHEMA)
    writer.write_table(table)
    rows.clear()


def build_object_corpus(
    paths: PipelinePaths,
    min_confidence: float = 0.15,
    batch_rows: int = 50_000,
    progress_every: int = 20_000,
) -> dict[str, Any]:
    started = time.time()
    paths.ensure_output_dirs()
    frame_manifest = pd.read_parquet(
        paths.manifests / "provided_keyframes.parquet",
        columns=[
            "video_id",
            "frame_id",
            "pts_time",
            "keyframe_number",
            "object_path",
            "object_exists",
        ],
    )
    object_frames = frame_manifest.loc[frame_manifest["object_exists"]].copy()
    object_frames = object_frames.sort_values(
        ["video_id", "keyframe_number"]
    ).reset_index(drop=True)

    output_dir = paths.workspace / "objects"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "corpus.parquet"
    temporary = output_dir / "corpus.parquet.tmp"

    parsed_files = 0
    failed_files = 0
    raw_detections = 0
    retained_detections = 0
    rows: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()

    writer = pq.ParquetWriter(
        temporary,
        OBJECT_SCHEMA,
        compression="zstd",
        use_dictionary=["video_id", "label", "label_normalized", "openimages_mid"],
    )
    try:
        for record in object_frames.itertuples(index=False):
            try:
                data = json.loads(Path(record.object_path).read_text(encoding="utf-8"))
                detections = _parse_parallel_detections(data)
                parsed_files += 1
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                failed_files += 1
                continue

            raw_detections += len(detections)
            for detection in detections:
                if detection["confidence"] < min_confidence:
                    continue
                normalized_label = normalize_object_label(detection["label"])
                label_counter[normalized_label] += 1
                rows.append(
                    {
                        "video_id": record.video_id,
                        "frame_id": int(record.frame_id),
                        "pts_time": float(record.pts_time),
                        "keyframe_number": int(record.keyframe_number),
                        "label_normalized": normalized_label,
                        **detection,
                    }
                )
                retained_detections += 1

            if len(rows) >= batch_rows:
                _flush_object_rows(writer, rows)
            if progress_every and parsed_files % progress_every == 0:
                print(
                    f"Parsed {parsed_files:,}/{len(object_frames):,} object files "
                    f"({retained_detections:,} retained detections)"
                )
        _flush_object_rows(writer, rows)
    finally:
        writer.close()

    os.replace(temporary, destination)
    report = {
        "pipeline_version": "0.1.0",
        "min_confidence": min_confidence,
        "object_files_expected": len(object_frames),
        "object_files_parsed": parsed_files,
        "object_files_failed": failed_files,
        "raw_detections": raw_detections,
        "retained_detections": retained_detections,
        "unique_normalized_labels": len(label_counter),
        "top_labels": dict(label_counter.most_common(30)),
        "elapsed_seconds": round(time.time() - started, 3),
        "output": str(destination),
    }
    atomic_write_json(report, paths.reports / "object_corpus.json")
    return report


def build_metadata_corpus(paths: PipelinePaths) -> dict[str, Any]:
    started = time.time()
    videos = pd.read_parquet(
        paths.manifests / "videos.parquet",
        columns=["video_id"],
    )
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for video_id in videos["video_id"]:
        metadata_path = paths.archive / "media-info" / f"{video_id}.json"
        if not metadata_path.exists():
            rows.append({"video_id": video_id, "metadata_exists": False})
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            failed.append(video_id)
            rows.append({"video_id": video_id, "metadata_exists": True})
            continue

        keywords = data.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        rows.append(
            {
                "video_id": video_id,
                "metadata_exists": True,
                "title": data.get("title"),
                "description": data.get("description"),
                "keywords": [str(value) for value in keywords],
                "keywords_text": " | ".join(str(value) for value in keywords),
                "author": data.get("author"),
                "channel_id": data.get("channel_id"),
                "publish_date": data.get("publish_date"),
                "watch_url": data.get("watch_url"),
                "youtube_length": data.get("length"),
            }
        )

    corpus = pd.DataFrame(rows)
    output_dir = paths.workspace / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "corpus.parquet"
    atomic_write_parquet(corpus, destination)
    report = {
        "pipeline_version": "0.1.0",
        "videos_total": len(corpus),
        "metadata_available": int(corpus["metadata_exists"].sum()),
        "metadata_missing": int((~corpus["metadata_exists"]).sum()),
        "metadata_failed": failed,
        "elapsed_seconds": round(time.time() - started, 3),
        "output": str(destination),
    }
    atomic_write_json(report, paths.reports / "metadata_corpus.json")
    return report
