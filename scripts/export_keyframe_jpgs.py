from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_video(video_id: str) -> Path:
    videos = pd.read_parquet(PROJECT_ROOT / "workspace" / "manifests" / "videos.parquet")
    match = videos.loc[videos["video_id"] == video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one video row for {video_id}, got {len(match)}")
    return Path(match.iloc[0]["video_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export selected keyframes as JPG files.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--encoder-id", default="openclip_vit_b32_openai_quickgelu")
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    started = time.time()
    video_path = find_video(args.video_id)
    keyframe_dir = PROJECT_ROOT / "workspace" / "keyframes" / args.encoder_id
    manifest_path = keyframe_dir / f"{args.video_id}.parquet"
    keyframes = pd.read_parquet(manifest_path).sort_values("frame_id").reset_index(drop=True)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive")
        keyframes = keyframes.iloc[: args.max_images].copy()
    if keyframes.empty or keyframes["frame_id"].duplicated().any():
        raise ValueError("Keyframe manifest is empty or contains duplicate frame IDs")

    image_dir = PROJECT_ROOT / "workspace" / "keyframe_images" / args.video_id
    image_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        int(row.frame_id): image_dir / f"frame_{int(row.frame_id):09d}.jpg"
        for row in keyframes.itertuples(index=False)
    }
    last_target = max(targets)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    exported = 0
    skipped = 0
    decoded = 0
    try:
        frame_id = 0
        while frame_id <= last_target:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            destination = targets.get(frame_id)
            if destination is not None:
                if destination.exists() and not args.force:
                    skipped += 1
                else:
                    temporary = destination.with_name(destination.stem + ".tmp.jpg")
                    ok = cv2.imwrite(
                        str(temporary), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality],
                    )
                    if not ok:
                        raise RuntimeError(f"Failed to write {temporary}")
                    os.replace(temporary, destination)
                    exported += 1
                if (exported + skipped) % 500 == 0:
                    print(f"Processed {exported + skipped:,}/{len(targets):,} JPGs", flush=True)
            frame_id += 1
    finally:
        capture.release()

    missing = [str(path) for path in targets.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} JPG outputs; first: {missing[0]}")

    output_manifest = keyframes.copy()
    output_manifest["image_path"] = output_manifest["frame_id"].map(
        lambda frame_id: str(targets[int(frame_id)])
    )
    output_manifest["image_exists"] = True
    jpg_manifest = image_dir / "manifest.parquet"
    temporary_manifest = image_dir / "manifest.parquet.tmp"
    output_manifest.to_parquet(temporary_manifest, index=False)
    os.replace(temporary_manifest, jpg_manifest)

    total_bytes = sum(path.stat().st_size for path in targets.values())
    report = {
        "video_id": args.video_id,
        "video_path": str(video_path),
        "source_manifest": str(manifest_path),
        "image_directory": str(image_dir),
        "jpg_manifest": str(jpg_manifest),
        "quality": args.quality,
        "keyframe_count": len(targets),
        "exported": exported,
        "skipped_existing": skipped,
        "decoded_frames": decoded,
        "total_size_mb": round(total_bytes / 1024 / 1024, 2),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = PROJECT_ROOT / "workspace" / "reports" / f"keyframe_jpgs_{args.video_id}.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
