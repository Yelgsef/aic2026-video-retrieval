from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENE_SCRIPT = ROOT / "scripts" / "smoke_gemini_scene.py"


def load_candidates(video_id: str, min_duration: float) -> list[dict[str, Any]]:
    shots = json.loads(
        (ROOT / "workspace" / "shots" / f"{video_id}.json").read_text(encoding="utf-8")
    )
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    row = videos.loc[videos.video_id == video_id]
    if len(row) != 1:
        raise RuntimeError(f"Missing video manifest row for {video_id}")
    fps = float(row.iloc[0].fps)
    keyframes = pd.read_parquet(
        ROOT / "workspace" / "keyframe_images" / video_id / "manifest.parquet"
    )
    counts = keyframes.groupby("shot_id").size().to_dict()

    candidates = []
    for shot_id, (start_frame, end_frame) in enumerate(shots["ranges"]):
        duration = (int(end_frame) - int(start_frame)) / fps
        keyframe_count = int(counts.get(shot_id, 0))
        if keyframe_count == 0:
            continue
        # Gemini is supplemental: skip ultra-short transition shots unless clustering
        # retained several distinct views.
        if duration < min_duration and keyframe_count < 3:
            continue
        score = math.log1p(duration) + 0.45 * math.log1p(keyframe_count)
        candidates.append(
            {
                "shot_id": shot_id,
                "duration": duration,
                "keyframe_count": keyframe_count,
                "priority": score,
            }
        )
    return candidates


def temporally_diverse_budget(
    candidates: list[dict[str, Any]], max_shots: int
) -> list[dict[str, Any]]:
    if len(candidates) <= max_shots:
        return sorted(candidates, key=lambda item: item["shot_id"])
    # Divide the timeline into bins and take the best candidate in each bin first.
    ordered = sorted(candidates, key=lambda item: item["shot_id"])
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for bin_index in range(max_shots):
        left = bin_index * len(ordered) // max_shots
        right = (bin_index + 1) * len(ordered) // max_shots
        bucket = ordered[left:right]
        if bucket:
            best = max(bucket, key=lambda item: item["priority"])
            selected.append(best)
            selected_ids.add(best["shot_id"])
    if len(selected) < max_shots:
        remaining = sorted(
            (item for item in candidates if item["shot_id"] not in selected_ids),
            key=lambda item: item["priority"],
            reverse=True,
        )
        selected.extend(remaining[: max_shots - len(selected)])
    return sorted(selected, key=lambda item: item["shot_id"])


def run_shot(video_id: str, shot_id: int, model: str, max_images: int, retries: int) -> dict:
    command = [
        sys.executable,
        str(SCENE_SCRIPT),
        "--video-id",
        video_id,
        "--shot-id",
        str(shot_id),
        "--max-images",
        str(max_images),
        "--model",
        model,
    ]
    last_error = ""
    for attempt in range(retries + 1):
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if process.returncode == 0:
            result_path = ROOT / "workspace" / "gemini_smoke" / video_id / f"shot_{shot_id:06d}.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return {"shot_id": shot_id, "status": "completed", "usage": payload.get("usage")}
        last_error = (process.stderr or process.stdout)[-2000:]
        if attempt < retries:
            time.sleep(2**attempt)
    return {"shot_id": shot_id, "status": "failed", "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted, resumable Gemini scene processing.")
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument(
        "--max-shots",
        type=int,
        default=12,
        help="Per-video Gemini budget; visual/ASR indexes still cover every shot.",
    )
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--min-duration", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = load_candidates(args.video_id, args.min_duration)
    selected = temporally_diverse_budget(candidates, args.max_shots)
    result_dir = ROOT / "workspace" / "gemini_smoke" / args.video_id
    pending = [
        item
        for item in selected
        if args.force or not (result_dir / f"shot_{item['shot_id']:06d}.json").exists()
    ]
    report: dict[str, Any] = {
        "video_id": args.video_id,
        "total_shots": len(json.loads((ROOT / "workspace" / "shots" / f"{args.video_id}.json").read_text(encoding="utf-8"))["ranges"]),
        "eligible_shots": len(candidates),
        "selected_shots": len(selected),
        "already_completed": len(selected) - len(pending),
        "pending": len(pending),
        "selection": selected,
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_shot,
                args.video_id,
                item["shot_id"],
                args.model,
                args.max_images,
                args.retries,
            ): item["shot_id"]
            for item in pending
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    report["results"] = sorted(results, key=lambda item: item["shot_id"])
    report["completed"] = sum(item["status"] == "completed" for item in results)
    report["failed"] = sum(item["status"] == "failed" for item in results)
    report["total_tokens"] = sum(
        (item.get("usage") or {}).get("total_token_count", 0)
        for item in results
    )
    report_path = result_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "completed": report["completed"], "failed": report["failed"], "total_tokens": report["total_tokens"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
