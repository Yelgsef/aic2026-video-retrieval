from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "workspace" / "benchmarks" / "pipeline_benchmark.jsonl"


def append_result(result: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys() -> set[tuple[str, str, str]]:
    if not OUTPUT.exists():
        return set()
    keys = set()
    for line in OUTPUT.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "completed":
            keys.add((item["video_id"], item["stage"], item["configuration"]))
    return keys


def run_stage(
    video_id: str, stage: str, configuration: str, command: list[str], duration: float
) -> dict:
    started = time.time()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    result = {
        "video_id": video_id,
        "stage": stage,
        "configuration": configuration,
        "status": "completed" if process.returncode == 0 else "failed",
        "elapsed_seconds": round(time.time() - started, 3),
        "returncode": process.returncode,
        "command": command,
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "video_duration_seconds": duration,
    }
    result["realtime_factor"] = result["elapsed_seconds"] / duration
    append_result(result)
    print(json.dumps({key: result[key] for key in ("video_id", "stage", "configuration", "status", "elapsed_seconds")}), flush=True)
    return result


def representative_videos(count: int) -> list[str]:
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    videos = videos[videos.video_readable & videos.duration_seconds.notna()].sort_values("duration_seconds")
    positions = [round(i * (len(videos) - 1) / max(1, count - 1)) for i in range(count)]
    return videos.iloc[positions].video_id.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable benchmark of current offline stages.")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--representative-count", type=int, default=3)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("omnishot", "sampling", "siglip-base", "siglip-so400m", "siglip-so400m-final", "asr-large-v3", "asr-turbo"),
        default=("omnishot", "sampling", "siglip-base", "siglip-so400m-final", "asr-large-v3"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    video_ids = args.video_ids or representative_videos(args.representative_count)
    done = completed_keys()
    checkpoint = ROOT / "models" / "omnishotcut" / "OmniShotCut_ckpt.pth"
    if not checkpoint.exists():
        candidates = [
            path for path in (ROOT / "models").rglob("*.pth")
            if "omnishot" in path.name.casefold()
        ]
        if not candidates:
            raise FileNotFoundError("OmniShotCut checkpoint not found")
        checkpoint = candidates[0]

    for video_id in video_ids:
        row = videos.loc[videos.video_id == video_id]
        if len(row) != 1:
            raise ValueError(f"Unknown or duplicate video ID: {video_id}")
        video_path = str(row.iloc[0].video_path)
        duration = float(row.iloc[0].duration_seconds)
        commands = {
            "omnishot": (
                "omnishotcut-default",
                ["uv", "run", "--project", "environments/omnishotcut", "python", "scripts/run_omnishotcut.py", "--input", video_path, "--checkpoint", str(checkpoint), "--output", str(ROOT / "workspace" / "shots" / f"{video_id}.json"), "--force"],
            ),
            "sampling": (
                "stride-5-shot-local",
                ["uv", "run", "--project", "environments/omnishotcut", "python", "scripts/build_sampling_manifest.py", "--video-id", video_id, "--stride", "5", "--stride-only", "--force"],
            ),
            "siglip-base": (
                "google/siglip2-base-patch16-224:batch16:unified",
                ["uv", "run", "--project", "environments/siglip2", "python", "scripts/build_siglip2_sample_index.py", "--video-id", video_id, "--model-id", "google/siglip2-base-patch16-224", "--batch-size", "16", "--unified-select", "--force"],
            ),
            "siglip-so400m": (
                "google/siglip2-so400m-patch14-384:batch8:unified",
                ["uv", "run", "--project", "environments/siglip2", "python", "scripts/build_siglip2_sample_index.py", "--video-id", video_id, "--model-id", "google/siglip2-so400m-patch14-384", "--batch-size", "8", "--unified-select", "--force"],
            ),
            "siglip-so400m-final": (
                "google/siglip2-so400m-patch14-384:batch8:base-final-only",
                ["uv", "run", "--project", "environments/siglip2", "python", "scripts/build_siglip2_final_index.py", "--video-id", video_id, "--source-encoder-id", "google_siglip2_base_patch16_224", "--model-id", "google/siglip2-so400m-patch14-384", "--batch-size", "8", "--force"],
            ),
            "asr-large-v3": (
                "faster-whisper:large-v3:float16:beam1",
                ["uv", "run", "--project", "environments/asr", "python", "scripts/transcribe_faster_whisper.py", "--video-id", video_id, "--model", "large-v3", "--device", "cuda", "--compute-type", "float16", "--beam-size", "1", "--force"],
            ),
            "asr-turbo": (
                "faster-whisper:large-v3-turbo:float16:beam1",
                ["uv", "run", "--project", "environments/asr", "python", "scripts/transcribe_faster_whisper.py", "--video-id", video_id, "--model", "large-v3-turbo", "--device", "cuda", "--compute-type", "float16", "--beam-size", "1", "--force"],
            ),
        }
        for stage in args.stages:
            configuration, command = commands[stage]
            key = (video_id, stage, configuration)
            if key in done and not args.force:
                print(f"SKIP completed {video_id} {stage}", flush=True)
                continue
            if stage not in {"omnishot", "sampling"} and not (ROOT / "workspace" / "samples" / f"{video_id}.parquet").exists() and stage.startswith("siglip"):
                raise FileNotFoundError(f"Missing samples for {video_id}; run omnishot and sampling first")
            result = run_stage(video_id, stage, configuration, command, duration)
            if result["status"] != "completed":
                raise RuntimeError(f"Benchmark failed: {video_id} {stage}. See {OUTPUT}")


if __name__ == "__main__":
    main()
