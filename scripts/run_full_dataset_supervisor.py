from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "workspace" / "scheduler" / "jobs.sqlite3"
LOCAL_STAGES = (
    "omnishot", "sampling", "siglip-base", "siglip-so400m-final",
    "asr-large-v3", "semantic-scenes", "objects-yoloworld", "fullres-ocr",
    "asr-normalize-prepare", "gemini-batch-prepare", "gemini-ocr-crop-prepare",
)


def ids_from_manifest() -> list[str]:
    table = pd.read_parquet(ROOT / "workspace/manifests/videos.parquet")
    return [str(value) for value in table["video_id"].tolist()]


def local_state(video_id: str) -> tuple[int, int, int]:
    if not DB_PATH.exists():
        return 0, 0, len(LOCAL_STAGES)
    db = sqlite3.connect(DB_PATH)
    marks = ",".join("?" for _ in LOCAL_STAGES)
    rows = db.execute(
        f"SELECT status,COUNT(*) FROM jobs WHERE video_id=? AND stage IN ({marks}) GROUP BY status",
        (video_id, *LOCAL_STAGES),
    ).fetchall()
    db.close()
    counts = {str(status): int(count) for status, count in rows}
    return counts.get("completed", 0), counts.get("failed", 0), sum(counts.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four local shards and resumable per-video 9Router/index jobs.")
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--video-ids", nargs="*", default=[])
    parser.add_argument("--model", default=None)
    parser.add_argument("--gemini-workers", type=int, default=4)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--quota-wait", type=int, default=1800)
    parser.add_argument("--retry-wait", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    video_ids = args.video_ids or ids_from_manifest()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    scheduler = ["uv", "run", "--project", "environments/siglip2", "python",
                 "scripts/run_dataset_scheduler.py", "--run",
                 "--shard-count", str(args.shard_count), "--shard-index"]
    init = subprocess.run(["uv", "run", "--project", "environments/siglip2", "python",
                           "scripts/run_dataset_scheduler.py", "--init",
                           "--shard-count", str(args.shard_count)], cwd=ROOT, env=env)
    if init.returncode:
        raise SystemExit(init.returncode)

    local_workers: list[subprocess.Popen[str]] = []
    for shard in range(args.shard_count):
        command = [*scheduler, str(shard), "--worker-id", f"full-supervisor:{shard}"]
        worker_env = env.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(shard)
        local_workers.append(subprocess.Popen(command, cwd=ROOT, env=worker_env))
    print(f"LOCAL_WORKERS_STARTED count={len(local_workers)} videos={len(video_ids)}", flush=True)

    remaining = set(video_ids)
    retry_at: dict[str, float] = {}
    active: tuple[str, subprocess.Popen[str]] | None = None
    try:
        while remaining or active is not None:
            if active is not None:
                video_id, process = active
                code = process.poll()
                if code is not None:
                    if code == 0:
                        print(f"GEMINI_COMPLETE {video_id}", flush=True)
                        remaining.remove(video_id)
                    elif code == 75:
                        retry_at[video_id] = time.time() + args.quota_wait
                        print(f"GEMINI_QUOTA_PAUSE {video_id} seconds={args.quota_wait}", flush=True)
                    else:
                        retry_at[video_id] = time.time() + args.retry_wait
                        print(f"GEMINI_RETRY {video_id} code={code} seconds={args.retry_wait}", flush=True)
                    active = None
            if active is None:
                now = time.time()
                ready = []
                for video_id in sorted(remaining):
                    completed, failed, total = local_state(video_id)
                    if failed:
                        print(f"LOCAL_FAILED {video_id} completed={completed} failed={failed}", flush=True)
                        continue
                    if completed == len(LOCAL_STAGES) and now >= retry_at.get(video_id, 0):
                        ready.append(video_id)
                if ready:
                    video_id = ready[0]
                    output = ROOT / "workspace" / f"offline_index_{video_id}"
                    command = ["uv", "run", "--project", "environments/siglip2", "python",
                               "scripts/run_gemini_subset_supervisor.py",
                               "--video-id", video_id, "--workers", str(args.gemini_workers),
                               "--min-interval", str(args.min_interval), "--output", str(output)]
                    if args.model:
                        command.extend(["--model", args.model])
                    print(f"GEMINI_START {video_id}", flush=True)
                    active = video_id, subprocess.Popen(command, cwd=ROOT, env=env)
            if remaining or active is not None:
                time.sleep(max(5, args.poll_seconds))
    finally:
        for process in local_workers:
            if process.poll() is None:
                process.terminate()
        for process in local_workers:
            process.wait()
    print("FULL_DATASET_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
