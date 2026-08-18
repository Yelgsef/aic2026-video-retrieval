from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "workspace" / "scheduler" / "jobs.sqlite3"
STOP_REQUESTED = False
STAGES = ("omnishot", "sampling", "siglip-base", "siglip-so400m-final", "asr-large-v3",
          "semantic-scenes", "objects-yoloworld", "fullres-ocr", "asr-normalize-prepare",
          "gemini-batch-prepare", "gemini-ocr-crop-prepare")
DEPENDENCY = {
    "omnishot": None,
    "sampling": "omnishot",
    "siglip-base": "sampling",
    "siglip-so400m-final": "siglip-base",
    "asr-large-v3": None,
    "semantic-scenes": ("siglip-so400m-final", "asr-large-v3"),
    "objects-yoloworld": "semantic-scenes",
    "fullres-ocr": "siglip-so400m-final",
    "asr-normalize-prepare": "asr-large-v3",
    "gemini-batch-prepare": "semantic-scenes",
    "gemini-ocr-crop-prepare": "fullres-ocr",
}


def configuration_hash() -> str:
    payload = {
        "pipeline_version": "0.5.0",
        "sampling_stride": 5,
        "sampling_fixed_coverage_anchor": 0,
        "temporal_chunk_frames": 67,
        "temporal_chunk_overlap": 10,
        "max_medoids_per_chunk": 2,
        "post_selection_coverage_window": 103,
        "visual_model": "google/siglip2-base-patch16-224",
        "visual_batch_size": 16,
        "rerank_model": "google/siglip2-so400m-patch14-384",
        "asr_model": "large-v3",
        "asr_compute_type": "float16",
        "asr_beam_size": 1,
        "semantic_scene_version": "0.2.0",
        "gemini_batch_schema_version": "0.1.0",
        "object_model": "yolov8s-worldv2",
        "object_confidence": 0.12,
        "ocr_model": "PP-OCRv6-small-vi",
        "ocr_confidence": 0.35,
        "ocr_gemini_threshold": 0.78,
        "ocr_batch_size": 16,
        "asr_normalization": "conditional-gemini-minimal",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            video_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            shard INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            worker_id TEXT,
            started_at REAL,
            heartbeat_at REAL,
            finished_at REAL,
            elapsed_seconds REAL,
            last_error TEXT,
            PRIMARY KEY(video_id, stage, config_hash)
        );
        CREATE INDEX IF NOT EXISTS jobs_status_idx
            ON jobs(config_hash, status, shard, stage);
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            video_id TEXT,
            stage TEXT,
            event TEXT NOT NULL,
            detail TEXT
        );
        """
    )
    return connection


def initialize(connection: sqlite3.Connection, shard_count: int) -> int:
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    config = configuration_hash()
    inserted = 0
    with connection:
        for index, row in enumerate(videos.sort_values("video_id").itertuples(index=False)):
            if not bool(row.video_readable):
                continue
            for stage in STAGES:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO jobs(video_id,stage,config_hash,shard) VALUES(?,?,?,?)",
                    (row.video_id, stage, config, index % shard_count),
                )
                inserted += cursor.rowcount
            # Existing databases may have been initialized with one shard.
            # Rebalance them when a multi-GPU run requests a new shard count.
            connection.execute(
                "UPDATE jobs SET shard=? WHERE video_id=? AND config_hash=?",
                (index % shard_count, row.video_id, config),
            )
    return inserted


def artifact_valid(video_id: str, stage: str) -> bool:
    try:
        if stage == "omnishot":
            value = json.loads((ROOT / "workspace" / "shots" / f"{video_id}.json").read_text(encoding="utf-8"))
            return value.get("shot_count", 0) > 0 and bool(value.get("ranges"))
        if stage == "sampling":
            table = pd.read_parquet(ROOT / "workspace" / "samples" / f"{video_id}.parquet")
            return len(table) > 0 and not table.frame_id.duplicated().any()
        if stage == "siglip-base":
            directory = ROOT / "workspace" / "retrieval" / "google_siglip2_base_patch16_224" / f"{video_id}_unified"
            vectors = np.load(directory / "embeddings.npy", mmap_mode="r")
            metadata = pd.read_parquet(directory / "metadata.parquet")
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            selection = config.get("selection", {})
            return (len(vectors) == len(metadata) > 0 and metadata.shot_id.nunique() > 0
                    and selection.get("temporal_chunk_frames") == 67
                    and selection.get("temporal_chunk_overlap") == 10
                    and selection.get("max_medoids_per_chunk") == 2
                    and selection.get("coverage_window_frames") == 103)
        if stage == "siglip-so400m-final":
            directory = ROOT / "workspace" / "retrieval" / "google_siglip2_so400m_patch14_384" / f"{video_id}_final_from_google_siglip2_base_patch16_224"
            vectors = np.load(directory / "embeddings.npy", mmap_mode="r")
            metadata = pd.read_parquet(directory / "metadata.parquet")
            return len(vectors) == len(metadata) > 0 and vectors.shape[1] == 1152
        if stage == "asr-large-v3":
            value = json.loads((ROOT / "workspace" / "asr" / "faster_whisper_large_v3" / f"{video_id}.json").read_text(encoding="utf-8"))
            return value.get("duration", 0) > 0 and "segments" in value
        if stage == "semantic-scenes":
            table = pd.read_parquet(ROOT / "workspace" / "semantic_scenes" / video_id / "scenes.parquet")
            return len(table) > 0 and table.scene_id.is_unique and bool((table.end_time > table.start_time).all())
        if stage == "gemini-batch-prepare":
            path = ROOT / "workspace" / "gemini_batches" / video_id / "requests.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            return len(lines) > 0 and all(json.loads(line).get("key") for line in lines)
        if stage == "objects-yoloworld":
            table = pd.read_parquet(ROOT / "workspace" / "objects_yoloworld" / video_id / "detections.parquet")
            report = json.loads((ROOT / "workspace" / "objects_yoloworld" / video_id / "report.json").read_text(encoding="utf-8"))
            return report.get("decoded_frames", 0) == report.get("final_frames", -1) and "frame_id" in table.columns
        if stage == "fullres-ocr":
            table = pd.read_parquet(ROOT / "workspace" / "ocr_fullres" / video_id / "regions.parquet")
            report = json.loads((ROOT / "workspace" / "ocr_fullres" / video_id / "report.json").read_text(encoding="utf-8"))
            return (len(table) == report.get("regions", -1) and report.get("frames", 0) > 0
                    and report.get("complete") is True
                    and report.get("det_model") == "PP-OCRv6_small_det"
                    and report.get("rec_model") == "PP-OCRv6_small_rec")
        if stage == "asr-normalize-prepare":
            path = ROOT / "workspace" / "asr_normalization" / video_id / "requests.jsonl"
            return path.exists() and (ROOT / "workspace" / "asr_normalization" / video_id / "report.json").exists()
        if stage == "gemini-ocr-crop-prepare":
            return (ROOT / "workspace" / "ocr_fullres" / video_id / "gemini_requests.jsonl").exists()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return False


def reconcile(connection: sqlite3.Connection, stale_seconds: int) -> dict[str, int]:
    config = configuration_hash()
    now = time.time()
    reclaimed = connection.execute(
        "UPDATE jobs SET status='pending',worker_id=NULL,started_at=NULL,heartbeat_at=NULL,last_error='stale lease reclaimed' "
        "WHERE config_hash=? AND status='running' AND heartbeat_at < ?",
        (config, now - stale_seconds),
    ).rowcount
    completed = 0
    rows = connection.execute(
        "SELECT video_id,stage FROM jobs WHERE config_hash=? AND status!='completed'",
        (config,),
    ).fetchall()
    for row in rows:
        if artifact_valid(row["video_id"], row["stage"]):
            connection.execute(
                "UPDATE jobs SET status='completed',finished_at=?,last_error=NULL WHERE video_id=? AND stage=? AND config_hash=?",
                (now, row["video_id"], row["stage"], config),
            )
            completed += 1
    connection.commit()
    return {"reclaimed": reclaimed, "artifacts_reconciled": completed}


def claim(connection: sqlite3.Connection, shard: int, worker_id: str, stages: tuple[str, ...],
          video_ids: tuple[str, ...] = ()) -> sqlite3.Row | None:
    config = configuration_hash()
    connection.execute("BEGIN IMMEDIATE")
    try:
        placeholders = ",".join("?" for _ in stages)
        shard_filter = " AND shard=?"
        video_filter = ""
        parameters: list[object] = [config, shard, *stages]
        if video_ids:
            # An explicit validation subset must work regardless of the shard
            # count used when the persistent database was first initialized.
            shard_filter = ""
            video_filter = f" AND video_id IN ({','.join('?' for _ in video_ids)})"
            parameters = [config, *stages, *video_ids]
        rows = connection.execute(
            f"SELECT * FROM jobs WHERE config_hash=?{shard_filter} AND status IN ('pending','retry') AND stage IN ({placeholders}){video_filter} ORDER BY video_id, CASE stage WHEN 'omnishot' THEN 1 WHEN 'sampling' THEN 2 WHEN 'siglip-base' THEN 3 WHEN 'siglip-so400m-final' THEN 4 WHEN 'asr-large-v3' THEN 5 WHEN 'semantic-scenes' THEN 6 WHEN 'objects-yoloworld' THEN 7 WHEN 'fullres-ocr' THEN 8 WHEN 'asr-normalize-prepare' THEN 9 WHEN 'gemini-batch-prepare' THEN 10 WHEN 'gemini-ocr-crop-prepare' THEN 11 ELSE 12 END",
            parameters,
        ).fetchall()
        selected = None
        for row in rows:
            dependency = DEPENDENCY[row["stage"]]
            if dependency is None:
                selected = row
                break
            dependencies = dependency if isinstance(dependency, tuple) else (dependency,)
            dependency_rows = connection.execute(
                f"SELECT stage,status FROM jobs WHERE video_id=? AND stage IN ({','.join('?' for _ in dependencies)}) AND config_hash=?",
                (row["video_id"], *dependencies, config),
            ).fetchall()
            if len(dependency_rows) == len(dependencies) and all(item["status"] == "completed" for item in dependency_rows):
                selected = row
                break
        if selected is None:
            connection.commit()
            return None
        now = time.time()
        connection.execute(
            "UPDATE jobs SET status='running',attempts=attempts+1,worker_id=?,started_at=?,heartbeat_at=? WHERE video_id=? AND stage=? AND config_hash=?",
            (worker_id, now, now, selected["video_id"], selected["stage"], config),
        )
        connection.commit()
        return selected
    except Exception:
        connection.rollback()
        raise


def command_for(video_id: str, stage: str) -> list[str]:
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    row = videos.loc[videos.video_id == video_id].iloc[0]
    checkpoint = ROOT / "models" / "omnishotcut" / "OmniShotCut_ckpt.pth"
    commands = {
        "omnishot": ["uv", "run", "--project", "environments/omnishotcut", "python", "scripts/run_omnishotcut.py", "--input", str(row.video_path), "--checkpoint", str(checkpoint), "--output", str(ROOT / "workspace" / "shots" / f"{video_id}.json"), "--decoder-backend", "auto", "--prefetch-windows", "3", "--preprocess-device", "cuda", "--force"],
        "sampling": ["uv", "run", "--project", "environments/omnishotcut", "python", "scripts/build_sampling_manifest.py", "--video-id", video_id, "--stride", "5", "--stride-only", "--coverage-anchor-interval", "0", "--force"],
        "siglip-base": ["uv", "run", "--project", "environments/siglip2", "python", "scripts/build_siglip2_sample_index.py", "--video-id", video_id, "--model-id", "google/siglip2-base-patch16-224", "--batch-size", "16", "--unified-select", "--force"],
        "siglip-so400m-final": ["uv", "run", "--project", "environments/siglip2", "python", "scripts/build_siglip2_final_index.py", "--video-id", video_id, "--source-encoder-id", "google_siglip2_base_patch16_224", "--model-id", "google/siglip2-so400m-patch14-384", "--batch-size", "8", "--force"],
        "asr-large-v3": ["uv", "run", "--project", "environments/asr", "python", "scripts/transcribe_faster_whisper.py", "--video-id", video_id, "--model", "large-v3", "--device", "cuda", "--compute-type", "float16", "--beam-size", "1", "--force"],
        "semantic-scenes": ["uv", "run", "--project", "environments/siglip2", "python", "scripts/build_semantic_scenes.py", "--video-id", video_id, "--force"],
        "objects-yoloworld": ["uv", "run", "--project", "environments/objects", "python", "scripts/extract_yoloworld_objects.py", "--video-id", video_id, "--batch-size", "8", "--force"],
        "gemini-batch-prepare": ["uv", "run", "--project", "environments/siglip2", "python", "scripts/gemini_batch_scenes.py", "prepare", "--video-id", video_id],
        "fullres-ocr": ["uv", "run", "--project", "environments/ocr", "python", "scripts/extract_fullres_ocr.py", "--video-id", video_id],
        "asr-normalize-prepare": ["uv", "run", "--project", "environments/gemini", "python", "scripts/prepare_gemini_asr_normalization.py", "--video-id", video_id],
        "gemini-ocr-crop-prepare": ["uv", "run", "--project", "environments/gemini", "python", "scripts/prepare_gemini_ocr_crop_batch.py", "--video-id", video_id],
    }
    return commands[stage]


def mark(connection: sqlite3.Connection, row: sqlite3.Row, status: str, elapsed: float, error: str | None = None) -> None:
    with connection:
        connection.execute(
            "UPDATE jobs SET status=?,finished_at=?,elapsed_seconds=?,last_error=?,heartbeat_at=? WHERE video_id=? AND stage=? AND config_hash=?",
            (status, time.time(), elapsed, error, time.time(), row["video_id"], row["stage"], row["config_hash"]),
        )
        connection.execute(
            "INSERT INTO events(created_at,video_id,stage,event,detail) VALUES(?,?,?,?,?)",
            (time.time(), row["video_id"], row["stage"], status, error),
        )


def status_report(connection: sqlite3.Connection) -> dict:
    rows = connection.execute(
        "SELECT stage,status,COUNT(*) AS count,SUM(COALESCE(elapsed_seconds,0)) AS seconds FROM jobs WHERE config_hash=? GROUP BY stage,status ORDER BY stage,status",
        (configuration_hash(),),
    ).fetchall()
    return {"config_hash": configuration_hash(), "database": str(DB_PATH), "counts": [dict(row) for row in rows]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Crash-resumable AIC dataset scheduler.")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--worker-id", default=f"{os.environ.get('COMPUTERNAME','worker')}:{os.getpid()}")
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means run until the queue is empty.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--video-ids", nargs="+", default=[],
                        help="Restrict work to an explicit validation subset.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--stale-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be within shard-count")

    connection = connect()
    if args.init:
        print(json.dumps({"inserted": initialize(connection, args.shard_count), **reconcile(connection, args.stale_seconds)}, indent=2))
    if args.status:
        print(json.dumps(status_report(connection), indent=2))
    if not args.run:
        return

    def request_stop(*_: object) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    reconcile(connection, args.stale_seconds)
    processed = 0
    while not STOP_REQUESTED and (args.max_jobs == 0 or processed < args.max_jobs):
        row = claim(connection, args.shard_index, args.worker_id, tuple(args.stages),
                    tuple(args.video_ids))
        if row is None:
            break
        started = time.time()
        process = subprocess.run(command_for(row["video_id"], row["stage"]), cwd=ROOT, text=True, capture_output=True)
        elapsed = time.time() - started
        valid = process.returncode == 0 and artifact_valid(row["video_id"], row["stage"])
        if valid:
            mark(connection, row, "completed", elapsed)
        else:
            error = (process.stderr or process.stdout)[-4000:]
            retry_status = "failed" if row["attempts"] + 1 >= args.max_attempts else "retry"
            mark(connection, row, retry_status, elapsed, error)
        processed += 1
        print(json.dumps({"video_id": row["video_id"], "stage": row["stage"], "ok": valid, "elapsed_seconds": round(elapsed, 2)}), flush=True)
    print(json.dumps(status_report(connection), indent=2))


if __name__ == "__main__":
    main()
