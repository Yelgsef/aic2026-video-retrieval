from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "workspace/scheduler/jobs.sqlite3"
DEFAULT_SUBSET = ROOT / "workspace/validation/subset20.json"
LOG = ROOT / "workspace/scheduler/shutdown_watcher.log"


def log(event: str, **detail: object) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **detail}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shut down Windows only after a scheduler subset completes.")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--shutdown-delay", type=int, default=60)
    args = parser.parse_args()
    video_ids = json.loads(args.subset.read_text(encoding="utf-8"))["video_ids"]
    placeholders = ",".join("?" for _ in video_ids)
    log("watcher_started", config_hash=args.config_hash, videos=len(video_ids))
    while True:
        with sqlite3.connect(DB) as db:
            rows = db.execute(
                f"SELECT status,COUNT(*) FROM jobs WHERE config_hash=? AND video_id IN ({placeholders}) GROUP BY status",
                (args.config_hash, *video_ids),
            ).fetchall()
        counts = dict(rows)
        total = sum(counts.values())
        expected = len(video_ids) * 11
        if total != expected:
            log("invalid_job_count", expected=expected, actual=total, counts=counts)
            return
        if counts.get("completed", 0) == expected:
            log("pipeline_completed", counts=counts, shutdown_delay=args.shutdown_delay)
            subprocess.run(["shutdown.exe", "/s", "/t", str(args.shutdown_delay),
                            "/c", "AIC subset-20 pipeline completed successfully."], check=True)
            return
        if counts.get("failed", 0) and not counts.get("running", 0) and not counts.get("retry", 0):
            log("pipeline_blocked_no_shutdown", counts=counts)
            return
        time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    main()
