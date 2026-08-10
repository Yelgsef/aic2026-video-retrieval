from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "workspace/gemini_batches/generic_jobs.sqlite3"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS jobs(
      task TEXT, video_id TEXT, input_hash TEXT PRIMARY KEY, model TEXT, input_path TEXT,
      uploaded_name TEXT, job_name TEXT, state TEXT, result_name TEXT, output_path TEXT,
      created_at TEXT, updated_at TEXT, error TEXT)""")
    return db


def api_client():
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def paths(task: str, video_id: str) -> tuple[Path, Path]:
    if task == "asr-normalize":
        root = ROOT / "workspace/asr_normalization" / video_id
    elif task == "ocr-crops":
        root = ROOT / "workspace/ocr_fullres" / video_id
    else:
        raise ValueError(task)
    return root / ("requests.jsonl" if task == "asr-normalize" else "gemini_requests.jsonl"), root / "responses.jsonl"


def latest(db: sqlite3.Connection, task: str, video_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM jobs WHERE task=? AND video_id=? ORDER BY created_at DESC LIMIT 1",
                     (task, video_id)).fetchone()
    if not row:
        raise ValueError("Run register first")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Gemini Batch manager for prepared auxiliary tasks.")
    parser.add_argument("command", choices=("register", "submit", "status", "download"))
    parser.add_argument("--task", choices=("asr-normalize", "ocr-crops"), required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    args = parser.parse_args()
    input_path, output_path = paths(args.task, args.video_id)
    with connect() as db:
        if args.command == "register":
            payload = input_path.read_bytes()
            if not payload.strip():
                print(json.dumps({"task": args.task, "video_id": args.video_id,
                                  "requests": 0, "status": "nothing_to_submit"}, indent=2))
                return
            digest = hashlib.sha256(payload).hexdigest()
            db.execute("""INSERT OR IGNORE INTO jobs(task,video_id,input_hash,model,input_path,state,
                output_path,created_at,updated_at) VALUES(?,?,?,?,?,'PREPARED',?,?,?)""",
                (args.task, args.video_id, digest, args.model, str(input_path), str(output_path), now(), now()))
            print(json.dumps({"task": args.task, "video_id": args.video_id, "requests": len(payload.splitlines()),
                              "sha256": digest, "input": str(input_path)}, indent=2)); return
        row = latest(db, args.task, args.video_id)
        client = api_client()
        if args.command == "submit":
            if row["job_name"]:
                print(json.dumps({"job": row["job_name"], "state": row["state"]})); return
            from google.genai import types
            uploaded = client.files.upload(file=row["input_path"], config=types.UploadFileConfig(
                display_name=f"aic-{args.task}-{args.video_id}-{row['input_hash'][:10]}", mime_type="jsonl"))
            job = client.batches.create(model=args.model, src=uploaded.name,
                config={"display_name": f"aic-{args.task}-{args.video_id}"})
            state = str(getattr(job.state, "name", job.state)).removeprefix("JobState.")
            db.execute("UPDATE jobs SET uploaded_name=?,job_name=?,state=?,updated_at=? WHERE input_hash=?",
                       (uploaded.name, job.name, state, now(), row["input_hash"]))
            print(json.dumps({"job": job.name, "state": state}, indent=2)); return
        if not row["job_name"]:
            raise ValueError("Batch has not been submitted")
        job = client.batches.get(name=row["job_name"])
        state = str(getattr(job.state, "name", job.state)).removeprefix("JobState.")
        result = getattr(getattr(job, "dest", None), "file_name", None)
        db.execute("UPDATE jobs SET state=?,result_name=?,updated_at=? WHERE input_hash=?",
                   (state, result, now(), row["input_hash"]))
        if args.command == "download":
            if state != "JOB_STATE_SUCCEEDED" or not result:
                raise RuntimeError(f"Batch is not ready: {state}")
            output_path.write_bytes(client.files.download(file=result))
        print(json.dumps({"job": row["job_name"], "state": state, "result": result,
                          "output": str(output_path) if args.command == "download" else None}, indent=2))


if __name__ == "__main__":
    main()
