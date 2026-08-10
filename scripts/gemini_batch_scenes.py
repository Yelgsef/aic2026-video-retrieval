from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "workspace" / "gemini_batches"
DB_PATH = BATCH_ROOT / "jobs.sqlite3"
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

SCHEMA = {
    "type": "object",
    "properties": {
        "summary_vi": {"type": "string"},
        "ocr_lines": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "frame_label": {"type": "string"}},
            "required": ["text", "frame_label"], "additionalProperties": False}},
        "people": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": {"type": "string"}},
        "search_keywords_vi": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary_vi", "ocr_lines", "people", "objects", "actions", "locations", "events",
                 "search_keywords_vi", "uncertainties"],
    "additionalProperties": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_name(value: Any) -> str:
    """Normalize SDK enums across google-genai releases."""
    return str(getattr(value, "name", value)).removeprefix("JobState.")


def db() -> sqlite3.Connection:
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        input_hash TEXT PRIMARY KEY, video_id TEXT NOT NULL, model TEXT NOT NULL,
        jsonl_path TEXT NOT NULL, uploaded_file_name TEXT, batch_job_name TEXT,
        state TEXT NOT NULL, result_file_name TEXT, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, last_error TEXT)""")
    return conn


def load_scenes(video_id: str) -> list[dict[str, Any]]:
    path = ROOT / "workspace" / "semantic_scenes" / video_id / "scenes.json"
    return json.loads(path.read_text(encoding="utf-8"))["scenes"]


def video_path(video_id: str) -> Path:
    rows = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    row = rows.loc[rows.video_id == video_id]
    if len(row) != 1:
        raise ValueError(f"Expected one manifest row for {video_id}")
    return Path(row.iloc[0].video_path)


def extract_frames(path: Path, frame_ids: set[int]) -> dict[int, Image.Image]:
    import cv2
    capture = cv2.VideoCapture(str(path))
    wanted = sorted(frame_ids)
    output: dict[int, Image.Image] = {}
    for frame_id in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if ok:
            output[frame_id] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    missing = frame_ids - output.keys()
    if missing:
        raise RuntimeError(f"Could not decode {len(missing)} representative frames")
    return output


def contact_sheet(scene: dict[str, Any], frames: dict[int, Image.Image], output: Path) -> None:
    reps = scene["representative_frames"]
    tile_w, tile_h, label_h, cols = 448, 252, 28, 2
    rows = (len(reps) + cols - 1) // cols
    sheet = Image.new("RGB", (tile_w * cols, (tile_h + label_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, rep in enumerate(reps):
        image = frames[int(rep["frame_id"])].copy()
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h), "black")
        canvas.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
        x, y = (index % cols) * tile_w, (index // cols) * (tile_h + label_h)
        sheet.paste(canvas, (x, y))
        label = f"F{index + 1} | frame {rep['frame_id']} | {float(rep['pts_time']):.2f}s"
        draw.text((x + 6, y + tile_h + 7), label, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=82, optimize=True)


def prompt(scene: dict[str, Any]) -> str:
    transcript = scene.get("asr_exact") or scene.get("asr_context") or "(không có transcript)"
    return f"""Phân tích scene video AIC từ các frame đại diện F1..Fn theo thời gian.
Chỉ ghi nhận chi tiết nhìn thấy rõ hoặc được transcript hỗ trợ; không đoán danh tính.
OCR phải chép nguyên văn và gắn frame_label. Mô tả/từ khóa bằng tiếng Việt, giữ nguyên tên riêng.
Khoảng thời gian: {scene['start_time']:.2f}s–{scene['end_time']:.2f}s.
Transcript ASR (có thể sai): {transcript}
Trả về đúng JSON theo schema."""


def prepare(args: argparse.Namespace) -> None:
    scenes = load_scenes(args.video_id)
    out_dir = BATCH_ROOT / args.video_id
    sheets = out_dir / "contact_sheets"
    jsonl = out_dir / "requests.jsonl"
    ids = {int(rep["frame_id"]) for scene in scenes for rep in scene["representative_frames"]}
    decoded = extract_frames(video_path(args.video_id), ids)
    lines: list[str] = []
    for scene_index, scene in enumerate(scenes):
        scene_id = str(scene["scene_id"])
        image_path = sheets / f"scene_{scene_index:05d}.jpg"
        contact_sheet(scene, decoded, image_path)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        request = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt(scene)},
                {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
            ]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json",
                                   "responseJsonSchema": SCHEMA},
        }
        lines.append(json.dumps({"key": scene_id, "request": request}, ensure_ascii=False))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with db() as conn:
        conn.execute("""INSERT INTO jobs(input_hash,video_id,model,jsonl_path,state,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?) ON CONFLICT(input_hash) DO UPDATE SET jsonl_path=excluded.jsonl_path,
            updated_at=excluded.updated_at""", (digest, args.video_id, args.model, str(jsonl), "PREPARED", now(), now()))
    print(json.dumps({"video_id": args.video_id, "scenes": len(scenes), "unique_frames": len(ids),
                      "jsonl": str(jsonl), "bytes": len(payload), "sha256": digest}, indent=2))


def client():
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured")
    retry = types.HttpRetryOptions(attempts=2, initial_delay=1, max_delay=4,
                                   http_status_codes=[429, 500, 502, 503, 504])
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                        http_options=types.HttpOptions(timeout=120_000, retry_options=retry))


def find_job(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE video_id=? ORDER BY created_at DESC LIMIT 1", (video_id,)).fetchone()
    if not row:
        raise ValueError(f"No prepared batch for {video_id}")
    return row


def submit(args: argparse.Namespace) -> None:
    from google.genai import types
    api = client()
    with db() as conn:
        row = find_job(conn, args.video_id)
        if row["batch_job_name"]:
            print(f"Already submitted: {row['batch_job_name']}")
            return
        uploaded_name = row["uploaded_file_name"]
        if not uploaded_name:
            uploaded = api.files.upload(file=row["jsonl_path"], config=types.UploadFileConfig(
                display_name=f"aic-{args.video_id}-{row['input_hash'][:12]}", mime_type="jsonl"))
            uploaded_name = uploaded.name
            conn.execute("UPDATE jobs SET uploaded_file_name=?,state='UPLOADED',updated_at=? WHERE input_hash=?",
                         (uploaded_name, now(), row["input_hash"]))
        job = api.batches.create(model=args.model, src=uploaded_name,
                                 config={"display_name": f"aic-{args.video_id}-{row['input_hash'][:12]}"})
        state = state_name(job.state)
        conn.execute("UPDATE jobs SET model=?,batch_job_name=?,state=?,updated_at=? WHERE input_hash=?",
                     (args.model, job.name, state, now(), row["input_hash"]))
        print(json.dumps({"job": job.name, "state": state}, indent=2))


def status(args: argparse.Namespace, download_result: bool = False) -> None:
    api = client()
    with db() as conn:
        row = find_job(conn, args.video_id)
        if not row["batch_job_name"]:
            raise ValueError("Batch is prepared but has not been submitted")
        job = api.batches.get(name=row["batch_job_name"])
        state = state_name(job.state)
        result_name = getattr(getattr(job, "dest", None), "file_name", None)
        conn.execute("UPDATE jobs SET state=?,result_file_name=?,updated_at=? WHERE input_hash=?",
                     (state, result_name, now(), row["input_hash"]))
        print(json.dumps({"job": job.name, "state": state, "result_file": result_name}, indent=2))
        if download_result:
            if state != "JOB_STATE_SUCCEEDED" or not result_name:
                raise RuntimeError(f"Batch is not ready: {state}")
            output = BATCH_ROOT / args.video_id / "responses.jsonl"
            output.write_bytes(api.files.download(file=result_name))
            print(f"Downloaded: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and manage resumable Gemini Batch scene enrichment.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "submit", "status", "download"):
        item = sub.add_parser(name)
        item.add_argument("--video-id", required=True)
        item.add_argument("--model", default="gemini-3.6-flash")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "submit":
        submit(args)
    else:
        status(args, download_result=args.command == "download")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
