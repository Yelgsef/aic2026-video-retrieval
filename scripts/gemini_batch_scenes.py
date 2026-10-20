from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aic_pipeline.scene_evidence import adaptive_frame_ids


BATCH_ROOT = ROOT / "workspace" / "gemini_batches"
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


def _contact_sheet(scene: dict[str, Any], frames: dict[int, Image.Image],
                   frame_ids: list[int] | None = None) -> Image.Image:
    ids = frame_ids or [int(row["frame_id"]) for row in scene["representative_frames"]]
    tile_w, tile_h, label_h, cols = 320, 180, 24, 5
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGB", (tile_w * cols, (tile_h + label_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, frame_id in enumerate(ids):
        image = frames[int(frame_id)].copy()
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h), "black")
        canvas.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
        x, y = (index % cols) * tile_w, (index // cols) * (tile_h + label_h)
        sheet.paste(canvas, (x, y))
        fps = max(1e-6, (int(scene["end_frame_exclusive"]) - int(scene["start_frame"]))
                  / max(1e-6, float(scene["end_time"]) - float(scene["start_time"])))
        pts_time = float(scene["start_time"]) + (int(frame_id) - int(scene["start_frame"])) / fps
        label = f"F{index + 1} | frame {frame_id} | {pts_time:.2f}s"
        draw.text((x + 6, y + tile_h + 7), label, fill="black", font=font)
    return sheet


def contact_sheet(scene: dict[str, Any], frames: dict[int, Image.Image], output: Path,
                  frame_ids: list[int] | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _contact_sheet(scene, frames, frame_ids).save(output, "JPEG", quality=82, optimize=True)


def prompt(scene: dict[str, Any]) -> str:
    transcript = scene.get("asr_exact") or scene.get("asr_context") or "(không có transcript)"
    return f"""Phân tích scene video AIC từ các frame đại diện F1..Fn theo thời gian.
Chỉ ghi nhận chi tiết nhìn thấy rõ hoặc được transcript hỗ trợ; không đoán danh tính.
OCR phải chép nguyên văn và gắn frame_label. Mô tả/từ khóa bằng tiếng Việt, giữ nguyên tên riêng.
Không đưa vào ocr_lines: đồng hồ/timestamp, ngày giờ phát sóng, watermark/logo kênh,
website/hotline cố định, ticker chạy hoặc footer lặp lại. Vẫn giữ lower-third nếu nó
chứa tên người, chức vụ, địa danh, tiêu đề hoặc thông tin riêng của scene.
Khoảng thời gian: {scene['start_time']:.2f}s–{scene['end_time']:.2f}s.
Transcript ASR (có thể sai): {transcript}
Trả về đúng JSON theo schema."""


def prepare(args: argparse.Namespace) -> None:
    scenes = load_scenes(args.video_id)
    out_dir = BATCH_ROOT / args.video_id
    jsonl = out_dir / "requests.jsonl"
    evidence = {str(scene["scene_id"]): adaptive_frame_ids(scene) for scene in scenes}
    ids = {frame_id for values in evidence.values() for frame_id in values}
    decoded = extract_frames(video_path(args.video_id), ids)
    lines: list[str] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        scene_ids = evidence[scene_id]
        image_buffer = BytesIO()
        _contact_sheet(scene, decoded, scene_ids).save(image_buffer, "JPEG", quality=82, optimize=True)
        encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        request = {
            "frame_ids": scene_ids,
            "contents": [{"role": "user", "parts": [
                {"text": prompt(scene)},
                {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
            ]}],
        }
        lines.append(json.dumps({"key": scene_id, "request": request}, ensure_ascii=False))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(json.dumps({"video_id": args.video_id, "scenes": len(scenes), "unique_frames": len(ids),
                      "jsonl": str(jsonl), "bytes": len(payload), "sha256": digest}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare scene contact sheets for 9Router enrichment.")
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--video-id", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
