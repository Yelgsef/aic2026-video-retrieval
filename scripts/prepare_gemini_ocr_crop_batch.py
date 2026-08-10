from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = {"type": "object", "properties": {
    "text": {"type": "string"}, "readable": {"type": "boolean"},
    "language": {"type": "string"}, "uncertainty": {"type": "string"}},
    "required": ["text", "readable", "language", "uncertainty"], "additionalProperties": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Gemini Batch requests for uncertain full-res OCR crops.")
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    root = ROOT / "workspace/ocr_fullres" / args.video_id
    source = root / "gemini_candidates.jsonl"
    candidates = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
    output = root / "gemini_requests.jsonl"
    lines = []
    for row in candidates:
        image = base64.b64encode(Path(row["crop_path"]).read_bytes()).decode("ascii")
        key = f"{args.video_id}:f{row['frame_id']}:r{row['region_index']}"
        prompt = ("Đọc chính xác chữ trong crop video này. Không suy đoán từ ngữ cảnh. "
                  f"OCR local gợi ý: {row['local_text']!r}. Nếu không đọc chắc chắn, đặt readable=false.")
        request = {"contents": [{"role": "user", "parts": [{"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image}}]}],
                   "generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingLevel": "MINIMAL"},
                    "responseMimeType": "application/json", "responseJsonSchema": SCHEMA}}
        lines.append(json.dumps({"key": key, "request": request}, ensure_ascii=False))
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(json.dumps({"video_id": args.video_id, "requests": len(lines), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
