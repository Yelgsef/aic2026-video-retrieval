from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"type": "object", "properties": {
    "normalized_text": {"type": "string"},
    "proper_nouns": {"type": "array", "items": {"type": "string"}},
    "changes": {"type": "array", "items": {"type": "string"}},
    "uncertainties": {"type": "array", "items": {"type": "string"}}},
    "required": ["normalized_text", "proper_nouns", "changes", "uncertainties"],
    "additionalProperties": False}


def suspicious(segment: dict) -> bool:
    words = segment.get("words", [])
    low = sum(float(word.get("probability", 1)) < 0.72 for word in words)
    text = str(segment.get("text", ""))
    return float(segment.get("avg_logprob", 0)) < -0.22 or low >= 2 or any(c.isdigit() for c in text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare conditional Gemini Batch ASR normalization.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--context-segments", type=int, default=1)
    args = parser.parse_args()
    source = ROOT / "workspace/asr/faster_whisper_large_v3" / f"{args.video_id}.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    segments = payload["segments"]
    output_dir = ROOT / "workspace/asr_normalization" / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    lines, selected = [], []
    for index, segment in enumerate(segments):
        if not suspicious(segment):
            continue
        left, right = max(0, index - args.context_segments), min(len(segments), index + args.context_segments + 1)
        context = " ".join(str(x["text"]).strip() for x in segments[left:right])
        prompt = f"""Chuẩn hóa đúng chính tả đoạn ASR tiếng Việt sau.
Chỉ sửa lỗi nhận dạng, dấu câu, chữ số, tên riêng dựa trên chính văn bản; không thêm sự kiện hay kiến thức mới.
Đoạn cần sửa: {segment['text']}
Ngữ cảnh lân cận: {context}
Nếu không chắc tên riêng, giữ nguyên và ghi vào uncertainties. Trả JSON theo schema."""
        request = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingLevel": "MINIMAL"},
                    "responseMimeType": "application/json", "responseJsonSchema": SCHEMA}}
        key = f"{args.video_id}:segment:{int(segment['segment_id'])}"
        lines.append(json.dumps({"key": key, "request": request}, ensure_ascii=False))
        selected.append({"key": key, "segment_id": int(segment["segment_id"]), "raw_text": segment["text"],
                         "start": segment["start"], "end": segment["end"]})
    (output_dir / "requests.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (output_dir / "selected.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
    report = {"video_id": args.video_id, "segments": len(segments), "selected": len(selected),
              "selection_rate": len(selected) / max(1, len(segments)),
              "output": str(output_dir / "requests.jsonl")}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
