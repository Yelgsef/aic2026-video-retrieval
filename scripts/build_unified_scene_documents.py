from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def aligned_segments(
    segments: list[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    aligned = []
    for segment in segments:
        overlap = interval_overlap(float(segment["start"]), float(segment["end"]), start, end)
        if overlap <= 0:
            continue
        aligned.append(
            {
                "segment_id": int(segment["segment_id"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "overlap_seconds": overlap,
                "text": segment["text"].strip(),
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
            }
        )
    return aligned


def aligned_words(
    words: list[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    """Assign each word to one shot using its temporal midpoint."""
    aligned = []
    for word in words:
        word_start = float(word["start"])
        word_end = float(word["end"])
        midpoint = (word_start + word_end) / 2
        if not (start <= midpoint < end):
            continue
        aligned.append(
            {
                "start": word_start,
                "end": word_end,
                "text": word["text"].strip(),
                "probability": word.get("probability"),
            }
        )
    return aligned


def compact_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def render_document(description: dict[str, Any], transcript: str) -> str:
    ocr = compact_unique([item["text"] for item in description.get("ocr_lines", [])])
    sections = [f"Nội dung hình ảnh: {description['summary_vi']}"]
    mappings = [
        ("Chữ trong hình", ocr),
        ("Lời nói", [transcript] if transcript else []),
        ("Người", description.get("people", [])),
        ("Vật thể", description.get("objects", [])),
        ("Hành động", description.get("actions", [])),
        ("Địa điểm", description.get("location", [])),
        ("Sự kiện", description.get("event", [])),
        ("Từ khóa", description.get("search_keywords_vi", [])),
    ]
    for label, values in mappings:
        values = compact_unique(values)
        if values:
            sections.append(f"{label}: {', '.join(values)}")
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align Gemini scene evidence with Whisper ASR.")
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--gemini-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    gemini_dir = args.gemini_dir or ROOT / "workspace" / "gemini_smoke" / args.video_id
    output_dir = args.output_dir or ROOT / "workspace" / "scene_documents" / args.video_id
    shot_path = ROOT / "workspace" / "shots" / f"{args.video_id}.json"
    asr_path = ROOT / "workspace" / "asr" / "faster_whisper_large_v3" / f"{args.video_id}.json"
    video_manifest_path = ROOT / "workspace" / "manifests" / "videos.parquet"

    shots = json.loads(shot_path.read_text(encoding="utf-8"))
    asr = json.loads(asr_path.read_text(encoding="utf-8"))
    videos = pd.read_parquet(video_manifest_path)
    video_row = videos.loc[videos.video_id == args.video_id]
    if len(video_row) != 1:
        raise RuntimeError(f"Expected exactly one video manifest row for {args.video_id}")
    fps = float(video_row.iloc[0].fps)

    documents: list[dict[str, Any]] = []
    for path in sorted(gemini_dir.glob("shot_*.json")):
        gemini = json.loads(path.read_text(encoding="utf-8"))
        shot_id = int(gemini["shot_id"])
        start_frame, end_frame_exclusive = map(int, shots["ranges"][shot_id])
        start_time = start_frame / fps
        end_time = end_frame_exclusive / fps
        segments = aligned_segments(asr["segments"], start_time, end_time)
        words = aligned_words(asr["words"], start_time, end_time)
        transcript = " ".join(word["text"] for word in words if word["text"]).strip()
        transcript_context = " ".join(
            compact_unique([segment["text"] for segment in segments])
        ).strip()
        description = gemini["description"]
        ocr_text = compact_unique([item["text"] for item in description.get("ocr_lines", [])])
        documents.append(
            {
                "document_id": f"{args.video_id}:shot:{shot_id:06d}",
                "video_id": args.video_id,
                "shot_id": shot_id,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame_exclusive,
                "start_time": start_time,
                "end_time": end_time,
                "keyframes": gemini["frames"],
                "visual_summary": description["summary_vi"],
                "ocr_text": ocr_text,
                "people": description.get("people", []),
                "objects": description.get("objects", []),
                "actions": description.get("actions", []),
                "locations": description.get("location", []),
                "events": description.get("event", []),
                "search_keywords": description.get("search_keywords_vi", []),
                "visual_uncertainties": description.get("uncertainties", []),
                "asr_text": transcript,
                "asr_context": transcript_context,
                "asr_segments": segments,
                "asr_words": words,
                "retrieval_text": render_document(description, transcript),
                "provenance": {
                    "visual_model": gemini["model"],
                    "asr_model": asr["model"],
                    "gemini_result": str(path),
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "scenes.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents),
        encoding="utf-8",
    )
    parquet_path = output_dir / "scenes.parquet"
    flat_rows = [
        {
            key: value
            for key, value in doc.items()
            if key not in {"keyframes", "asr_segments", "asr_words", "provenance"}
        }
        for doc in documents
    ]
    pd.DataFrame(flat_rows).to_parquet(parquet_path, index=False)
    report = {
        "video_id": args.video_id,
        "document_count": len(documents),
        "with_asr": sum(bool(doc["asr_text"]) for doc in documents),
        "with_ocr": sum(bool(doc["ocr_text"]) for doc in documents),
        "jsonl": str(jsonl_path),
        "parquet": str(parquet_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
