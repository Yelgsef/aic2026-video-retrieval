from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def unique(values: list[Any]) -> list[str]:
    output, seen = [], set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def load_gemini(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["scene_id"]): row for row in records}


def retrieval_text(scene: Any, description: dict[str, Any]) -> str:
    ocr = unique([item.get("text", "") for item in description.get("ocr_lines", [])])
    sections = [
        ("Nội dung hình ảnh", [description.get("summary_vi", "")]),
        ("Chữ trong hình", ocr),
        ("Lời nói", [scene.asr_exact, scene.asr_context]),
        ("Người", description.get("people", [])),
        ("Vật thể", description.get("objects", [])),
        ("Hành động", description.get("actions", [])),
        ("Địa điểm", description.get("locations", [])),
        ("Sự kiện", description.get("events", [])),
        ("Từ khóa", description.get("search_keywords_vi", [])),
    ]
    return "\n".join(f"{label}: {', '.join(unique(values))}" for label, values in sections if unique(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build versioned multimodal documents from semantic scenes.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--gemini", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    scenes = pd.read_parquet(ROOT / "workspace" / "semantic_scenes" / args.video_id / "scenes.parquet")
    gemini_path = args.gemini or ROOT / "workspace" / "gemini_parsed" / args.video_id / "scenes.jsonl"
    gemini = load_gemini(gemini_path)
    output = args.output_dir or ROOT / "workspace" / "scene_documents" / args.video_id
    documents = []
    for scene in scenes.sort_values("scene_index").itertuples(index=False):
        record = gemini.get(str(scene.scene_id), {})
        description = record.get("description", {})
        ocr_lines = description.get("ocr_lines", [])
        document = {
            "schema_version": "0.2.0", "document_id": str(scene.scene_id), "video_id": args.video_id,
            "scene_index": int(scene.scene_index), "start_frame": int(scene.start_frame),
            "end_frame_exclusive": int(scene.end_frame_exclusive), "start_time": float(scene.start_time),
            "end_time": float(scene.end_time), "shot_ids": [int(v) for v in scene.shot_ids],
            "representative_frame_ids": [int(v) for v in scene.representative_frame_ids],
            "asr_text": str(scene.asr_exact or ""), "asr_context": str(scene.asr_context or ""),
            "visual_summary": str(description.get("summary_vi", "")), "ocr_lines": ocr_lines,
            "ocr_text": unique([item.get("text", "") for item in ocr_lines]),
            "people": unique(description.get("people", [])), "objects": unique(description.get("objects", [])),
            "actions": unique(description.get("actions", [])), "locations": unique(description.get("locations", [])),
            "events": unique(description.get("events", [])),
            "search_keywords": unique(description.get("search_keywords_vi", [])),
            "uncertainties": unique(description.get("uncertainties", [])),
            "has_gemini": bool(record), "retrieval_text": retrieval_text(scene, description),
            "provenance": {"semantic_scenes": f"workspace/semantic_scenes/{args.video_id}/scenes.parquet",
                           "gemini": record.get("provenance"), "asr_model": "faster-whisper-large-v3",
                           "visual_model": "google/siglip2-so400m-patch14-384"},
        }
        documents.append(document)
    output.mkdir(parents=True, exist_ok=True)
    jsonl = output / "scenes.jsonl"
    jsonl.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents), encoding="utf-8")
    flat = [{k: v for k, v in doc.items() if k not in {"ocr_lines", "provenance"}} for doc in documents]
    pd.DataFrame(flat).to_parquet(output / "scenes.parquet", index=False)
    report = {"schema_version": "0.2.0", "video_id": args.video_id, "documents": len(documents),
              "with_gemini": sum(doc["has_gemini"] for doc in documents),
              "with_asr": sum(bool(doc["asr_text"]) for doc in documents), "output": str(output)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
