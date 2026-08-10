from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.ocr_filter import filter_ocr_lines, recurrence_counts


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


def retrieval_text(scene: Any, description: dict[str, Any], detected_objects: list[str] | None = None,
                   searchable_ocr: list[str] | None = None, normalized_asr: str | None = None) -> str:
    ocr = searchable_ocr if searchable_ocr is not None else unique(
        [item.get("text", "") for item in description.get("ocr_lines", [])])
    sections = [
        ("Nội dung hình ảnh", [description.get("summary_vi", "")]),
        ("Chữ trong hình", ocr),
        ("Lời nói", [normalized_asr or scene.asr_exact, scene.asr_context]),
        ("Người", description.get("people", [])),
        ("Vật thể", list(description.get("objects", [])) + list(detected_objects or [])),
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
    descriptions = [record.get("description", {}) for record in gemini.values()]
    ocr_recurrence = recurrence_counts(descriptions)
    scene_count = len(scenes)
    detections_path = ROOT / "workspace" / "objects_yoloworld" / args.video_id / "detections.parquet"
    detected_by_scene: dict[int, list[str]] = {}
    if detections_path.exists():
        detections = pd.read_parquet(detections_path)
        detections = detections.loc[detections.confidence.ge(0.20)]
        for scene_index, group in detections.groupby("scene_index"):
            labels = (group.sort_values("confidence", ascending=False)
                      .drop_duplicates("label").head(30).label.astype(str).tolist())
            detected_by_scene[int(scene_index)] = labels
    fullres_path = ROOT / "workspace" / "ocr_fullres" / args.video_id / "regions.parquet"
    fullres = pd.read_parquet(fullres_path) if fullres_path.exists() else pd.DataFrame()
    asr_normalized_path = ROOT / "workspace" / "asr_normalization" / args.video_id / "normalized.json"
    asr_segments = (json.loads(asr_normalized_path.read_text(encoding="utf-8")).get("segments", [])
                    if asr_normalized_path.exists() else [])
    output = args.output_dir or ROOT / "workspace" / "scene_documents" / args.video_id
    documents = []
    for scene in scenes.sort_values("scene_index").itertuples(index=False):
        record = gemini.get(str(scene.scene_id), {})
        description = record.get("description", {})
        detected_objects = detected_by_scene.get(int(scene.scene_index), [])
        ocr_lines = description.get("ocr_lines", [])
        ocr_text_raw = unique([item.get("text", "") for item in ocr_lines])
        ocr_text, ocr_filtered = filter_ocr_lines(
            ocr_lines, recurrence=ocr_recurrence, scene_count=scene_count)
        fullres_ocr: list[str] = []
        if not fullres.empty:
            regions = fullres.loc[(fullres.frame_id >= int(scene.start_frame))
                                  & (fullres.frame_id < int(scene.end_frame_exclusive))
                                  & fullres.searchable.astype(bool)]
            text_column = "final_text" if "final_text" in regions else "text_raw"
            fullres_ocr = unique(regions[text_column].astype(str).tolist())
        ocr_text_raw = unique(ocr_text_raw + fullres_ocr)
        ocr_text = unique(ocr_text + fullres_ocr)
        normalized_parts = [str(item["normalized_text"]).strip() for item in asr_segments
                            if float(item["end"]) > float(scene.start_time)
                            and float(item["start"]) < float(scene.end_time)]
        normalized_asr = " ".join(part for part in normalized_parts if part) or str(scene.asr_exact or "")
        document = {
            "schema_version": "0.3.0", "document_id": str(scene.scene_id), "video_id": args.video_id,
            "scene_index": int(scene.scene_index), "start_frame": int(scene.start_frame),
            "end_frame_exclusive": int(scene.end_frame_exclusive), "start_time": float(scene.start_time),
            "end_time": float(scene.end_time), "shot_ids": [int(v) for v in scene.shot_ids],
            "representative_frame_ids": [int(v) for v in scene.representative_frame_ids],
            "asr_text_raw": str(scene.asr_exact or ""), "asr_text": normalized_asr,
            "asr_context": str(scene.asr_context or ""),
            "visual_summary": str(description.get("summary_vi", "")), "ocr_lines": ocr_lines,
            "ocr_text_raw": ocr_text_raw, "ocr_text": ocr_text, "ocr_filtered": ocr_filtered,
            "people": unique(description.get("people", [])),
            "objects": unique(list(description.get("objects", [])) + detected_objects),
            "detected_objects": detected_objects,
            "actions": unique(description.get("actions", [])), "locations": unique(description.get("locations", [])),
            "events": unique(description.get("events", [])),
            "search_keywords": unique(description.get("search_keywords_vi", [])),
            "uncertainties": unique(description.get("uncertainties", [])),
            "has_gemini": bool(record), "has_object_detection": bool(detected_objects),
            "has_fullres_ocr": bool(fullres_ocr),
            "retrieval_text": retrieval_text(scene, description, detected_objects, ocr_text, normalized_asr),
            "provenance": {"semantic_scenes": f"workspace/semantic_scenes/{args.video_id}/scenes.parquet",
                           "gemini": record.get("provenance"), "asr_model": "faster-whisper-large-v3",
                           "asr_normalization": "gemini-conditional" if asr_segments else None,
                           "visual_model": "google/siglip2-so400m-patch14-384",
                           "fullres_ocr_model": "PP-OCRv6-medium-vi" if fullres_ocr else None,
                           "object_model": "yolov8s-worldv2" if detected_objects else None},
        }
        documents.append(document)
    output.mkdir(parents=True, exist_ok=True)
    jsonl = output / "scenes.jsonl"
    jsonl.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents), encoding="utf-8")
    flat = [{k: v for k, v in doc.items() if k not in {"ocr_lines", "ocr_filtered", "provenance"}}
            for doc in documents]
    pd.DataFrame(flat).to_parquet(output / "scenes.parquet", index=False)
    report = {"schema_version": "0.3.0", "video_id": args.video_id, "documents": len(documents),
              "with_gemini": sum(doc["has_gemini"] for doc in documents),
              "with_object_detection": sum(doc["has_object_detection"] for doc in documents),
              "with_asr": sum(bool(doc["asr_text"]) for doc in documents),
              "ocr_raw_lines": sum(len(doc["ocr_text_raw"]) for doc in documents),
              "ocr_searchable_lines": sum(len(doc["ocr_text"]) for doc in documents),
              "ocr_filtered_lines": sum(len(doc["ocr_filtered"]) for doc in documents),
              "ocr_filter_reasons": dict(Counter(
                  row["reason"] for doc in documents for row in doc["ocr_filtered"])),
              "output": str(output)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
