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
from aic_pipeline.object_spatial import aggregate_scene_objects, object_spatial_text


def unique(values: list[Any]) -> list[str]:
    output, seen = [], set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def load_gemini(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["scene_id"]): row for row in records}


def retrieval_text(scene: Any, description: dict[str, Any], detected_objects: list[str] | None = None,
                   searchable_ocr: list[str] | None = None, normalized_asr: str | None = None,
                   spatial_text: str = "") -> str:
    ocr = searchable_ocr if searchable_ocr is not None else unique(
        [item.get("text", "") for item in description.get("ocr_lines", [])])
    sections = [
        ("Nội dung hình ảnh", [description.get("summary_vi", "")]),
        ("Chữ trong hình", ocr),
        ("Lời nói", [normalized_asr or scene.asr_exact, scene.asr_context]),
        ("Người", description.get("people", [])),
        ("Vật thể", list(description.get("objects", [])) + list(detected_objects or [])),
        ("Đếm vị trí quan hệ vật thể", [spatial_text]),
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
    gemini_report_path = gemini_path.parent / "report.json"
    gemini_report = (json.loads(gemini_report_path.read_text(encoding="utf-8"))
                     if gemini_report_path.exists() else {})
    descriptions = [record.get("description", {}) for record in gemini.values()]
    ocr_recurrence = recurrence_counts(descriptions)
    scene_count = len(scenes)
    detections_path = ROOT / "workspace" / "objects_yoloworld" / args.video_id / "detections.parquet"
    detected_by_scene: dict[int, list[str]] = {}
    spatial_by_scene: dict[int, dict[str, Any]] = {}
    object_rows = 0
    if detections_path.exists():
        detections = pd.read_parquet(detections_path)
        object_rows = len(detections)
        spatial_by_scene = aggregate_scene_objects(detections)
        detections = detections.loc[detections.confidence.ge(0.20)]
        for scene_index, group in detections.groupby("scene_index"):
            labels = (group.sort_values("confidence", ascending=False)
                      .drop_duplicates("label").head(30).label.astype(str).tolist())
            detected_by_scene[int(scene_index)] = labels
    fullres_path = ROOT / "workspace" / "ocr_fullres" / args.video_id / "regions.parquet"
    fullres = pd.read_parquet(fullres_path) if fullres_path.exists() else pd.DataFrame()
    fullres_report_path = fullres_path.parent / "report.json"
    fullres_report = (json.loads(fullres_report_path.read_text(encoding="utf-8"))
                      if fullres_report_path.exists() else {})
    fullres_by_scene: dict[int, pd.DataFrame] = {}
    fullres_all_by_scene: dict[int, pd.DataFrame] = {}
    fullres_recurrence: Counter[str] = Counter()
    if not fullres.empty:
        for scene in scenes.itertuples(index=False):
            scene_regions = fullres.loc[
                (fullres.frame_id >= int(scene.start_frame))
                & (fullres.frame_id < int(scene.end_frame_exclusive))]
            fullres_all_by_scene[int(scene.scene_index)] = scene_regions
            regions = scene_regions.loc[scene_regions.searchable.astype(bool)]
            fullres_by_scene[int(scene.scene_index)] = regions
            fullres_recurrence.update({
                value for value in {
                    str(text).strip().casefold()
                    for text in regions.get("final_text", regions.text_raw).astype(str)
                    if str(text).strip()
                }
            })
    asr_normalized_path = ROOT / "workspace" / "asr_normalization" / args.video_id / "normalized.json"
    asr_segments = (json.loads(asr_normalized_path.read_text(encoding="utf-8")).get("segments", [])
                    if asr_normalized_path.exists() else [])
    output = args.output_dir or ROOT / "workspace" / "scene_documents" / args.video_id
    documents = []
    for scene in scenes.sort_values("scene_index").itertuples(index=False):
        record = gemini.get(str(scene.scene_id), {})
        description = record.get("description", {})
        detected_objects = detected_by_scene.get(int(scene.scene_index), [])
        object_spatial = spatial_by_scene.get(int(scene.scene_index), {"objects": {}, "relations": []})
        spatial_text = object_spatial_text(object_spatial)
        ocr_lines = description.get("ocr_lines", [])
        ocr_text_raw = unique([item.get("text", "") for item in ocr_lines])
        ocr_text, ocr_filtered = filter_ocr_lines(
            ocr_lines, recurrence=ocr_recurrence, scene_count=scene_count)
        ocr_filtered = [{**item, "source": "vlm"} for item in ocr_filtered]
        fullres_ocr: list[str] = []
        fullres_ocr_lines: list[dict[str, Any]] = []
        fullres_ocr_filtered: list[dict[str, str]] = []
        regions = fullres_by_scene.get(int(scene.scene_index), pd.DataFrame())
        if not regions.empty:
            text_column = "final_text" if "final_text" in regions else "text_raw"
            candidates = [{"text": str(row[text_column]), "frame_label": str(row["frame_id"])}
                          for _, row in regions.iterrows()]
            fullres_ocr, fullres_ocr_filtered = filter_ocr_lines(
                candidates, recurrence=fullres_recurrence, scene_count=scene_count)
            accepted = {text.casefold() for text in fullres_ocr}
            fullres_ocr_lines = [
                {"text": str(row[text_column]), "frame_id": int(row["frame_id"]),
                 "pts_time": float(row["pts_time"]), "polygon": json_value(row["polygon"]),
                 "confidence": float(row["confidence"]),
                 "gemini_verified": bool(row.get("gemini_verified", False))}
                for _, row in regions.iterrows()
                if str(row[text_column]).strip().casefold() in accepted
            ]
        all_regions = fullres_all_by_scene.get(int(scene.scene_index), pd.DataFrame())
        fullres_raw = (all_regions["text_raw"].astype(str).tolist()
                       if not all_regions.empty else [])
        ocr_text_raw = unique(ocr_text_raw + fullres_raw)
        ocr_text = unique(ocr_text + fullres_ocr)
        local_fullres_filtered = [
            {"text": str(row["text_raw"]), "reason": str(row["filter_reason"]),
             "frame_label": str(row["frame_id"]), "source": "pp_ocr"}
            for _, row in all_regions.iterrows()
            if not bool(row["searchable"]) and row.get("filter_reason")
        ] if not all_regions.empty else []
        fullres_ocr_filtered = [
            {**item, "source": "pp_ocr"} for item in fullres_ocr_filtered
        ]
        ocr_filtered.extend(local_fullres_filtered)
        ocr_filtered.extend(fullres_ocr_filtered)
        normalized_parts = [str(item["normalized_text"]).strip() for item in asr_segments
                            if float(item["end"]) > float(scene.start_time)
                            and float(item["start"]) < float(scene.end_time)]
        normalized_asr = " ".join(part for part in normalized_parts if part) or str(scene.asr_exact or "")
        document = {
            "schema_version": "0.5.0", "document_id": str(scene.scene_id), "video_id": args.video_id,
            "scene_index": int(scene.scene_index), "start_frame": int(scene.start_frame),
            "end_frame_exclusive": int(scene.end_frame_exclusive), "start_time": float(scene.start_time),
            "end_time": float(scene.end_time), "shot_ids": [int(v) for v in scene.shot_ids],
            "representative_frame_ids": [int(v) for v in scene.representative_frame_ids],
            "asr_text_raw": str(scene.asr_exact or ""), "asr_text": normalized_asr,
            "asr_context": str(scene.asr_context or ""),
            "visual_summary": str(description.get("summary_vi", "")), "ocr_lines": ocr_lines,
            "fullres_ocr_lines": fullres_ocr_lines,
            "ocr_text_raw": ocr_text_raw, "ocr_text": ocr_text, "ocr_filtered": ocr_filtered,
            "people": unique(description.get("people", [])),
            "objects": unique(list(description.get("objects", [])) + detected_objects),
            "detected_objects": detected_objects,
            "object_spatial": object_spatial, "object_spatial_text": spatial_text,
            "actions": unique(description.get("actions", [])), "locations": unique(description.get("locations", [])),
            "events": unique(description.get("events", [])),
            "search_keywords": unique(description.get("search_keywords_vi", [])),
            "uncertainties": unique(description.get("uncertainties", [])),
            "has_gemini": bool(record), "has_vlm_enrichment": bool(record), "has_object_detection": bool(detected_objects),
            "has_fullres_ocr": bool(fullres_ocr),
            "retrieval_text": retrieval_text(scene, description, detected_objects, ocr_text,
                                               normalized_asr, spatial_text),
            "provenance": {"semantic_scenes": f"workspace/semantic_scenes/{args.video_id}/scenes.parquet",
                           "vlm_enrichment": record.get("provenance"), "asr_model": "faster-whisper-large-v3",
                           "asr_normalization": "9router-conditional" if asr_segments else None,
                           "visual_model": "google/siglip2-so400m-patch14-384",
                           "fullres_ocr_model": {
                               "detector": fullres_report.get("det_model"),
                               "recognizer": fullres_report.get("rec_model"),
                           } if fullres_report else None,
                           "object_model": "yolov8s-worldv2" if detected_objects else None},
        }
        documents.append(document)
    output.mkdir(parents=True, exist_ok=True)
    jsonl = output / "scenes.jsonl"
    jsonl.write_text("".join(json.dumps(doc, ensure_ascii=False, default=json_value) + "\n"
                                 for doc in documents), encoding="utf-8")
    flat = [{k: v for k, v in doc.items() if k not in {
        "ocr_lines", "fullres_ocr_lines", "ocr_filtered", "provenance"}}
            for doc in documents]
    pd.DataFrame(flat).to_parquet(output / "scenes.parquet", index=False)
    vlm_scene_ids = set(gemini)
    has_vlm_file = gemini_path.exists()
    vlm_coverage = len(vlm_scene_ids & set(scenes.scene_id.astype(str))) / max(1, len(scenes))
    object_report_path = detections_path.parent / "report.json"
    object_report = (json.loads(object_report_path.read_text(encoding="utf-8"))
                     if object_report_path.exists() else {})
    object_ready = bool(
        detections_path.exists()
        and object_report.get("decoded_frames") == object_report.get("final_frames")
        and object_report.get("schema_version") == "0.2.0"
    )
    ocr_ready = bool(
        fullres_path.exists() and fullres_report.get("complete") is True
        and fullres_report.get("expected_frames", 0) > 0
        and fullres_report.get("decoded_frames") == fullres_report.get("expected_frames")
        and fullres_report.get("det_model") == "PP-OCRv6_medium_det"
        and fullres_report.get("rec_model") == "PP-OCRv6_medium_rec"
    )
    source_status = {
        "semantic_scenes": {
            "available": True, "records": len(scenes),
            "path": f"workspace/semantic_scenes/{args.video_id}/scenes.parquet",
        },
        "vlm_enrichment": {
            "available": (has_vlm_file and vlm_coverage == 1
                          and not gemini_report.get("errors")),
            "file_present": has_vlm_file, "records": len(vlm_scene_ids),
            "scene_coverage": vlm_coverage,
            "path": str(gemini_path),
        },
        "object_detection": {
            "available": object_ready, "file_present": detections_path.exists(),
            "records": object_rows,
            "decoded_frames": object_report.get("decoded_frames", 0),
            "expected_frames": object_report.get("final_frames", 0),
            "path": str(detections_path),
        },
        "fullres_ocr": {
            "available": ocr_ready, "records": len(fullres),
            "decoded_frames": fullres_report.get("decoded_frames", 0),
            "expected_frames": fullres_report.get("expected_frames", 0),
            "path": str(fullres_path),
        },
        "asr_normalization": {
            "available": asr_normalized_path.exists(), "records": len(asr_segments),
            "optional": True,
            "path": str(asr_normalized_path),
        },
    }
    warnings = []
    for name in ("vlm_enrichment", "object_detection", "fullres_ocr"):
        if not source_status[name]["available"]:
            warnings.append(f"{name} input is missing or incomplete")
    if source_status["vlm_enrichment"].get("scene_coverage", 0) < 1:
        warnings.append("VLM enrichment does not cover every semantic scene")
    report = {"schema_version": "0.5.0", "video_id": args.video_id, "documents": len(documents),
              "sources": source_status, "warnings": warnings,
              "with_gemini": sum(doc["has_gemini"] for doc in documents),
              "with_object_detection": sum(doc["has_object_detection"] for doc in documents),
              "with_asr": sum(bool(doc["asr_text"]) for doc in documents),
              "ocr_raw_lines": sum(len(doc["ocr_text_raw"]) for doc in documents),
              "ocr_searchable_lines": sum(len(doc["ocr_text"]) for doc in documents),
              "ocr_filtered_lines": sum(len(doc["ocr_filtered"]) for doc in documents),
              "ocr_filter_reasons": dict(Counter(
                  row["reason"] for doc in documents for row in doc["ocr_filtered"])),
              "ocr_filter_counts_by_source": dict(Counter(
                  row.get("source", "unknown") for doc in documents for row in doc["ocr_filtered"])),
              "output": str(output)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
