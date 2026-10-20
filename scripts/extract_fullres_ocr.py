from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# The Windows Paddle wheel expects cuDNN on PATH. The Python cuDNN package
# keeps DLLs inside site-packages, so register that directory explicitly.
if sys.platform == "win32":
    for site in map(Path, sys.path):
        for component in ("cublas", "cudnn"):
            dlls = site / "nvidia" / component / "bin"
            if dlls.exists():
                os.add_dll_directory(str(dlls))
                os.environ["PATH"] = str(dlls) + os.pathsep + os.environ.get("PATH", "")

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.ocr_filter import filter_ocr_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PP-OCR on original-resolution final keyframes.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--gemini-threshold", type=float, default=0.78)
    parser.add_argument("--max-gemini-crops", type=int, default=80)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=0,
                        help="OCR every Nth source frame instead of final keyframes")
    parser.add_argument("--output-root", type=Path,
                        help="Alternative output root for audits; defaults to workspace/ocr_fullres")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--det-model", default="PP-OCRv6_medium_det")
    parser.add_argument("--rec-model", default="PP-OCRv6_medium_rec")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_gemini_crops < 0:
        parser.error("--max-gemini-crops cannot be negative")

    output_root = args.output_root or (ROOT / "workspace" / "ocr_fullres")
    output = output_root / args.video_id
    result_path = output / "regions.parquet"
    if result_path.exists() and not args.force:
        report_path = output / "report.json"
        previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        if (previous.get("complete") is True
                and previous.get("decoded_frames") == previous.get("expected_frames")
                and previous.get("det_model") == args.det_model
                and previous.get("rec_model") == args.rec_model):
            print(json.dumps({"video_id": args.video_id, "status": "exists", "output": str(result_path)}))
            return
    manifest = pd.read_parquet(ROOT / "workspace/manifests/videos.parquet")
    match = manifest.loc[manifest.video_id == args.video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one manifest row for {args.video_id}")
    video_path = Path(match.iloc[0].video_path)
    if args.frame_stride:
        if args.frame_stride < 1:
            raise ValueError("frame-stride must be positive")
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        frame_ids = list(range(0, frame_count, args.frame_stride))
        frames = pd.DataFrame({"frame_id": frame_ids,
                               "pts_time": [value / fps for value in frame_ids]})
    else:
        index = ROOT / "workspace/retrieval/google_siglip2_so400m_patch14_384" / (
            f"{args.video_id}_final_from_google_siglip2_base_patch16_224")
        frames = pd.read_parquet(index / "metadata.parquet").sort_values("frame_id")
    if args.limit_frames:
        frames = frames.head(args.limit_frames)

    scenes = pd.read_parquet(ROOT / "workspace/semantic_scenes" / args.video_id / "scenes.parquet")
    scenes = scenes.sort_values("start_frame")
    starts = scenes.start_frame.to_numpy(dtype="int64")
    ends = scenes.end_frame_exclusive.to_numpy(dtype="int64")
    scene_indices = scenes.scene_index.to_numpy(dtype="int64")

    def scene_for_frame(frame_id: int) -> int | None:
        position = int(starts.searchsorted(frame_id, side="right")) - 1
        if position >= 0 and frame_id < ends[position]:
            return int(scene_indices[position])
        return None

    from paddleocr import PaddleOCR
    engine = PaddleOCR(text_detection_model_name=args.det_model,
                       text_recognition_model_name=args.rec_model,
                       use_doc_orientation_classify=False, use_doc_unwarping=False,
                       use_textline_orientation=False, text_rec_score_thresh=args.confidence)
    output.mkdir(parents=True, exist_ok=True)
    crop_dir = output / "gemini_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    candidates: list[dict] = []

    def process_batch(images: list, metadata_rows: list) -> None:
        for image, frame, result in zip(images, metadata_rows, engine.predict(images)):
            frame_id = int(frame.frame_id)
            for region_index, (text, score, polygon) in enumerate(zip(
                    result.get("rec_texts", []), result.get("rec_scores", []), result.get("rec_polys", []))):
                text = str(text).strip(); score = float(score)
                searchable, rejected = filter_ocr_lines([{"text": text, "frame_label": str(frame_id)}])
                points = [[int(x), int(y)] for x, y in polygon]
                xs, ys = [p[0] for p in points], [p[1] for p in points]; pad = 12
                x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
                x2, y2 = min(image.shape[1], max(xs) + pad), min(image.shape[0], max(ys) + pad)
                scene_index = scene_for_frame(frame_id)
                row = {"video_id": args.video_id, "scene_index": scene_index,
                       "frame_id": frame_id, "pts_time": float(frame.pts_time),
                       "region_index": region_index, "text_raw": text, "confidence": score,
                       "polygon": points, "searchable": bool(searchable),
                       "filter_reason": rejected[0]["reason"] if rejected else None,
                       "needs_gemini": False, "crop_path": None}
                if searchable and score < args.gemini_threshold and x2 > x1 and y2 > y1:
                    crop_path = crop_dir / f"f{frame_id:09d}_r{region_index:03d}.jpg"
                    if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2],
                                       [cv2.IMWRITE_JPEG_QUALITY, 94]):
                        raise OSError(f"Could not write OCR crop: {crop_path}")
                    row["needs_gemini"], row["crop_path"] = True, str(crop_path)
                    candidates.append({"scene_index": scene_index, "frame_id": frame_id,
                                       "region_index": region_index,
                                       "local_text": text, "confidence": score, "crop_path": str(crop_path)})
                rows.append(row)

    # Sequential decode avoids hundreds of expensive H.264 random seeks.
    wanted = {int(row.frame_id): row for row in frames.itertuples(index=False)}
    last_frame = max(wanted) if wanted else -1
    capture = cv2.VideoCapture(str(video_path)); batch_images, batch_rows = [], []
    frame_id = 0
    decoded_frames = 0
    while frame_id <= last_frame:
        ok, image = capture.read()
        if not ok:
            break
        if frame_id in wanted:
            decoded_frames += 1
            batch_images.append(image); batch_rows.append(wanted[frame_id])
            if len(batch_images) >= args.batch_size:
                process_batch(batch_images, batch_rows); batch_images, batch_rows = [], []
        frame_id += 1
    if batch_images:
        process_batch(batch_images, batch_rows)
    capture.release()
    # Spend Gemini only on the least confident useful regions.
    by_scene: dict[int | None, list[dict]] = {}
    for candidate in candidates:
        by_scene.setdefault(candidate["scene_index"], []).append(candidate)
    for values in by_scene.values():
        values.sort(key=lambda x: x["confidence"])
    selected = []
    # Round-robin scenes so text-heavy scenes do not consume the full crop budget.
    while len(selected) < args.max_gemini_crops:
        added = False
        for values in by_scene.values():
            if values:
                selected.append(values.pop(0))
                added = True
                if len(selected) >= args.max_gemini_crops:
                    break
        if not added:
            break
    allowed = {(x["frame_id"], x["region_index"]) for x in selected}
    for row in rows:
        if row["needs_gemini"] and (row["frame_id"], row["region_index"]) not in allowed:
            row["needs_gemini"] = False
    pd.DataFrame(rows).to_parquet(result_path, index=False)
    (output / "gemini_candidates.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
    report = {"video_id": args.video_id, "det_model": args.det_model, "rec_model": args.rec_model,
              "frame_stride": args.frame_stride or None,
              "expected_frames": len(frames), "decoded_frames": decoded_frames,
              "complete": decoded_frames == len(frames), "regions": len(rows),
              "searchable_regions": sum(x["searchable"] for x in rows),
              "filtered_regions": sum(not x["searchable"] for x in rows),
              "gemini_candidates": len(selected), "output": str(result_path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise RuntimeError(f"Decoded {decoded_frames}/{len(frames)} requested OCR frames")


if __name__ == "__main__":
    main()
