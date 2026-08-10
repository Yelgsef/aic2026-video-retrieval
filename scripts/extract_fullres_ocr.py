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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = ROOT / "workspace" / "ocr_fullres" / args.video_id
    result_path = output / "regions.parquet"
    if result_path.exists() and not args.force:
        report_path = output / "report.json"
        previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        if previous.get("complete") is True:
            print(json.dumps({"video_id": args.video_id, "status": "exists", "output": str(result_path)}))
            return
    manifest = pd.read_parquet(ROOT / "workspace/manifests/videos.parquet")
    match = manifest.loc[manifest.video_id == args.video_id]
    if len(match) != 1:
        raise ValueError(f"Expected one manifest row for {args.video_id}")
    video_path = Path(match.iloc[0].video_path)
    index = ROOT / "workspace/retrieval/google_siglip2_so400m_patch14_384" / (
        f"{args.video_id}_final_from_google_siglip2_base_patch16_224")
    frames = pd.read_parquet(index / "metadata.parquet").sort_values("frame_id")
    source_frame_count = len(frames)
    if args.limit_frames:
        frames = frames.head(args.limit_frames)

    from paddleocr import PaddleOCR
    engine = PaddleOCR(lang="vi", use_doc_orientation_classify=False, use_doc_unwarping=False,
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
                row = {"video_id": args.video_id, "frame_id": frame_id, "pts_time": float(frame.pts_time),
                       "region_index": region_index, "text_raw": text, "confidence": score,
                       "polygon": points, "searchable": bool(searchable),
                       "filter_reason": rejected[0]["reason"] if rejected else None,
                       "needs_gemini": False, "crop_path": None}
                if searchable and score < args.gemini_threshold and x2 > x1 and y2 > y1:
                    crop_path = crop_dir / f"f{frame_id:09d}_r{region_index:03d}.jpg"
                    cv2.imwrite(str(crop_path), image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 94])
                    row["needs_gemini"], row["crop_path"] = True, str(crop_path)
                    candidates.append({"frame_id": frame_id, "region_index": region_index,
                                       "local_text": text, "confidence": score, "crop_path": str(crop_path)})
                rows.append(row)

    # Sequential decode avoids hundreds of expensive H.264 random seeks.
    wanted = {int(row.frame_id): row for row in frames.itertuples(index=False)}
    last_frame = max(wanted) if wanted else -1
    capture = cv2.VideoCapture(str(video_path)); batch_images, batch_rows = [], []
    frame_id = 0
    while frame_id <= last_frame:
        ok, image = capture.read()
        if not ok:
            break
        if frame_id in wanted:
            batch_images.append(image); batch_rows.append(wanted[frame_id])
            if len(batch_images) >= args.batch_size:
                process_batch(batch_images, batch_rows); batch_images, batch_rows = [], []
        frame_id += 1
    if batch_images:
        process_batch(batch_images, batch_rows)
    capture.release()
    # Spend Gemini only on the least confident useful regions.
    allowed = {(x["frame_id"], x["region_index"]) for x in
               sorted(candidates, key=lambda x: x["confidence"])[:args.max_gemini_crops]}
    for row in rows:
        if row["needs_gemini"] and (row["frame_id"], row["region_index"]) not in allowed:
            row["needs_gemini"] = False
    selected = [x for x in candidates if (x["frame_id"], x["region_index"]) in allowed]
    pd.DataFrame(rows).to_parquet(result_path, index=False)
    (output / "gemini_candidates.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
    report = {"video_id": args.video_id, "frames": len(frames), "source_frames": source_frame_count,
              "complete": len(frames) == source_frame_count, "regions": len(rows),
              "searchable_regions": sum(x["searchable"] for x in rows),
              "filtered_regions": sum(not x["searchable"] for x in rows),
              "gemini_candidates": len(selected), "output": str(result_path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
