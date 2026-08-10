from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLOWorld


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_CLASSES = [
    "person", "face", "helmet", "hat", "glasses", "mask", "shirt", "jacket", "uniform",
    "bag", "backpack", "handbag", "phone", "tablet", "laptop", "book", "paper", "sign",
    "microphone", "camera", "television", "computer monitor", "traffic light", "traffic sign",
    "license plate", "car", "motorcycle", "bicycle", "bus", "truck", "boat", "airplane", "train",
    "chair", "table", "desk", "bed", "sofa", "door", "window", "building", "house", "bridge",
    "tree", "flower", "plant", "road", "beach", "umbrella", "bottle", "cup", "bowl", "plate",
    "food", "knife", "scissors", "clock", "flag", "ball", "dog", "cat", "bird", "fish",
    "turtle", "horse", "cow", "buffalo", "elephant", "fire", "smoke", "debris", "wood log",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-World detection on final AIC keyframes.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/yolo-world/yolov8s-worldv2.pt")
    parser.add_argument("--confidence", type=float, default=0.12)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"YOLO-World checkpoint is missing: {args.checkpoint}")
    output_dir = ROOT / "workspace/objects_yoloworld" / args.video_id
    destination = output_dir / "detections.parquet"
    if destination.exists() and not args.force:
        raise FileExistsError(f"Object detections already exist: {destination}")
    manifest = pd.read_parquet(ROOT / "workspace/manifests/videos.parquet").set_index("video_id")
    video_path = str(manifest.loc[args.video_id].video_path)
    metadata_path = (ROOT / "workspace/retrieval/google_siglip2_so400m_patch14_384"
                     / f"{args.video_id}_final_from_google_siglip2_base_patch16_224/metadata.parquet")
    frames_meta = pd.read_parquet(metadata_path).sort_values("frame_id").reset_index(drop=True)
    scenes = pd.read_parquet(ROOT / "workspace/semantic_scenes" / args.video_id / "scenes.parquet").sort_values("scene_index")
    scene_ends = scenes.end_frame_exclusive.to_numpy()
    scene_indices = scenes.scene_index.to_numpy()
    frame_to_scene = {int(frame_id): int(scene_indices[min(np.searchsorted(scene_ends, frame_id, side="right"),
                                                          len(scene_indices) - 1)])
                      for frame_id in frames_meta.frame_id}
    model = YOLOWorld(str(args.checkpoint))
    model.set_classes(args.classes)
    capture = cv2.VideoCapture(video_path)
    rows, decoded, started = [], 0, time.perf_counter()
    crop_dir = output_dir / "crops"
    for start in range(0, len(frames_meta), args.batch_size):
        batch_meta = frames_meta.iloc[start:start + args.batch_size]
        images, valid_meta = [], []
        for frame in batch_meta.itertuples(index=False):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame.frame_id))
            ok, image = capture.read()
            if ok:
                images.append(image)
                valid_meta.append(frame)
                decoded += 1
        if not images:
            continue
        results = model.predict(images, conf=args.confidence, iou=args.iou,
                                max_det=args.max_detections, device=0, verbose=False)
        for image, frame, result in zip(images, valid_meta, results):
            height, width = image.shape[:2]
            boxes = result.boxes
            for detection_index in range(len(boxes)):
                box = boxes.xyxy[detection_index].detach().cpu().tolist()
                confidence = float(boxes.conf[detection_index].detach().cpu())
                class_id = int(boxes.cls[detection_index].detach().cpu())
                label = str(result.names[class_id])
                x1, y1, x2, y2 = box
                crop_path = None
                if args.save_crops:
                    left, top = max(0, int(x1)), max(0, int(y1))
                    right, bottom = min(width, int(x2 + 0.999)), min(height, int(y2 + 0.999))
                    if right > left and bottom > top:
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = crop_dir / f"f{int(frame.frame_id):08d}_d{detection_index:03d}.jpg"
                        cv2.imwrite(str(crop_path), image[top:bottom, left:right],
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                rows.append({"video_id": args.video_id, "frame_id": int(frame.frame_id),
                             "pts_time": float(frame.pts_time), "scene_index": frame_to_scene[int(frame.frame_id)],
                             "shot_id": int(frame.shot_id), "detection_index": detection_index,
                             "label": label, "confidence": confidence,
                             "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                             "x1_norm": float(x1 / width), "y1_norm": float(y1 / height),
                             "x2_norm": float(x2 / width), "y2_norm": float(y2 / height),
                             "frame_width": width, "frame_height": height,
                             "crop_path": str(crop_path) if crop_path else None})
    capture.release()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["video_id", "frame_id", "pts_time", "scene_index", "shot_id",
                                            "detection_index", "label", "confidence", "x1", "y1", "x2", "y2",
                                            "x1_norm", "y1_norm", "x2_norm", "y2_norm", "frame_width", "frame_height",
                                            "crop_path"])
    table.to_parquet(destination, index=False)
    report = {"schema_version": "0.1.0", "video_id": args.video_id,
              "model": str(args.checkpoint), "classes": args.classes,
              "final_frames": len(frames_meta), "decoded_frames": decoded,
              "detections": len(table), "frames_with_detections": int(table.frame_id.nunique()) if len(table) else 0,
              "confidence": args.confidence, "iou": args.iou,
              "elapsed_seconds": time.perf_counter() - started, "output": str(destination)}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
