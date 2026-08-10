from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed offline-index frames with an independent OpenCLIP model.")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index")
    parser.add_argument("--model", type=Path, default=ROOT / "models/openclip-vit-l14-laion2b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = args.index / "frame_embeddings_openclip.npy"
    if destination.exists() and not args.force:
        raise FileExistsError(f"OpenCLIP index exists: {destination}")
    frames = pd.read_parquet(args.index / "frames.parquet")
    manifest = pd.read_parquet(ROOT / "workspace/manifests/videos.parquet").set_index("video_id")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = CLIPModel.from_pretrained(str(args.model), dtype=dtype, local_files_only=True).to(device).eval()
    dimension = int(model.config.projection_dim)
    matrix = np.empty((len(frames), dimension), dtype=np.float32)
    started, decoded = time.perf_counter(), 0

    def encode(images: list[Image.Image]) -> np.ndarray:
        try:
            inputs = processor(images=images, return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
            with torch.inference_mode():
                values = model.get_image_features(**inputs).float()
                values = torch.nn.functional.normalize(values, dim=-1)
            return values.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(images) == 1:
                raise
            middle = len(images) // 2
            return np.concatenate((encode(images[:middle]), encode(images[middle:])), axis=0)

    for video_id, group in frames.groupby("video_id", sort=False):
        capture = cv2.VideoCapture(str(manifest.loc[video_id].video_path))
        batch_images: list[Image.Image] = []
        batch_rows: list[int] = []

        def flush() -> None:
            nonlocal decoded
            if not batch_images:
                return
            matrix[batch_rows] = encode(batch_images)
            decoded += len(batch_images)
            batch_images.clear()
            batch_rows.clear()

        for row_index, row in group.sort_values("frame_id").iterrows():
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame_id))
            ok, image = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"Could not decode {video_id} frame {row.frame_id}")
            batch_images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            batch_rows.append(int(row_index))
            if len(batch_images) >= args.batch_size:
                flush()
        flush()
        capture.release()
        print(json.dumps({"video_id": video_id, "decoded": len(group)}), flush=True)
    np.save(destination, matrix)
    report = {"schema_version": "0.1.0", "model": str(args.model), "frames": len(frames),
              "decoded_frames": decoded, "dimension": dimension, "device": str(device),
              "elapsed_seconds": time.perf_counter() - started, "output": str(destination)}
    (args.index / "openclip_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
