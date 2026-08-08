from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from aic_pipeline.siglip_retrieval import encoder_id_from_model, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Base-selected final frames with a stronger SigLIP model.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--source-encoder-id", default="google_siglip2_base_patch16_224")
    parser.add_argument("--model-id", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = ROOT / "workspace" / "keyframes" / args.source_encoder_id / f"{args.video_id}.parquet"
    metadata = pd.read_parquet(source).sort_values("frame_id").reset_index(drop=True)
    if metadata.empty or metadata.frame_id.duplicated().any():
        raise ValueError("Final keyframe manifest is empty or has duplicate frame IDs")
    videos = pd.read_parquet(ROOT / "workspace" / "manifests" / "videos.parquet")
    row = videos.loc[videos.video_id == args.video_id]
    if len(row) != 1:
        raise ValueError(f"Expected one video row for {args.video_id}")
    video_path = Path(row.iloc[0].video_path)
    target_encoder = encoder_id_from_model(args.model_id)
    output = ROOT / "workspace" / "retrieval" / target_encoder / f"{args.video_id}_final_from_{args.source_encoder_id}"
    if (output / "embeddings.npy").exists() and not args.force:
        raise FileExistsError(f"Output exists: {output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(args.model_id, dtype=dtype, local_files_only=True).to(device).eval()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    targets = metadata.frame_id.astype(int).tolist()
    position = frame_id = decoded = 0
    images: list[Image.Image] = []
    chunks: list[np.ndarray] = []
    started = time.time()

    def encode(batch: list[Image.Image]) -> torch.Tensor:
        try:
            inputs = processor(images=batch, return_tensors="pt").to(device)
            with torch.inference_mode():
                vectors = model.get_image_features(**inputs)
                return torch.nn.functional.normalize(vectors.float(), dim=-1).cpu()
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(batch) // 2
            return torch.cat((encode(batch[:midpoint]), encode(batch[midpoint:])), dim=0)

    def flush() -> None:
        if images:
            chunks.append(encode(images).numpy())
            images.clear()

    try:
        while position < len(targets):
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if frame_id == targets[position]:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                position += 1
                if len(images) >= args.batch_size:
                    flush()
            frame_id += 1
        flush()
    finally:
        capture.release()
    if position != len(targets):
        raise RuntimeError(f"Found {position}/{len(targets)} requested frames")

    embeddings = np.concatenate(chunks)
    config = {
        "pipeline_version": "0.2.0",
        "video_id": args.video_id,
        "model_id": args.model_id,
        "encoder_id": target_encoder,
        "index_kind": "final_keyframes_rerank",
        "source_encoder_id": args.source_encoder_id,
        "source_manifest": str(source),
        "video_path": str(video_path),
        "count": len(metadata),
        "dimension": int(embeddings.shape[1]),
        "normalized": True,
        "similarity": "cosine_via_inner_product",
        "device": str(device),
        "dtype": str(dtype),
        "decoded_frames": decoded,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_index(output, embeddings, metadata, config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
