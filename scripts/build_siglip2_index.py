from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.siglip_retrieval import MODEL_ID, encoder_id_from_model, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an exact SigLIP 2 image retrieval index.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    manifest_path = PROJECT_ROOT / "workspace" / "keyframe_images" / args.video_id / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path).sort_values("frame_id").reset_index(drop=True)
    if manifest.empty:
        raise ValueError(f"Empty keyframe manifest: {manifest_path}")
    missing = [p for p in manifest["image_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")

    encoder_id = encoder_id_from_model(args.model_id)
    output_dir = PROJECT_ROOT / "workspace" / "retrieval" / encoder_id / args.video_id
    if (output_dir / "embeddings.npy").exists() and not args.force:
        raise FileExistsError(f"Index exists: {output_dir}. Pass --force to replace it.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading {args.model_id} on {device} ({dtype})", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_id, local_files_only=not args.allow_download, use_fast=False
    )
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, local_files_only=not args.allow_download
    ).to(device).eval()

    started = time.time()
    chunks: list[np.ndarray] = []
    for start in range(0, len(manifest), args.batch_size):
        batch = manifest.iloc[start : start + args.batch_size]
        images = []
        for image_path in batch["image_path"]:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        chunks.append(features.cpu().numpy())
        done = min(start + args.batch_size, len(manifest))
        print(f"Embedded {done:,}/{len(manifest):,}", flush=True)

    embeddings = np.concatenate(chunks, axis=0)
    config = {
        "model_id": args.model_id,
        "encoder_id": encoder_id,
        "video_id": args.video_id,
        "count": len(manifest),
        "dimension": int(embeddings.shape[1]),
        "normalized": True,
        "similarity": "cosine_via_inner_product",
        "device": str(device),
        "dtype": str(dtype),
        "source_manifest": str(manifest_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_index(output_dir, embeddings, manifest, config)
    print(f"Saved {len(manifest):,} vectors to {output_dir}")


if __name__ == "__main__":
    main()
