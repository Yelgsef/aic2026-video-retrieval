from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.siglip_retrieval import (
    ENCODER_ID,
    fuse_variant_scores,
    l2_normalize,
    load_index,
    rank_with_shot_dedup,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a SigLIP 2 keyframe index with text.")
    parser.add_argument("query")
    parser.add_argument("--query-en", help="Optional English translation of the query")
    parser.add_argument("--variant", action="append", default=[], help="Additional visual query; repeatable")
    parser.add_argument(
        "--fusion", choices=["zscore_max", "max", "mean"], default="zscore_max"
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--encoder-id", default=ENCODER_ID)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    index_dir = PROJECT_ROOT / "workspace" / "retrieval" / args.encoder_id / args.video_id
    embeddings, metadata, config = load_index(index_dir)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        config["model_id"], local_files_only=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        config["model_id"], dtype=dtype, local_files_only=True
    ).to(device).eval()
    max_text_length = int(model.config.text_config.max_position_embeddings)
    variants = [args.query]
    if args.query_en:
        variants.append(args.query_en)
    variants.extend(args.variant)
    variants = list(dict.fromkeys(text.strip() for text in variants if text.strip()))
    inputs = processor(
        text=variants,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        query_embeddings = model.get_text_features(**inputs).float().cpu().numpy()

    query_embeddings = l2_normalize(query_embeddings)
    scores = fuse_variant_scores(query_embeddings @ np.asarray(embeddings).T, args.fusion)
    results = rank_with_shot_dedup(scores, metadata, args.top_k)
    columns = [c for c in ["rank", "score", "video_id", "shot_id", "frame_id", "pts_time", "image_path"] if c in results]
    pd.set_option("display.max_colwidth", 120)
    print(results[columns].to_string(index=False))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(args.save, index=False)
        print(f"Saved results to {args.save}")


if __name__ == "__main__":
    main()
