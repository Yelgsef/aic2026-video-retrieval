from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate SigLIP 2 text-to-keyframe retrieval.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--encoder-id", default=ENCODER_ID)
    parser.add_argument("--index-name", help="Index directory name; defaults to video-id")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--query-batch-size", type=int, default=20)
    parser.add_argument("--vi-only", action="store_true", help="Disable English query fusion")
    args = parser.parse_args()

    query_path = PROJECT_ROOT / "workspace" / "validation" / f"queries_{args.video_id}.json"
    query_document = json.loads(query_path.read_text(encoding="utf-8"))
    queries = query_document["queries"]
    translation_path = (
        PROJECT_ROOT / "workspace" / "validation" / f"query_translations_{args.video_id}.json"
    )
    translations = (
        json.loads(translation_path.read_text(encoding="utf-8"))
        if translation_path.exists() and not args.vi_only
        else {}
    )
    index_name = args.index_name or args.video_id
    index_dir = PROJECT_ROOT / "workspace" / "retrieval" / args.encoder_id / index_name
    image_embeddings, metadata, config = load_index(index_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        config["model_id"], local_files_only=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        config["model_id"], dtype=dtype, local_files_only=True
    ).to(device).eval()
    max_text_length = int(model.config.text_config.max_position_embeddings)

    flat_texts = []
    variant_slices = []
    for row in queries:
        variants = [row["query_text_vi"]]
        if row["query_id"] in translations:
            variants.append(translations[row["query_id"]])
        start = len(flat_texts)
        flat_texts.extend(variants)
        variant_slices.append(slice(start, len(flat_texts)))

    query_vectors = []
    for start in range(0, len(flat_texts), args.query_batch_size):
        texts = flat_texts[start : start + args.query_batch_size]
        inputs = processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            vectors = model.get_text_features(**inputs).float().cpu().numpy()
        query_vectors.append(vectors)
    query_vectors = l2_normalize(np.concatenate(query_vectors, axis=0))
    variant_scores = query_vectors @ np.asarray(image_embeddings).T

    cutoffs = (1, 5, 10, 20, 50, 100)
    rows = []
    ranking_rows = []
    shot_ids = metadata["shot_id"].to_numpy()
    for query_index, query in enumerate(queries):
        fused_scores = fuse_variant_scores(
            variant_scores[variant_slices[query_index]], "zscore_max"
        )
        order = np.argsort(-fused_scores)
        target_shot = int(query["ground_truth_shot_id"])
        frame_positions = np.flatnonzero(shot_ids[order] == target_shot)
        first_frame_rank = int(frame_positions[0] + 1) if len(frame_positions) else None
        dedup = rank_with_shot_dedup(fused_scores, metadata, top_k=len(metadata))
        shot_positions = np.flatnonzero(dedup["shot_id"].to_numpy() == target_shot)
        first_rank = int(shot_positions[0] + 1) if len(shot_positions) else None
        row = {
            "query_id": query["query_id"],
            "query_type": query["query_type"],
            "query_text_vi": query["query_text_vi"],
            "ground_truth_shot_id": target_shot,
            "first_relevant_frame_rank": first_frame_rank,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        }
        for cutoff in cutoffs:
            row[f"hit_at_{cutoff}"] = bool(first_rank and first_rank <= cutoff)
        rows.append(row)
        for rank, index_row in enumerate(order[:100], start=1):
            item = metadata.iloc[int(index_row)]
            ranking_rows.append(
                {
                    "query_id": query["query_id"],
                    "rank": rank,
                    "score": float(fused_scores[index_row]),
                    "is_relevant_shot": bool(int(item.shot_id) == target_shot),
                    "video_id": item.video_id,
                    "shot_id": int(item.shot_id),
                    "frame_id": int(item.frame_id),
                    "pts_time": float(item.pts_time),
                    "image_path": item.get("image_path"),
                }
            )

    results = pd.DataFrame(rows)
    rankings = pd.DataFrame(ranking_rows)
    validation_dir = PROJECT_ROOT / "workspace" / "validation"
    mode_id = "vi_only" if not translations else "vi_en_zscore_max"
    artifact_id = f"{args.encoder_id}_{index_name}_{mode_id}"
    result_path = validation_dir / f"{artifact_id}_{args.video_id}.parquet"
    ranking_path = validation_dir / f"{artifact_id}_rankings_{args.video_id}.parquet"
    results.to_parquet(result_path, index=False)
    rankings.to_parquet(ranking_path, index=False)
    report = {
        "video_id": args.video_id,
        "model_id": config["model_id"],
        "query_language": "vi",
        "query_mode": "vi_en_zscore_max_fusion" if translations else "vi_only",
        "query_count": len(results),
        "indexed_keyframes": len(metadata),
        "device": str(device),
        "mrr": float(results["reciprocal_rank"].mean()),
        **{f"recall_at_{k}": float(results[f"hit_at_{k}"].mean()) for k in cutoffs},
        "per_query_results": str(result_path),
        "top_100_rankings": str(ranking_path),
        "warning": query_document.get("warning"),
    }
    report_path = validation_dir / f"{artifact_id}_{args.video_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nPer-query first relevant deduplicated-shot rank:")
    print(results[["query_id", "query_type", "first_relevant_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
