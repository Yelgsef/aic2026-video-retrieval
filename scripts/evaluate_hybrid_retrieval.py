from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.siglip_retrieval import l2_normalize, load_index
from aic_pipeline.text_embeddings import E5_ENCODER_ID, mean_pool, robust_zscore


def encode_siglip_queries(texts: list[str], model_id: str, device: torch.device) -> np.ndarray:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(model_id, dtype=dtype, local_files_only=True).to(device).eval()
    max_length = int(model.config.text_config.max_position_embeddings)
    inputs = processor(
        text=texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        vectors = model.get_text_features(**inputs).float().cpu().numpy()
    del inputs, model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return l2_normalize(vectors)


def encode_e5_queries(texts: list[str], model_id: str, device: torch.device) -> np.ndarray:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, dtype=dtype, local_files_only=True).to(device).eval()
    inputs = tokenizer(
        ["query: " + text for text in texts], padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        output = model(**inputs)
        vectors = mean_pool(output.last_hidden_state, inputs["attention_mask"])
    vectors = vectors.float().cpu().numpy()
    return l2_normalize(vectors)


def metrics(first_ranks: list[int | None], cutoffs: tuple[int, ...]) -> dict[str, float]:
    reciprocal = [1.0 / rank if rank else 0.0 for rank in first_ranks]
    return {
        "mrr": float(np.mean(reciprocal)),
        **{
            f"recall_at_{cutoff}": float(np.mean([bool(rank and rank <= cutoff) for rank in first_ranks]))
            for cutoff in cutoffs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SigLIP + multilingual E5 object fusion.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--visual-encoder-id", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    query_doc = json.loads(
        (PROJECT_ROOT / "workspace" / "validation" / f"queries_{args.video_id}.json")
        .read_text(encoding="utf-8")
    )
    queries = query_doc["queries"]
    texts = [row["query_text_vi"] for row in queries]

    visual_dir = PROJECT_ROOT / "workspace" / "retrieval" / args.visual_encoder_id / args.video_id
    image_embeddings, frame_metadata, visual_config = load_index(visual_dir)
    siglip_queries = encode_siglip_queries(texts, visual_config["model_id"], device)
    frame_scores = siglip_queries @ np.asarray(image_embeddings).T

    object_dir = PROJECT_ROOT / "workspace" / "retrieval" / E5_ENCODER_ID / args.video_id
    object_embeddings = np.load(object_dir / "embeddings.npy")
    object_metadata = pd.read_parquet(object_dir / "metadata.parquet")
    object_config = json.loads((object_dir / "config.json").read_text(encoding="utf-8"))
    e5_queries = encode_e5_queries(texts, object_config["model_id"], device)
    object_scores = e5_queries @ object_embeddings.T

    shot_table = frame_metadata[["video_id", "shot_id"]].drop_duplicates().sort_values("shot_id")
    shot_ids = shot_table["shot_id"].astype(int).to_numpy()
    frame_shots = frame_metadata["shot_id"].astype(int).to_numpy()
    object_shot_to_col = {int(shot): index for index, shot in enumerate(object_metadata["shot_id"])}
    weights = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
    cutoffs = (1, 5, 10, 20, 50, 100)
    ranks_by_weight: dict[float, list[int | None]] = {weight: [] for weight in weights}
    detail_rows = []

    for query_index, query in enumerate(queries):
        visual_shot_scores = np.array(
            [frame_scores[query_index, frame_shots == shot].max() for shot in shot_ids],
            dtype=np.float32,
        )
        visual_z = robust_zscore(visual_shot_scores)
        object_z = np.zeros(len(shot_ids), dtype=np.float32)
        available = np.array([shot in object_shot_to_col for shot in shot_ids])
        if available.any():
            raw_available = np.array(
                [object_scores[query_index, object_shot_to_col[shot]] for shot in shot_ids[available]],
                dtype=np.float32,
            )
            object_z[available] = robust_zscore(raw_available)
        target = int(query["ground_truth_shot_id"])
        for weight in weights:
            combined = visual_z + weight * object_z
            order = np.argsort(-combined)
            positions = np.flatnonzero(shot_ids[order] == target)
            rank = int(positions[0] + 1) if len(positions) else None
            ranks_by_weight[weight].append(rank)
            detail_rows.append(
                {
                    "query_id": query["query_id"], "query_type": query["query_type"],
                    "object_weight": weight, "first_relevant_rank": rank,
                }
            )

    sweep = {str(weight): metrics(ranks, cutoffs) for weight, ranks in ranks_by_weight.items()}
    best_weight = max(
        weights,
        key=lambda weight: (
            sweep[str(weight)]["recall_at_20"],
            sweep[str(weight)]["recall_at_50"],
            sweep[str(weight)]["mrr"],
        ),
    )
    output_dir = PROJECT_ROOT / "workspace" / "validation"
    details_path = output_dir / f"hybrid_sweep_{args.video_id}.parquet"
    pd.DataFrame(detail_rows).to_parquet(details_path, index=False)
    report = {
        "video_id": args.video_id,
        "visual_model_id": visual_config["model_id"],
        "object_text_model_id": object_config["model_id"],
        "indexed_frames": len(frame_metadata),
        "visual_shots": len(shot_ids),
        "object_covered_shots": len(object_metadata),
        "selection_metric": "Recall@20, then Recall@50, then MRR",
        "best_object_weight": best_weight,
        "best_metrics": sweep[str(best_weight)],
        "weight_sweep": sweep,
        "details": str(details_path),
        "warning": "Weight selected on a small weak-label validation set; confirm on more queries/videos.",
    }
    report_path = output_dir / f"hybrid_retrieval_{args.video_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
