from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.siglip_retrieval import load_index
from aic_pipeline.text_embeddings import robust_zscore
from scripts.evaluate_hybrid_retrieval import encode_siglip_queries, metrics


def shot_max_scores(scores: np.ndarray, frame_shots: np.ndarray, shot_ids: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.array([row[frame_shots == shot].max() for shot in shot_ids]) for row in scores]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate medoid+dense shot score fusion.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    base = PROJECT_ROOT / "workspace" / "retrieval" / args.encoder_id
    coarse_embeddings, coarse_meta, config = load_index(base / args.video_id)
    dense_embeddings, dense_meta, _ = load_index(base / f"{args.video_id}_samples")
    query_doc = json.loads(
        (PROJECT_ROOT / "workspace" / "validation" / f"queries_{args.video_id}.json")
        .read_text(encoding="utf-8")
    )
    queries = query_doc["queries"]
    query_vectors = encode_siglip_queries(
        [row["query_text_vi"] for row in queries], config["model_id"], torch.device(args.device)
    )
    shot_ids = np.sort(coarse_meta["shot_id"].astype(int).unique())
    coarse = shot_max_scores(
        query_vectors @ np.asarray(coarse_embeddings).T,
        coarse_meta["shot_id"].astype(int).to_numpy(), shot_ids,
    )
    dense = shot_max_scores(
        query_vectors @ np.asarray(dense_embeddings).T,
        dense_meta["shot_id"].astype(int).to_numpy(), shot_ids,
    )
    coarse = np.stack([robust_zscore(row) for row in coarse])
    dense = np.stack([robust_zscore(row) for row in dense])

    dense_weights = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    cutoffs = (1, 5, 10, 20, 50, 100)
    sweep = {}
    for weight in dense_weights:
        ranks = []
        combined = coarse + weight * dense
        for index, query in enumerate(queries):
            order = np.argsort(-combined[index])
            positions = np.flatnonzero(shot_ids[order] == int(query["ground_truth_shot_id"]))
            ranks.append(int(positions[0] + 1) if len(positions) else None)
        sweep[str(weight)] = metrics(ranks, cutoffs)
    best_weight = max(
        dense_weights,
        key=lambda weight: (
            sweep[str(weight)]["recall_at_5"],
            sweep[str(weight)]["recall_at_10"],
            sweep[str(weight)]["recall_at_20"],
            sweep[str(weight)]["recall_at_50"],
            sweep[str(weight)]["mrr"],
        ),
    )
    report = {
        "video_id": args.video_id,
        "model_id": config["model_id"],
        "coarse_frames": len(coarse_meta),
        "dense_frames": len(dense_meta),
        "formula": "zscore(coarse_shot_max) + dense_weight * zscore(dense_shot_max)",
        "best_dense_weight": best_weight,
        "best_metrics": sweep[str(best_weight)],
        "weight_sweep": sweep,
        "warning": "Weight selected on 20 weak-label queries; validate on more videos.",
    }
    destination = PROJECT_ROOT / "workspace" / "validation" / f"multiscale_{args.video_id}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
