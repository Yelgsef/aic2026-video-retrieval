from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]


def text_embedding(text: str, model_id: str) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(model_id, dtype=dtype, local_files_only=True).to(device).eval()
    inputs = processor(text=[text], padding="max_length", max_length=64, return_tensors="pt").to(device)
    with torch.inference_mode():
        vector = model.get_text_features(**inputs).float().cpu().numpy()[0]
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def fts_candidates(database: Path, query: str, limit: int) -> list[int]:
    stopwords = {"và", "với", "có", "một", "những", "nhiều", "của", "trong", "trên", "tại", "là"}
    terms = [term.replace('"', "") for term in query.casefold().split()
             if term.strip() and term.casefold() not in stopwords]
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    with sqlite3.connect(database) as connection:
        return [int(row[0]) - 1 for row in connection.execute(
            "SELECT rowid FROM scene_fts WHERE scene_fts MATCH ? ORDER BY bm25(scene_fts) LIMIT ?",
            (expression, limit),
        )]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Query the offline SigLIP2 + FTS5 hybrid index.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace" / "offline_index")
    parser.add_argument("--model-id", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--visual-weight", type=float, default=2.0)
    parser.add_argument("--text-weight", type=float, default=0.25)
    args = parser.parse_args()

    table = pd.read_parquet(args.index / "scenes.parquet")
    embeddings = np.load(args.index / "frame_embeddings.npy", mmap_mode="r")
    frames = pd.read_parquet(args.index / "frames.parquet")
    query = text_embedding(args.query, args.model_id)
    frame_scores = np.asarray(embeddings @ query, dtype=np.float32)
    visual_scores = np.full(len(table), -np.inf, dtype=np.float32)
    np.maximum.at(visual_scores, frames.scene_row.to_numpy(dtype=np.int64), frame_scores)
    visual_order = np.argsort(-visual_scores)[: min(args.candidate_k, len(table))]
    text_order = fts_candidates(args.index / "search.sqlite3", args.query, args.candidate_k)

    fused: dict[int, float] = {}
    for rank, row in enumerate(visual_order, 1):
        fused[int(row)] = fused.get(int(row), 0.0) + args.visual_weight / (60 + rank)
    for rank, row in enumerate(text_order, 1):
        fused[int(row)] = fused.get(int(row), 0.0) + args.text_weight / (60 + rank)
    order = sorted(fused, key=fused.get, reverse=True)[: args.top_k]
    results = []
    for rank, row_id in enumerate(order, 1):
        row = table.iloc[row_id]
        results.append({"rank": rank, "fusion_score": fused[row_id], "visual_score": float(visual_scores[row_id]),
                        "video_id": row.video_id, "scene_index": int(row.scene_index),
                        "start_time": float(row.start_time), "end_time": float(row.end_time),
                        "representative_frame_ids": [int(value) for value in row.representative_frame_ids],
                        "asr_text": "" if pd.isna(row.asr_text) else str(row.asr_text)})
    print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
