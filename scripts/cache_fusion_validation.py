from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.fusion_profiles import classify_retrieval_query
from aic_pipeline.offline_search_engine import OfflineSearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache per-component retrieval scores for fusion tuning.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    scenes = pd.read_parquet(args.index / "scenes.parquet")
    indexed = set(scenes.video_id.astype(str))
    labels = labels.loc[labels.video_id.astype(str).isin(indexed)]
    completed = {}
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        completed = {row["query_id"]: row for row in old.get("queries", [])}
    engine = OfflineSearchEngine(ROOT, args.index)
    rows = []
    for label in labels.itertuples(index=False):
        query_id = str(label.query_id)
        if query_id in completed:
            rows.append(completed[query_id]); continue
        started = time.perf_counter()
        response = engine.search(str(label.query_text_vi), top_k=args.top_k,
                                 candidate_k=args.candidate_k, rerank_k=args.rerank_k,
                                 fusion_mode="normalized", field_aware=True, openclip=True,
                                 visual_pooling="max", calibrate_openclip=False,
                                 reranker_fusion="rank")
        candidates = [{key: item.get(key) for key in (
            "video_id", "scene_index", "start_frame", "end_frame_exclusive",
            "fusion_score", "visual_score", "openclip_score", "semantic_score",
            "rapidfuzz_score", "field_coverage_score", "object_spatial_score", "reranker_score")}
                      for item in response["results"]]
        row = {"query_id": query_id, "query_type": str(label.query_type),
               "profile_kind": classify_retrieval_query(str(label.query_text_vi)),
               "query": str(label.query_text_vi), "video_id": str(label.video_id),
               "start_frame": int(label.start_frame),
               "end_frame_exclusive": int(label.end_frame_exclusive),
               "latency_seconds": time.perf_counter() - started, "candidates": candidates}
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"partial": True, "queries": rows}, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"query_id": query_id, "profile": row["profile_kind"],
                          "candidates": len(candidates)}, ensure_ascii=False), flush=True)
    report = {"partial": False, "index": str(args.index), "count": len(rows),
              "top_k": args.top_k, "candidate_k": args.candidate_k,
              "rerank_k": args.rerank_k, "queries": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
