from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from aic_pipeline.task_router import route_query


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Query the canonical offline multi-model retrieval engine.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--rerank-k", type=int, default=30)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--fusion-mode", choices=("rrf", "normalized"), default="normalized")
    parser.add_argument("--field-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--openclip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-pooling", choices=("max", "top3_mean", "max_top3"), default="max")
    parser.add_argument("--calibrate-openclip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker-fusion", choices=("rank", "score", "hybrid", "auto"), default="auto")
    parser.add_argument("--device")
    args = parser.parse_args()
    engine = OfflineSearchEngine(ROOT, args.index, args.device)
    task = route_query(args.query)
    calibrate_openclip = task.task == "KIS" if args.calibrate_openclip is None else args.calibrate_openclip
    result = engine.search(
        args.query, top_k=args.top_k, candidate_k=args.candidate_k,
        rerank_k=0 if args.no_reranker else args.rerank_k,
        rerank_batch_size=args.rerank_batch_size,
        fusion_mode=args.fusion_mode, field_aware=args.field_aware,
        openclip=args.openclip, visual_pooling=args.visual_pooling,
        calibrate_openclip=calibrate_openclip, reranker_fusion=args.reranker_fusion,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
