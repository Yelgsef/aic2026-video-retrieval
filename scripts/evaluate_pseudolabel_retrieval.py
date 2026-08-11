from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=ROOT / "workspace" / "validation" / "review_pack_50" / "annotations_autolabeled.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "validation" / "pseudolabel_retrieval_eval.json")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace" / "offline_index")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task", choices=("KIS", "QA"), help="Optionally evaluate one task only")
    parser.add_argument("--fusion-mode", choices=("rrf", "normalized"), default="rrf")
    parser.add_argument("--field-aware", action="store_true")
    parser.add_argument("--openclip", action="store_true")
    parser.add_argument("--visual-pooling", choices=("max", "top3_mean", "max_top3"), default="max")
    parser.add_argument("--calibrate-openclip", action="store_true")
    parser.add_argument("--reranker-fusion", choices=("rank", "score", "hybrid", "auto"), default="rank")
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    if args.task:
        labels = labels.loc[labels.query_type.eq(args.task)]
    indexed_videos = set(pd.read_parquet(args.index / "scenes.parquet").video_id.astype(str))
    skipped = labels.loc[~labels.video_id.astype(str).isin(indexed_videos)]
    labels = labels.loc[labels.video_id.astype(str).isin(indexed_videos)]
    if args.limit:
        labels = labels.head(args.limit)
    engine = OfflineSearchEngine(ROOT, args.index)
    rows = []
    for label in labels.itertuples(index=False):
        started = time.perf_counter()
        response = engine.search(str(label.query_text_vi), top_k=100, rerank_k=30,
                                 fusion_mode=args.fusion_mode, field_aware=args.field_aware,
                                 openclip=args.openclip, visual_pooling=args.visual_pooling,
                                 calibrate_openclip=args.calibrate_openclip,
                                 reranker_fusion=args.reranker_fusion)
        scene_id = str(getattr(label, "scene_id", "")).strip()
        if scene_id and scene_id.casefold() != "nan":
            expected = int(scene_id.rsplit(":", 1)[-1])
            matches = [str(item["video_id"]) == str(label.video_id)
                       and int(item["scene_index"]) == expected for item in response["results"]]
        else:
            expected = None
            start, end = int(label.start_frame), int(label.end_frame_exclusive)
            matches = [str(item["video_id"]) == str(label.video_id)
                       and int(item["start_frame"]) < end
                       and int(item["end_frame_exclusive"]) > start
                       for item in response["results"]]
        rank = matches.index(True) + 1 if any(matches) else None
        rows.append({"query_id": label.query_id, "query_type": label.query_type, "expected_scene": expected,
                     "expected_video_id": str(label.video_id),
                     "expected_start_frame": int(getattr(label, "start_frame", -1)),
                     "expected_end_frame_exclusive": int(getattr(label, "end_frame_exclusive", -1)),
                     "rank": rank, "latency_seconds": time.perf_counter() - started})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    by_task = {}
    for task in sorted({row["query_type"] for row in rows}):
        selected = [row for row in rows if row["query_type"] == task]
        by_task[task] = {"count": len(selected),
                         "mrr": sum(1 / row["rank"] if row["rank"] else 0 for row in selected) / len(selected),
                         "recall_at_1": sum(row["rank"] == 1 for row in selected) / len(selected),
                         "recall_at_5": sum(row["rank"] is not None and row["rank"] <= 5 for row in selected) / len(selected),
                         "recall_at_10": sum(row["rank"] is not None and row["rank"] <= 10 for row in selected) / len(selected)}
    report = {"pseudo_labels": True, "fusion_mode": args.fusion_mode, "field_aware": args.field_aware,
              "openclip": args.openclip,
              "visual_pooling": args.visual_pooling,
              "calibrate_openclip": args.calibrate_openclip,
              "reranker_fusion": args.reranker_fusion,
              "count": len(rows), "skipped_missing_index": len(skipped),
              "skipped_query_ids": skipped.query_id.astype(str).tolist(), "by_task": by_task,
              "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows),
              "mean_reciprocal_rank": sum(1 / r["rank"] if r["rank"] else 0 for r in rows) / len(rows),
              "recall_at_1": sum(r["rank"] == 1 for r in rows) / len(rows),
              "recall_at_5": sum(r["rank"] is not None and r["rank"] <= 5 for r in rows) / len(rows),
              "recall_at_10": sum(r["rank"] is not None and r["rank"] <= 10 for r in rows) / len(rows),
              "recall_at_20": sum(r["rank"] is not None and r["rank"] <= 20 for r in rows) / len(rows),
              "recall_at_50": sum(r["rank"] is not None and r["rank"] <= 50 for r in rows) / len(rows),
              "recall_at_100": sum(r["rank"] is not None and r["rank"] <= 100 for r in rows) / len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
