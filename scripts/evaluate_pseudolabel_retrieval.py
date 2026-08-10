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
    parser.add_argument("--fusion-mode", choices=("rrf", "normalized"), default="rrf")
    parser.add_argument("--field-aware", action="store_true")
    parser.add_argument("--openclip", action="store_true")
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    if args.limit:
        labels = labels.head(args.limit)
    engine = OfflineSearchEngine(ROOT, args.index)
    rows = []
    for label in labels.itertuples(index=False):
        started = time.perf_counter()
        response = engine.search(str(label.query_text_vi), top_k=100, rerank_k=30,
                                 fusion_mode=args.fusion_mode, field_aware=args.field_aware,
                                 openclip=args.openclip)
        expected = int(str(label.scene_id).rsplit(":", 1)[-1])
        returned = [int(item["scene_index"]) for item in response["results"]]
        rank = returned.index(expected) + 1 if expected in returned else None
        rows.append({"query_id": label.query_id, "query_type": label.query_type, "expected_scene": expected,
                     "rank": rank, "latency_seconds": time.perf_counter() - started})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {"pseudo_labels": True, "fusion_mode": args.fusion_mode, "field_aware": args.field_aware,
              "openclip": args.openclip,
              "count": len(rows), "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows),
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
