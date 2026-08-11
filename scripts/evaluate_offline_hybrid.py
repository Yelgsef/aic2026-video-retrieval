from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def target_scene(scenes: pd.DataFrame, shot_id: int) -> int:
    matches = scenes[scenes.shot_ids.map(lambda values: shot_id in set(values))]
    if len(matches) != 1:
        raise ValueError(f"Shot {shot_id} maps to {len(matches)} semantic scenes")
    return int(matches.iloc[0].scene_index)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate the offline hybrid query pipeline on weak temporal labels.")
    parser.add_argument("--queries", type=Path, default=ROOT / "workspace" / "validation" / "queries_L21_V001.parquet")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace" / "offline_index")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "validation" / "offline_hybrid_eval.json")
    args = parser.parse_args()
    queries = pd.read_parquet(args.queries)
    if args.limit:
        queries = queries.head(args.limit)
    scene_cache: dict[str, pd.DataFrame] = {}
    results, reciprocal_ranks, latencies = [], [], []
    for query in queries.itertuples(index=False):
        if query.video_id not in scene_cache:
            scene_cache[query.video_id] = pd.read_parquet(
                ROOT / "workspace" / "semantic_scenes" / query.video_id / "scenes.parquet")
        expected = target_scene(scene_cache[query.video_id], int(query.ground_truth_shot_id))
        command = [sys.executable, str(ROOT / "scripts" / "query_offline_index.py"), str(query.query_text_vi),
                   "--index", str(args.index), "--top-k", str(args.top_k)]
        if args.no_reranker:
            command.append("--no-reranker")
        started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        latency = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(process.stderr[-2000:])
        payload = json.loads(process.stdout)
        expected_key = (str(query.video_id), expected)
        returned = [(str(row["video_id"]), int(row["scene_index"])) for row in payload["results"]]
        rank = returned.index(expected_key) + 1 if expected_key in returned else None
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        latencies.append(latency)
        results.append({"query_id": query.query_id, "query": query.query_text_vi, "expected_scene": expected,
                        "rank": rank, "hit_at_1": rank == 1, "hit_at_5": rank is not None and rank <= 5,
                        "hit_at_10": rank is not None and rank <= 10, "latency_seconds": latency,
                        "intent": payload["intent"], "reranker_used": payload["reranker_used"]})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    count = len(results)
    report = {"query_count": count, "weak_labels": True, "reranker_enabled": not args.no_reranker,
              "recall_at_1": sum(row["hit_at_1"] for row in results) / count,
              "recall_at_5": sum(row["hit_at_5"] for row in results) / count,
              "recall_at_10": sum(row["hit_at_10"] for row in results) / count,
              "mrr": sum(reciprocal_ranks) / count, "mean_latency_seconds": sum(latencies) / count,
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
