from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from aic_pipeline.qa_gemini import GeminiQAEngine, local_exact_ocr_answer, qa_grounding_query
from scripts.run_l21_first20_querypack import QUERIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA regression without a localhost subprocess.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--query", help="Run one ad-hoc QA question instead of the built-in regression set.")
    parser.add_argument("--query-id", default="qa_adhoc")
    args = parser.parse_args()
    engine = OfflineSearchEngine(ROOT, args.index, args.device)
    qa = GeminiQAEngine(ROOT)
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    queries = [("QA", args.query_id, args.query)] if args.query else QUERIES
    for task, query_id, question in queries:
        if task != "QA":
            continue
        if args.only and query_id not in args.only:
            continue
        started = time.perf_counter()
        try:
            retrieval = engine.search(
                qa_grounding_query(question), top_k=60, rerank_k=30,
                fusion_mode="normalized", field_aware=True,
                openclip=engine.openclip_vectors is not None,
                calibrate_openclip=False, reranker_fusion="rank",
                query_transformer=False,
            )
            answer = (local_exact_ocr_answer(ROOT, question, retrieval["results"][0])
                      if retrieval["results"] else None)
            if answer is None:
                answer = qa.answer(question, retrieval["results"], 24)
            result = {"answer": answer, "retrieval": retrieval}
            status, error = "ok", None
        except Exception as exc:
            result, status, error = {}, "error", f"{type(exc).__name__}: {exc}"
        record = {"query_id": query_id, "task": task, "query": question,
                  "status": status, "error": error,
                  "elapsed_seconds": round(time.perf_counter() - started, 3),
                  "result": result}
        (args.output / f"{query_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        print(json.dumps({k: record[k] for k in ("query_id", "status", "error", "elapsed_seconds")},
                         ensure_ascii=False), flush=True)
    (args.output / "all_results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
