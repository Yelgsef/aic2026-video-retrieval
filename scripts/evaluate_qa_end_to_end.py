from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from aic_pipeline.qa_gemini import GeminiQAEngine, qa_grounding_query


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c) and c.isalnum() or c.isspace()).split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + Gemini VQA on QA pseudo-labels.")
    parser.add_argument("--labels", type=Path, default=ROOT / "workspace/validation/review_pack_50/annotations_autolabeled.csv")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index_subset10")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/validation/qa_end_to_end_subset10.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--accepted-ranges", type=Path)
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    labels = labels.loc[labels.query_type.eq("QA")]
    if args.query_id:
        labels = labels.loc[labels.query_id.isin(args.query_id)]
    if args.limit:
        labels = labels.head(args.limit)
    retrieval = OfflineSearchEngine(ROOT, args.index)
    qa = GeminiQAEngine(ROOT)
    multi_ranges = {}
    if args.accepted_ranges:
        multi_ranges = {item["query_id"]: item["accepted_ranges"] for item in
                        (json.loads(line) for line in args.accepted_ranges.read_text(encoding="utf-8").splitlines()
                         if line.strip())}
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in labels.itertuples(index=False):
        started = time.perf_counter()
        try:
            found = retrieval.search(qa_grounding_query(str(label.query_text_vi)),
                                     top_k=max(30, args.candidates), rerank_k=30)
            answer = qa.answer(str(label.query_text_vi), found["results"], args.candidates)
            expected = normalize(label.answer)
            actual = normalize(answer["answer"])
            score = ratio(expected, actual) / 100
            answer_match = expected in actual or actual in expected or score >= 0.8
            accepted = multi_ranges.get(str(label.query_id), [{"video_id": str(label.video_id),
                                                               "start_frame": int(label.start_frame),
                                                               "end_frame_exclusive": int(label.end_frame_exclusive)}])
            frame_hit = any(str(answer["video_id"]) == str(item["video_id"])
                            and int(item["start_frame"]) <= int(answer["frame_id"])
                            < int(item["end_frame_exclusive"]) for item in accepted)
            rows.append({"query_id": label.query_id, "query": label.query_text_vi,
                         "expected_answer": label.answer, "predicted_answer": answer["answer"],
                         "expected_video_id": label.video_id, "predicted_video_id": answer["video_id"],
                         "predicted_frame_id": answer["frame_id"], "frame_hit": frame_hit,
                         "answer_match": answer_match, "answer_similarity": score,
                         "confidence": answer["confidence"], "status": "completed",
                         "accepted_range_count": len(accepted),
                         "object_grounding_used": answer.get("object_grounding_used", False),
                         "object_crop_count": answer.get("object_crop_count", 0),
                         "latency_seconds": time.perf_counter() - started})
        except Exception as exc:
            rows.append({"query_id": label.query_id, "query": label.query_text_vi,
                         "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                         "frame_hit": False, "answer_match": False,
                         "latency_seconds": time.perf_counter() - started})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        # Crash-resumable checkpoint: external VQA calls must not lose prior rows.
        args.output.write_text(json.dumps({"partial": True, "rows": rows}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    count = len(rows)
    report = {"pseudo_labels": True, "count": count,
              "completed": sum(r["status"] == "completed" for r in rows),
              "frame_hit_rate": sum(r["frame_hit"] for r in rows) / count,
              "answer_match_rate": sum(r["answer_match"] for r in rows) / count,
              "joint_accuracy": sum(r["frame_hit"] and r["answer_match"] for r in rows) / count,
              "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / count,
              "rows": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
