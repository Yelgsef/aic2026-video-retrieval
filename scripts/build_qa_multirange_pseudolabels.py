from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from google.genai import types
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from aic_pipeline.qa_gemini import GeminiQAEngine, qa_grounding_query


class AcceptedEvidence(BaseModel):
    candidate_labels: list[str] = Field(description="All candidates that visibly support the reference answer")
    evidence_vi: str
    confidence: float = Field(ge=0, le=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative multi-range QA pseudo-labels.")
    parser.add_argument("--labels", type=Path, default=ROOT / "workspace/validation/review_pack_50/annotations_autolabeled.csv")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index_subset10")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/validation/qa_multirange_subset10.jsonl")
    parser.add_argument("--candidates", type=int, default=24)
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    labels = labels.loc[labels.query_type.eq("QA")]
    engine = OfflineSearchEngine(ROOT, args.index)
    qa = GeminiQAEngine(ROOT)
    records = []
    for label in labels.itertuples(index=False):
        results = engine.search(qa_grounding_query(str(label.query_text_vi)),
                                top_k=args.candidates, rerank_k=30)["results"]
        candidates = []
        for result in results:
            item = dict(result)
            item["best_frame_id"] = int((result.get("top_visual_frame_ids") or
                                         [result["best_frame_id"]])[0])
            candidates.append(item)
        sheet, mapping = qa._sheet(candidates)  # shared production contact-sheet layout
        prompt = f"""Conservatively annotate valid evidence for an AIC visual QA query.
Question: {label.query_text_vi}
Reference answer: {label.answer}
Return every candidate label where the image visibly supports both the described scene and the reference answer.
Do not accept a candidate based only on similarity, outside knowledge, or the supplied answer.
An empty list is valid. Candidate labels: {', '.join(mapping)}.
"""
        response = qa.client.models.generate_content(
            model=qa.model, contents=[prompt, sheet],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=AcceptedEvidence,
                temperature=0.0, max_output_tokens=768,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")))
        decision = AcceptedEvidence.model_validate_json(response.text)
        ranges = [{"video_id": str(label.video_id), "scene_id": str(label.scene_id),
                   "start_frame": int(label.start_frame),
                   "end_frame_exclusive": int(label.end_frame_exclusive),
                   "source": "original_pseudo_label"}]
        seen = {(ranges[0]["video_id"], ranges[0]["start_frame"], ranges[0]["end_frame_exclusive"])}
        for candidate_label in decision.candidate_labels:
            candidate = mapping.get(candidate_label.strip().upper())
            if candidate is None:
                continue
            key = (str(candidate["video_id"]), int(candidate["start_frame"]),
                   int(candidate["end_frame_exclusive"]))
            if key not in seen:
                ranges.append({"video_id": key[0],
                               "scene_id": f"{key[0]}:scene:{int(candidate['scene_index']):06d}",
                               "start_frame": key[1], "end_frame_exclusive": key[2],
                               "source": "gemini_equivalent_evidence"})
                seen.add(key)
        record = {"query_id": str(label.query_id), "query_type": "QA",
                  "query_text_vi": str(label.query_text_vi), "answer": str(label.answer),
                  "accepted_ranges": ranges, "verified": False,
                  "confidence": decision.confidence, "evidence_vi": decision.evidence_vi,
                  "model": qa.model}
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
                               encoding="utf-8")
        print(json.dumps({"query_id": label.query_id, "accepted_ranges": len(ranges),
                          "confidence": decision.confidence}, ensure_ascii=False), flush=True)
        time.sleep(0.25)
    print(json.dumps({"output": str(args.output), "queries": len(records),
                      "ranges": sum(len(x["accepted_ranges"]) for x in records)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
