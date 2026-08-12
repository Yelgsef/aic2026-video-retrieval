from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from scripts.evaluate_trake_end_to_end import choose_video


def query_type(path: Path) -> str:
    match = re.search(r"-(kis|qa|trake)\.txt$", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer query type: {path}")
    return match.group(1).upper()


def clean_text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8-sig").split())


def trake_events(text: str) -> list[str]:
    matches = list(re.finditer(r"(?:^|\s)E\d+\s*:\s*", text, re.IGNORECASE))
    events = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        event = text[match.end():end].strip(" .;|")
        # Some files append an English gloss after all Vietnamese events.
        event = re.split(r"\s{2,}|\|", event, maxsplit=1)[0].strip()
        if event:
            events.append(event)
    return events or [text]


def compact_result(row: dict) -> dict:
    result = {key: row.get(key) for key in (
        "video_id", "scene_index", "start_frame", "end_frame_exclusive",
        "best_frame_id", "fusion_score", "reranker_score", "scene_context")}
    result["scene_id"] = f'{row["video_id"]}:scene:{int(row["scene_index"]):06d}'
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index_subset20")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "workspace/validation/legacy_queries_2025_results.json")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    paths = sorted(args.queries.rglob("*.txt"), key=lambda path: path.name)
    engine = OfflineSearchEngine(ROOT, args.index)
    rows = []
    for number, path in enumerate(paths, 1):
        kind, text = query_type(path), clean_text(path)
        started = time.perf_counter()
        if kind == "TRAKE":
            events = trake_events(text)
            selected, video_scores, windows = choose_video(
                engine, events, fusion_mode="rrf", field_aware=True,
                openclip=False, reranker_fusion="rank", object_video_weight=0.9)
            ordered_videos = sorted(video_scores, key=lambda key: (
                video_scores[key]["coverage"], video_scores[key]["video_fusion_score"]), reverse=True)
            output = {"events": events, "selected_video": selected,
                      "video_candidates": [{"video_id": key, **video_scores[key]} for key in ordered_videos[:10]],
                      "event_results": [[compact_result(item) for item in result] for result in windows]}
        else:
            response = engine.search(
                text, top_k=args.top_k, candidate_k=300, rerank_k=30,
                fusion_mode="normalized", field_aware=True, openclip=True,
                visual_pooling="max", calibrate_openclip=kind == "KIS",
                reranker_fusion="auto")
            output = {"route": response.get("task"), "query_expansion": response.get("query_expansion"),
                      "results": [compact_result(item) for item in response["results"]]}
        elapsed = time.perf_counter() - started
        rows.append({"query_id": path.stem, "query_type": kind, "query_text": text,
                     "latency_seconds": elapsed, **output})
        print(f"[{number:02d}/{len(paths)}] {path.stem}: {elapsed:.2f}s", flush=True)

    latencies = [row["latency_seconds"] for row in rows]
    by_type = {}
    for kind in sorted({row["query_type"] for row in rows}):
        selected = [row for row in rows if row["query_type"] == kind]
        by_type[kind] = {"count": len(selected),
                         "mean_latency_seconds": sum(row["latency_seconds"] for row in selected) / len(selected)}
    report = {"ground_truth_available": False, "accuracy_metrics_available": False,
              "index": str(args.index), "indexed_scenes": len(engine.scenes),
              "indexed_videos": int(engine.scenes.video_id.nunique()), "count": len(rows),
              "mean_latency_seconds": sum(latencies) / len(latencies), "by_type": by_type, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = []
    for row in rows:
        results = row.get("results") or []
        first = results[0] if results else {}
        flat.append({"query_id": row["query_id"], "query_type": row["query_type"],
                     "query_text": row["query_text"], "top_video_id": row.get("selected_video") or first.get("video_id"),
                     "top_scene_id": first.get("scene_id"), "top_frame_id": first.get("best_frame_id"),
                     "latency_seconds": row["latency_seconds"]})
    pd.DataFrame(flat).to_csv(args.output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
