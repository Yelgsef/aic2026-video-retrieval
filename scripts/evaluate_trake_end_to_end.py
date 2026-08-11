from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.offline_search_engine import OfflineSearchEngine
from aic_pipeline.temporal_alignment import TemporalAligner
from aic_pipeline.xclip_reranker import XClipWindowReranker
from aic_pipeline.retrieval_fusion import robust_unit


def choose_video(engine: OfflineSearchEngine, events: list[str],
                 fusion_mode: str = "rrf", field_aware: bool = False,
                 openclip: bool = False, visual_pooling: str = "max",
                 calibrate_openclip: bool = False,
                 reranker_fusion: str = "rank",
                 object_video_weight: float = 0.0) -> tuple[str, dict, list[list[dict]]]:
    per_event, per_event_object, event_results, videos = [], [], [], set()
    for event in events:
        result = engine.search(event, top_k=100, rerank_k=30, fusion_mode=fusion_mode,
                               field_aware=field_aware, openclip=openclip,
                               visual_pooling=visual_pooling,
                               calibrate_openclip=calibrate_openclip,
                               reranker_fusion=reranker_fusion)
        event_results.append(result["results"])
        scores = {}
        object_scores = {}
        for row in result["results"]:
            video_id = row["video_id"]
            scores[video_id] = max(scores.get(video_id, float("-inf")), float(row["fusion_score"]))
            if row.get("object_spatial_score") is not None:
                object_scores[video_id] = max(object_scores.get(video_id, 0.0),
                                              float(row["object_spatial_score"]))
        per_event.append(scores)
        per_event_object.append(object_scores)
        videos.update(scores)
    summary = {}
    ordered_videos = sorted(videos)
    fusion_raw = np.array([sum(scores.get(video, 0.0) for scores in per_event) / len(per_event)
                           for video in ordered_videos], dtype=np.float32)
    object_raw = np.array([sum(scores.get(video, 0.0) for scores in per_event_object) / len(per_event_object)
                           for video in ordered_videos], dtype=np.float32)
    fusion_units, object_units = robust_unit(fusion_raw), robust_unit(object_raw)
    for video_id in videos:
        available = [scores[video_id] for scores in per_event if video_id in scores]
        position = ordered_videos.index(video_id)
        blended = ((1 - object_video_weight) * float(fusion_units[position])
                   + object_video_weight * float(object_units[position]))
        summary[video_id] = {"coverage": len(available) / len(per_event),
                             "mean_event_score": sum(available) / len(available),
                             "mean_object_score": float(object_raw[position]),
                             "video_fusion_score": blended}
    selected = max(summary, key=lambda v: (summary[v]["coverage"], summary[v]["video_fusion_score"]))
    windows = [[row for row in results if row["video_id"] == selected][:5]
               for results in event_results]
    return selected, summary, windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate video selection and temporal alignment for TRAKE.")
    parser.add_argument("--labels", type=Path, default=ROOT / "workspace/validation/trake_pseudolabels_subset10.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "workspace/offline_index_subset10")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/validation/trake_end_to_end_subset10.json")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N labels for smoke tests")
    parser.add_argument("--decoder", choices=("dense", "windowed"), default="windowed")
    parser.add_argument("--fusion-mode", choices=("rrf", "normalized"), default="rrf")
    parser.add_argument("--field-aware", action="store_true")
    parser.add_argument("--short-clip", action="store_true")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--clip-scales", type=float, nargs="+",
                        help="Optional multi-scale short-clip windows in seconds")
    parser.add_argument("--motion-rerank", action="store_true")
    parser.add_argument("--global-peak-weight", type=float, default=0.35)
    parser.add_argument("--xclip", action="store_true")
    parser.add_argument("--xclip-weight", type=float, default=0.25)
    parser.add_argument("--object-video-weight", type=float, default=0.0)
    parser.add_argument("--openclip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visual-pooling", choices=("max", "top3_mean", "max_top3"), default="max")
    parser.add_argument("--calibrate-openclip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reranker-fusion", choices=("rank", "score", "hybrid", "auto"), default="rank")
    args = parser.parse_args()
    labels = [json.loads(x) for x in args.labels.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit is not None:
        labels = labels[:args.limit]
    engine = OfflineSearchEngine(ROOT, args.index)
    aligner = TemporalAligner(ROOT)
    xclip = XClipWindowReranker(ROOT) if args.xclip else None
    rows = []
    for label in labels:
        started = time.perf_counter()
        events = [x["description_vi"] for x in label["events"]]
        selected_video, candidates, windows = choose_video(
            engine, events, args.fusion_mode, args.field_aware, args.openclip, args.visual_pooling,
            args.calibrate_openclip, args.reranker_fusion, args.object_video_weight)
        if xclip is not None:
            windows = xclip.rerank(selected_video, events, windows, weight=args.xclip_weight)
        alignment = aligner.align(selected_video, events, windows, decoder=args.decoder,
                                  short_clip=args.short_clip, clip_seconds=args.clip_seconds,
                                  motion_rerank=args.motion_rerank,
                                  clip_scales=tuple(args.clip_scales) if args.clip_scales else None,
                                  global_peak_weight=args.global_peak_weight)
        predicted = alignment["events"]
        hits = [selected_video == label["video_id"] and truth["start_frame"] <= pred["frame_id"] < truth["end_frame_exclusive"]
                for truth, pred in zip(label["events"], predicted)]
        increasing = all(a["frame_id"] < b["frame_id"] for a, b in zip(predicted, predicted[1:]))
        row = {"query_id": label["query_id"], "expected_video_id": label["video_id"],
               "predicted_video_id": selected_video, "video_hit": selected_video == label["video_id"],
               "event_hits": hits, "all_events_hit": all(hits), "strictly_increasing": increasing,
               "predicted_frame_ids": [x["frame_id"] for x in predicted],
               "latency_seconds": time.perf_counter() - started, "video_candidates": candidates}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    event_count = sum(len(r["event_hits"]) for r in rows)
    report = {"pseudo_labels": True, "decoder": args.decoder, "fusion_mode": args.fusion_mode,
              "field_aware": args.field_aware, "count": len(rows),
              "short_clip": args.short_clip, "clip_seconds": args.clip_seconds if args.short_clip else None,
              "clip_scales": args.clip_scales if args.short_clip else None,
              "motion_rerank": args.motion_rerank,
              "global_peak_weight": args.global_peak_weight,
              "xclip": args.xclip, "xclip_weight": args.xclip_weight if args.xclip else None,
              "object_video_weight": args.object_video_weight,
              "openclip": args.openclip, "visual_pooling": args.visual_pooling,
              "calibrate_openclip": args.calibrate_openclip,
              "reranker_fusion": args.reranker_fusion,
              "video_accuracy": sum(r["video_hit"] for r in rows) / len(rows),
              "event_hit_rate": sum(sum(r["event_hits"]) for r in rows) / event_count,
              "sequence_accuracy": sum(r["all_events_hit"] for r in rows) / len(rows),
              "increasing_rate": sum(r["strictly_increasing"] for r in rows) / len(rows),
              "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows), "rows": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
