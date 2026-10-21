from __future__ import annotations

from typing import Any, Callable

from aic_pipeline.task_router import route_query
from backend.runtime import Runtime, public_search

BACKEND_CANDIDATE_K = 100


def _candidate_score(candidate: dict[str, Any]) -> float:
    return float(candidate.get("fusion_score", candidate.get("score", 0.0)))


def _rank_trake(event_results: list[dict[str, Any]], maximum: int = 10):
    by_video: dict[str, list[list[dict[str, Any]]]] = {}
    for event_index, result in enumerate(event_results):
        for candidate in result.get("results", []):
            video_id = str(candidate.get("video_id", ""))
            if video_id:
                by_video.setdefault(video_id, [[] for _ in event_results])[event_index].append(candidate)

    valid, rejected = [], []
    for video_id, groups in by_video.items():
        absent = [index + 1 for index, group in enumerate(groups) if not group]
        if absent:
            frames = [max(group, key=_candidate_score) for group in groups if group]
            rejected.append({"video_id": video_id, "frames": frames,
                             "score": sum(_candidate_score(row) for row in frames),
                             "rejection_reason": f"Missing event(s): {', '.join(map(str, absent))}"})
            continue
        paths = [([candidate], _candidate_score(candidate)) for candidate in groups[0]]
        for group in groups[1:]:
            next_paths = []
            for candidate in group:
                frame_id = int(candidate.get("best_frame_id", candidate.get("frame_id", 0)))
                earlier = [path for path in paths
                           if int(path[0][-1].get("best_frame_id", path[0][-1].get("frame_id", 0))) < frame_id]
                if earlier:
                    frames, score = max(earlier, key=lambda path: path[1])
                    next_paths.append(([*frames, candidate], score + _candidate_score(candidate)))
            paths = next_paths
        if paths:
            frames, score = max(paths, key=lambda path: path[1])
            valid.append({"video_id": video_id, "frames": frames, "score": score})
        else:
            frames = [max(group, key=_candidate_score) for group in groups]
            rejected.append({"video_id": video_id, "frames": frames,
                             "score": sum(_candidate_score(row) for row in frames),
                             "rejection_reason": "Frames are not in the requested event order"})
    return (sorted(valid, key=lambda row: row["score"], reverse=True)[:maximum],
            sorted(rejected, key=lambda row: row["score"], reverse=True)[:maximum])


def execute_query(runtime: Runtime, query_type: str, query: str, *, top_k: int = 100,
                  rerank_k: int = 30,
                  openclip: bool = True, query_transformer: bool = False,
                  reranker_enabled: bool = True,
                  progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    engine = runtime.engine
    if query_type != "TRAKE":
        result = engine.search(query, top_k=top_k, candidate_k=BACKEND_CANDIDATE_K,
                               rerank_k=rerank_k if reranker_enabled else 0,
                               openclip=openclip and engine.openclip_available,
                               query_transformer=query_transformer,
                               progress_callback=progress_callback)
        return public_search(runtime, result)

    plan = route_query(query)
    if len(plan.events) < 2:
        raise ValueError("TRAKE requires at least two ordered events")
    event_results = [engine.search(event, top_k=50, candidate_k=BACKEND_CANDIDATE_K,
                                   rerank_k=12 if reranker_enabled else 0,
                                   openclip=False, query_transformer=query_transformer)
                    for event in plan.events]
    public_events = [{**{key: value for key, value in result.items() if key != "results"},
                      "results": [runtime.materialize_candidate(candidate, rank)
                                  for rank, candidate in enumerate(result.get("results", []), 1)]}
                     for result in event_results]
    rows, rejected = _rank_trake(event_results, maximum=min(10, top_k))
    rows = [{**row, "frames": [runtime.materialize_candidate(candidate, rank)
                                for rank, candidate in enumerate(row["frames"], 1)]}
            for row in rows]
    rejected = [{**row, "frames": [runtime.materialize_candidate(candidate, rank)
                                    for rank, candidate in enumerate(row["frames"], 1)]}
                for row in rejected]
    return {"task_plan": plan.to_dict(), "event_results": public_events,
            "results": public_events[0].get("results", []) if public_events else [],
            "trake_rows": rows, "trake_rejected": rejected,
            "query_transformer_used": bool(query_transformer and all(
                result.get("query_plan", {}).get("transformer_used") for result in event_results)),
            "query_transformer_available": engine.query_transformer.available}
