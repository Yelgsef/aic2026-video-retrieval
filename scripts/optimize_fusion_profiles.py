from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.fusion_profiles import DEFAULT_WEIGHTS
from aic_pipeline.retrieval_fusion import robust_unit


FEATURES = ("visual", "openclip", "semantic", "lexical", "coverage", "object", "reranker")
SOURCE_KEYS = {"visual": "visual_score", "openclip": "openclip_score",
               "semantic": "semantic_score", "lexical": "rapidfuzz_score",
               "coverage": "field_coverage_score", "object": "object_spatial_score",
               "reranker": "reranker_score"}


def feature_matrix(query: dict) -> np.ndarray:
    if "_feature_matrix" in query:
        return query["_feature_matrix"]
    candidates = query["candidates"]
    columns = []
    for feature in FEATURES:
        raw = np.array([float(row.get(SOURCE_KEYS[feature]) or 0.0) for row in candidates], dtype=np.float32)
        columns.append(robust_unit(raw))
    return np.stack(columns, axis=1)


def relevant(candidate: dict, query: dict) -> bool:
    return (str(candidate["video_id"]) == str(query["video_id"])
            and int(candidate["start_frame"]) < int(query["end_frame_exclusive"])
            and int(candidate["end_frame_exclusive"]) > int(query["start_frame"]))


def rank_for(query: dict, weights: np.ndarray | None) -> int | None:
    if weights is None:
        order = np.argsort(-np.array([float(row["fusion_score"]) for row in query["candidates"]]))
    else:
        order = np.argsort(-(feature_matrix(query) @ weights))
    relevance = query.get("_relevance")
    for rank, position in enumerate(order, 1):
        if (bool(relevance[int(position)]) if relevance is not None
                else relevant(query["candidates"][int(position)], query)):
            return rank
    return None


def metrics(queries: list[dict], weights: np.ndarray | None) -> dict[str, float]:
    ranks = [rank_for(query, weights) for query in queries]
    count = max(1, len(ranks))
    return {"count": len(ranks),
            "mrr": sum(1 / rank if rank else 0 for rank in ranks) / count,
            "r1": sum(rank == 1 for rank in ranks) / count,
            "r5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
            "r10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
            "r100": sum(rank is not None and rank <= 100 for rank in ranks) / count}


def objective(values: dict[str, float]) -> float:
    return values["mrr"] + 0.35 * values["r5"] + 0.20 * values["r10"] + 0.05 * values["r100"]


def candidates(kind: str, samples: int, seed: int) -> list[np.ndarray]:
    default = np.array([DEFAULT_WEIGHTS[kind][name] for name in FEATURES], dtype=np.float32)
    rng = np.random.default_rng(seed)
    output = [default]
    # Concentration keeps the search near interpretable, conservative profiles.
    for concentration in (20.0, 50.0):
        for vector in rng.dirichlet(np.maximum(default * concentration, 0.15), samples // 2):
            output.append((0.25 * default + 0.75 * vector).astype(np.float32))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-grouped CV for cached multimodal fusion scores.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    queries = json.loads(args.cache.read_text(encoding="utf-8"))["queries"]
    for query in queries:
        query["_feature_matrix"] = feature_matrix(query)
        query["_relevance"] = np.array([relevant(row, query) for row in query["candidates"]], dtype=bool)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        grouped[f"{str(query['query_type']).lower()}_{query['profile_kind']}"].append(query)
    report = {"cache": str(args.cache), "grouping": "video_id", "folds": args.folds,
              "features": list(FEATURES), "profiles": {}}
    for profile_name, selected in sorted(grouped.items()):
        kind = str(selected[0]["profile_kind"])
        pool = candidates(kind, args.samples, args.seed + len(profile_name))
        videos = sorted({str(query["video_id"]) for query in selected})
        fold_rows = []
        for fold in range(min(args.folds, len(videos))):
            held = {video for index, video in enumerate(videos) if index % args.folds == fold}
            train = [query for query in selected if str(query["video_id"]) not in held]
            test = [query for query in selected if str(query["video_id"]) in held]
            if not train or not test:
                continue
            best = max(pool, key=lambda weights: objective(metrics(train, weights)))
            fold_rows.append({"held_videos": sorted(held), "count": len(test),
                              "baseline": metrics(test, None), "tuned": metrics(test, best),
                              "weights": {name: float(value) for name, value in zip(FEATURES, best)}})
        final = max(pool, key=lambda weights: objective(metrics(selected, weights)))
        baseline = metrics(selected, None)
        tuned = metrics(selected, final)
        cv_baseline_mrr = sum(row["baseline"]["mrr"] * row["count"] for row in fold_rows) / len(selected)
        cv_tuned_mrr = sum(row["tuned"]["mrr"] * row["count"] for row in fold_rows) / len(selected)
        report["profiles"][profile_name] = {"query_type": str(selected[0]["query_type"]),
            "profile_kind": kind, "count": len(selected), "videos": len(videos),
            "baseline": baseline, "tuned_all_data": tuned,
            "selected_weights": {name: float(value) for name, value in zip(FEATURES, final)},
            "cv_baseline_mrr": cv_baseline_mrr, "cv_tuned_mrr": cv_tuned_mrr,
            "cv_accepted": cv_tuned_mrr > cv_baseline_mrr and tuned["r100"] >= baseline["r100"],
            "folds": fold_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({kind: {key: value for key, value in data.items() if key != "folds"}
                      for kind, data in report["profiles"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
