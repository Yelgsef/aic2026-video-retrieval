from __future__ import annotations

from dataclasses import dataclass

from .object_spatial import fold_text, parse_object_constraints


@dataclass(frozen=True)
class RetrievalProfile:
    kind: str
    weights: dict[str, float]


DEFAULT_WEIGHTS = {
    "semantic": {"visual": 0.34, "openclip": 0.18, "semantic": 0.22,
                 "lexical": 0.05, "coverage": 0.06, "object": 0.0, "reranker": 0.15},
    "object": {"visual": 0.25, "openclip": 0.12, "semantic": 0.16,
               "lexical": 0.05, "coverage": 0.07, "object": 0.22, "reranker": 0.13},
    "spatial": {"visual": 0.20, "openclip": 0.10, "semantic": 0.13,
                "lexical": 0.04, "coverage": 0.06, "object": 0.34, "reranker": 0.13},
    "count": {"visual": 0.18, "openclip": 0.09, "semantic": 0.12,
              "lexical": 0.04, "coverage": 0.05, "object": 0.40, "reranker": 0.12},
    "ocr": {"visual": 0.10, "openclip": 0.04, "semantic": 0.20,
            "lexical": 0.28, "coverage": 0.10, "object": 0.03, "reranker": 0.25},
    "asr": {"visual": 0.08, "openclip": 0.03, "semantic": 0.27,
            "lexical": 0.25, "coverage": 0.10, "object": 0.02, "reranker": 0.25},
}

# Accepted only after video-grouped cross-validation on 2026-08-11. Profiles
# absent from this map retain the robust production fusion unchanged.
CV_ACCEPTED_WEIGHTS = {
    "QA:count": {"visual": 0.3069116771, "openclip": 0.0647099093,
                 "semantic": 0.0620087609, "lexical": 0.0560474098,
                 "coverage": 0.0700756833, "object": 0.3868817985,
                 "reranker": 0.0533647798},
}


def classify_retrieval_query(query: str, intent: str = "visual") -> str:
    if intent in {"ocr", "asr"}:
        return intent
    folded = fold_text(query)
    constraints = parse_object_constraints(query)
    if "bao nhieu" in folded or "may " in f"{folded} " or any(
            constraint["count"] is not None for constraint in constraints):
        return "count"
    if any(constraint["positions"] for constraint in constraints) or any(
            value in folded for value in ("ben trai", "ben phai", "o giua", "phia tren", "phia duoi")):
        return "spatial"
    if constraints:
        return "object"
    return "semantic"


def retrieval_profile(query: str, intent: str = "visual",
                      overrides: dict[str, dict[str, float]] | None = None) -> RetrievalProfile:
    kind = classify_retrieval_query(query, intent)
    weights = dict((overrides or {}).get(kind, DEFAULT_WEIGHTS[kind]))
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        raise ValueError(f"Retrieval profile {kind} has no positive weight")
    return RetrievalProfile(kind, {name: max(0.0, value) / total for name, value in weights.items()})


def accepted_profile(task: str, query: str, intent: str = "visual") -> RetrievalProfile | None:
    kind = classify_retrieval_query(query, intent)
    weights = CV_ACCEPTED_WEIGHTS.get(f"{task.upper()}:{kind}")
    return retrieval_profile(query, intent, {kind: weights}) if weights else None
