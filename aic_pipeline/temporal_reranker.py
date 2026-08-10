from __future__ import annotations

import numpy as np

from .retrieval_fusion import robust_unit


MOTION_TERMS = ("đi", "chạy", "di chuyển", "quay", "rời", "đặt", "lấy", "mở", "đóng",
                "đứng lên", "ngồi xuống", "bơi", "bay", "tiến vào", "đi ra", "sau đó")


def motion_aware_scores(events: list[str], embeddings: np.ndarray, scores: np.ndarray,
                        weight: float = 0.08) -> tuple[np.ndarray, list[bool]]:
    """Candidate-only motion prior using embedding change, with no extra model load.

    This is deliberately gated by action language and must remain an ablation
    until a real TRAKE validation set shows a gain.
    """
    if len(embeddings) < 3:
        return scores, [False] * len(events)
    novelty = np.zeros(len(embeddings), dtype=np.float32)
    novelty[1:-1] = 1 - np.sum(embeddings[:-2] * embeddings[2:], axis=1)
    novelty = robust_unit(novelty)
    output = scores.copy()
    gated = []
    for index, event in enumerate(events):
        active = any(term in event.casefold() for term in MOTION_TERMS)
        gated.append(active)
        if active:
            output[index] += weight * novelty
    return output, gated
