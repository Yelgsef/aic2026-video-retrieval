from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskPlan:
    task: str
    retrieval_query: str
    question: str
    events: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict:
        value = asdict(self)
        value["events"] = list(self.events)
        return value


def split_events(query: str) -> tuple[str, ...]:
    numbered = re.findall(r"(?:^|[;(]\s*)\d+[.)-]?\s*([^;()]+)", query)
    if len(numbered) >= 2:
        return tuple(" ".join(value.strip(" .,").split()) for value in numbered)
    markers = re.split(r"\b(?:sau đó|tiếp theo|cuối cùng|rồi)\b", query, flags=re.IGNORECASE)
    return tuple(value.strip(" .,;:") for value in markers if value.strip()) if len(markers) >= 3 else ()


def route_query(query: str) -> TaskPlan:
    normalized = " ".join(query.strip().split())
    events = split_events(normalized)
    if events:
        return TaskPlan("TRAKE", normalized, "", events, 0.9)
    folded = normalized.casefold()
    question_markers = ("bao nhiêu", "màu gì", "là gì", "ai ", "ở đâu", "khi nào", "như thế nào",
                        "đang cầm gì", "đang làm gì", "có gì", "loại gì")
    if "?" in normalized or any(marker in folded for marker in question_markers):
        return TaskPlan("QA", normalized, normalized, (), 0.85)
    return TaskPlan("KIS", normalized, "", (), 0.95)
