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
    # Accept the formats commonly used by AIC queries: `(1) ...; (2) ...`,
    # `1. ...\n2. ...`, and `E1: ...`. Slice between markers instead of
    # matching event text with `[^;()]+`, because event descriptions may
    # legitimately contain parentheses.
    marker = re.compile(
        r"(?:^|[;\n])\s*(?:\(\s*\d+\s*\)|E\s*\d+\s*:?|\d+\s*[.):-])\s*",
        flags=re.IGNORECASE,
    )
    matches = list(marker.finditer(query))
    if len(matches) >= 2:
        numbered = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
            value = " ".join(query[match.end():end].strip(" ;.,").split())
            if value:
                numbered.append(value)
        if len(numbered) >= 2:
            return tuple(numbered)
    markers = re.split(r"\b(?:sau đó|tiếp theo|cuối cùng|rồi)\b", query, flags=re.IGNORECASE)
    return tuple(value.strip(" .,;:") for value in markers if value.strip()) if len(markers) >= 3 else ()


def route_query(query: str) -> TaskPlan:
    normalized = " ".join(query.strip().split())
    # Parse event markers before whitespace normalization so line breaks remain
    # valid separators for numbered TRAKE events.
    events = split_events(query.strip())
    if events:
        return TaskPlan("TRAKE", normalized, "", events, 0.9)
    folded = normalized.casefold()
    question_markers = ("bao nhiêu", "màu gì", "là gì", "ở đâu", "khi nào", "như thế nào",
                        "đang cầm gì", "đang làm gì", "có gì", "loại gì")
    asks_who = re.search(r"(?<!\w)ai(?!\w)", folded) is not None
    if "?" in normalized or asks_who or any(marker in folded for marker in question_markers):
        return TaskPlan("QA", normalized, normalized, (), 0.85)
    return TaskPlan("KIS", normalized, "", (), 0.95)
