from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class QAReasoningPlan:
    kind: str
    needs_temporal: bool
    needs_object_crop: bool
    needs_exact_text: bool
    instruction: str


def fold_answer(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text).casefold()).replace("đ", "d")
    return " ".join("".join(c for c in value if not unicodedata.combining(c)
                              and (c.isalnum() or c.isspace())).split())


ANSWER_ALIASES = (
    ({"bot bao ve", "chot bao ve", "tram bao ve", "nha bao ve"}, "guard_booth"),
    ({"rua bien", "con rua bien"}, "sea_turtle"),
    ({"viet", "viet len giay", "ghi chep", "dang viet"}, "writing"),
    ({"xanh duong", "mau xanh duong", "xanh dam", "mau xanh dam"}, "blue"),
    ({"cam", "mau cam"}, "orange"),
)


def canonical_answer(text: str) -> str:
    folded = fold_answer(text)
    for aliases, canonical in ANSWER_ALIASES:
        if folded in aliases:
            return canonical
    return folded


def answers_equivalent(left: str, right: str) -> bool:
    a, b = canonical_answer(left), canonical_answer(right)
    return bool(a and b and (a == b or a in b or b in a))


def classify_question(question: str) -> QAReasoningPlan:
    folded = fold_answer(question)
    # The requested answer slot is more important than motion words appearing
    # in the scene description (for example, "động vật nào ... đang di chuyển").
    if any(term in folded for term in ("con vat nao", "loai dong vat nao", "dong vat nao")):
        return QAReasoningPlan("object", False, True, False,
                               "Identify the animal from both the full context and the close crop.")
    if any(term in folded for term in ("huong nao", "ve phia nao", "sang trai", "sang phai", "tien ve", "lui ve")):
        return QAReasoningPlan("motion", True, False, False,
                               "Use the chronological frames to infer motion direction; do not infer it from one frame.")
    if any(term in folded for term in ("ghi chu", "ghi so", "bien so", "dong chu", "van ban", "tieu de")):
        return QAReasoningPlan("ocr", False, True, True,
                               "Read the exact visible characters from the clearest full-resolution crop.")
    if any(term in folded for term in ("cay nao", "loai cay nao", "rau gi", "loai rau nao")):
        return QAReasoningPlan("named_entity", False, False, True,
                               "Return a plant name only when that name is visibly printed or present in supplied evidence; never infer a species from outside knowledge.")
    if any(term in folded for term in ("bao nhieu", "may nguoi", "may con", "so luong")):
        return QAReasoningPlan("count", False, True, False,
                               "Count only clearly visible instances; avoid counting the same object twice.")
    if "mau gi" in folded or "mau nao" in folded or "co mau" in folded:
        return QAReasoningPlan("color", False, True, False,
                               "Judge color from the target object crop and account for lighting or shadows.")
    if any(term in folded for term in ("dang cam", "vat gi", "con vat", "dong vat", "la gi")):
        return QAReasoningPlan("object", False, True, False,
                               "Identify the object from both the subject-object context and the close crop.")
    if any(term in folded for term in ("phia nao", "ben trai", "ben phai", "o dau", "nam ngay canh")):
        return QAReasoningPlan("spatial", False, True, False,
                               "Use the full scene for spatial relations; left/right refer to the image unless stated otherwise.")
    return QAReasoningPlan("general", False, False, False,
                           "Answer only from clearly visible evidence in the selected scene.")


def temporal_frame_ids(start_frame: int, end_frame_exclusive: int, center_frame: int,
                       maximum: int = 5) -> list[int]:
    if maximum <= 0 or end_frame_exclusive <= start_frame:
        return []
    last = end_frame_exclusive - 1
    center = min(max(center_frame, start_frame), last)
    if maximum == 1 or last == start_frame:
        return [center]
    radius = min(center - start_frame, last - center)
    if radius > 0:
        low, high = center - radius, center + radius
    else:
        low, high = start_frame, last
    values = [round(low + index * (high - low) / (maximum - 1)) for index in range(maximum)]
    return list(dict.fromkeys(int(value) for value in values))
