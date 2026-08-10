from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


TIME_OR_DATE = re.compile(
    r"^(?:\d{1,2}[:h]\d{2}(?::\d{2})?(?:\s*[ap]m)?|"
    r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|"
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2})$",
    re.IGNORECASE,
)
BOILERPLATE = re.compile(
    r"\b(?:www\.|https?://|facebook|youtube|tiktok|hotline|đường dây nóng|"
    r"subscribe|đăng ký kênh|bản quyền|copyright)\b",
    re.IGNORECASE,
)
CHANNEL_WATERMARK = re.compile(
    r"^(?:htv|vtv|thvl|sctv|vnews|antv|qptv|đài truyền hình)\b",
    re.IGNORECASE,
)


def normalized_ocr(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return " ".join(text.split())


def recurrence_counts(descriptions: list[dict[str, Any]]) -> Counter[str]:
    """Count an OCR string at most once per scene."""
    counts: Counter[str] = Counter()
    for description in descriptions:
        lines = {normalized_ocr(item.get("text", "")).casefold()
                 for item in description.get("ocr_lines", []) if normalized_ocr(item.get("text", ""))}
        counts.update(lines)
    return counts


def filter_ocr_lines(lines: list[dict[str, Any]], recurrence: Counter[str] | None = None,
                     scene_count: int = 0) -> tuple[list[str], list[dict[str, str]]]:
    searchable: list[str] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    recurring_threshold = max(5, math.ceil(scene_count * 0.25))
    for item in lines:
        text = normalized_ocr(item.get("text", ""))
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        reason = None
        compact = re.sub(r"\s+", "", text)
        if TIME_OR_DATE.fullmatch(text):
            reason = "timestamp_or_date"
        elif len(compact) <= 2 and not any(char.isalpha() for char in compact):
            reason = "non_semantic_short_text"
        elif BOILERPLATE.search(text):
            reason = "channel_or_footer_boilerplate"
        elif (recurrence and scene_count and recurrence[key] >= recurring_threshold
              and CHANNEL_WATERMARK.search(text)):
            # Recurrence alone is insufficient: a programme title/logo can be
            # a valid OCR query (e.g. "60 giây"). Only discard recurring text
            # that also looks like a broadcaster watermark.
            reason = "repeated_channel_watermark"
        if reason:
            rejected.append({"text": text, "reason": reason,
                             "frame_label": str(item.get("frame_label", ""))})
        else:
            searchable.append(text)
    return searchable, rejected
